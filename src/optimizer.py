"""LP-MPC rolling-horizon battery dispatch controller."""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tqdm import tqdm

from src.data_pipeline import ETA, C_BAT, P_BAT_MAX, P_GRID_MAX, SOC_INIT, DT

SOC_MIN, SOC_MAX = 0.05, 0.95   # 5 % buffer on each side


# ── Precomputed LP structure ──────────────────────────────────────────────────

def _build_Aeq(H: int) -> np.ndarray:
    """Build the fixed equality-constraint matrix for a given horizon H.

    Variable layout  (total n = 4H + H+1 = 5H+1):
      [P_bat_c(0..H-1) | P_bat_d(0..H-1) | P_grid+(0..H-1) | P_grid-(0..H-1) | SoC(0..H)]
    Rows: H energy-balance + H SoC-dynamics + 1 initial-SoC = 2H+1
    """
    n   = 4 * H + (H + 1)
    neq = 2 * H + 1
    A   = np.zeros((neq, n))

    ic  = lambda t: t           # P_bat_c
    id_ = lambda t: H + t       # P_bat_d
    igp = lambda t: 2 * H + t   # P_grid+
    igm = lambda t: 3 * H + t   # P_grid-
    is_ = lambda t: 4 * H + t   # SoC

    for t in range(H):
        # Energy balance: P_grid+ - P_grid- + P_bat_d - P_bat_c = L - PV
        A[t, igp(t)] =  1;  A[t, igm(t)] = -1
        A[t, id_(t)] =  1;  A[t, ic(t)]  = -1

        # SoC dynamics: SoC[t+1] - SoC[t] - P_bat_c*eta*dt/C + P_bat_d/eta*dt/C = 0
        A[H + t, is_(t + 1)] =  1
        A[H + t, is_(t)]     = -1
        A[H + t, ic(t)]      = -ETA * DT / C_BAT
        A[H + t, id_(t)]     =  (1.0 / ETA) * DT / C_BAT

    # Initial SoC: SoC[0] = soc_init
    A[2 * H, is_(0)] = 1
    return A


def _bounds(H: int) -> list:
    return ([(0, P_BAT_MAX)] * H +        # P_bat_c
            [(0, P_BAT_MAX)] * H +        # P_bat_d
            [(0, P_GRID_MAX)] * H +       # P_grid+
            [(0, P_GRID_MAX)] * H +       # P_grid-
            [(SOC_MIN, SOC_MAX)] * (H + 1))  # SoC


# ── Single MPC step ───────────────────────────────────────────────────────────

def mpc_step(soc: float,
             load_fcst: np.ndarray,
             pv:        np.ndarray,
             buy:       np.ndarray,
             sell:      np.ndarray,
             H:         int,
             Aeq:       np.ndarray,
             bounds_h:  list) -> tuple[float, float]:
    """Solve one LP horizon window.

    Returns (P_battery_action, new_SoC).
    P_battery: positive = discharging, negative = charging (PDF convention).
    """
    H_actual = min(H, len(load_fcst))

    # Cost vector: minimise import cost, maximise export revenue
    c = np.zeros(4 * H + (H + 1))
    for t in range(H_actual):
        c[2 * H + t] =  buy[t]  * DT   # P_grid+ cost
        c[3 * H + t] = -sell[t] * DT   # P_grid- revenue (negative = income)

    # RHS: energy balance + SoC dynamics + initial SoC
    beq = np.zeros(2 * H + 1)
    for t in range(H_actual):
        beq[t] = float(load_fcst[t]) - float(pv[t])   # energy balance RHS
    # beq[H..2H-1] = 0 (SoC dynamics)
    beq[2 * H] = soc                                   # initial SoC

    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds_h, method='highs')

    if res.status != 0:
        # Fallback: relax grid export limit (load spike may exceed grid cap)
        bounds_relaxed = list(bounds_h)
        for t in range(H):
            # Relax P_grid+ upper bound for the energy balance row
            bounds_relaxed[2 * H + t] = (0, max(P_GRID_MAX, float(load_fcst[t]) + 1))
        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds_relaxed, method='highs')

    if res.status != 0:
        return 0.0, soc  # complete fallback: do nothing

    x = res.x
    P_bat = x[H] - x[0]   # P_bat_d[0] - P_bat_c[0]  (positive = discharge)
    new_soc = x[4 * H + 1]
    return float(P_bat), float(np.clip(new_soc, SOC_MIN, SOC_MAX))


# ── Rolling horizon simulation ────────────────────────────────────────────────

def run_simulation(df: pd.DataFrame,
                   load_forecast: np.ndarray,
                   H: int = 96,
                   desc: str = 'MPC') -> dict:
    """Run the full rolling-horizon MPC over df (2025 test set).

    Parameters
    ----------
    df            : 2025 DataFrame with pv_p, buy_price, Selling_price_eur_kwh, load_p
    load_forecast : (N,) array of 1-step-ahead load predictions (pre-computed)
    H             : look-ahead horizon in timesteps

    Returns dict with arrays and scalar bill.
    """
    N   = len(df)
    pv  = df['pv_p'].values
    buy = df['buy_price'].values
    sell = df['Selling_price_eur_kwh'].ffill().values
    load_true = df['load_p'].values

    Aeq      = _build_Aeq(H)
    bounds_h = _bounds(H)

    P_battery = np.zeros(N)
    P_grid    = np.zeros(N)
    soc_traj  = np.zeros(N + 1)
    soc_traj[0] = SOC_INIT
    bill = 0.0

    for t in tqdm(range(N), desc=f'  {desc} H={H}', leave=False, ncols=80):
        end      = min(t + H, N)
        h_actual = end - t

        P_bat, new_soc = mpc_step(
            soc      = soc_traj[t],
            load_fcst= load_forecast[t:end],
            pv       = pv[t:end],
            buy      = buy[t:end],
            sell     = sell[t:end],
            H        = H,
            Aeq      = Aeq,
            bounds_h = bounds_h,
        )

        # Physical feasibility clamp given actual SoC
        if P_bat > 0:   # discharging
            max_dis = (soc_traj[t] - SOC_MIN) * C_BAT * ETA / DT
            P_bat = float(np.clip(P_bat, 0, max(0, max_dis)))
        else:           # charging
            max_chg = (SOC_MAX - soc_traj[t]) * C_BAT / (ETA * DT)
            P_bat = float(np.clip(P_bat, -max(0, max_chg), 0))

        P_battery[t] = P_bat

        # Actual SoC update (use actual executed power, not planned)
        if P_bat < 0:
            soc_traj[t + 1] = soc_traj[t] + abs(P_bat) * ETA * DT / C_BAT
        else:
            soc_traj[t + 1] = soc_traj[t] - P_bat / ETA * DT / C_BAT
        soc_traj[t + 1] = float(np.clip(soc_traj[t + 1], 0.0, 1.0))

        # Actual grid flow uses real load (not forecast)
        pg = load_true[t] - pv[t] - P_bat
        pg = float(np.clip(pg, -P_GRID_MAX, P_GRID_MAX))
        P_grid[t] = pg

        # Billing
        if pg > 0:
            bill += pg * buy[t] * DT
        else:
            bill -= abs(pg) * sell[t] * DT

    return dict(
        P_battery=P_battery,
        P_grid=P_grid,
        SoC=soc_traj,
        bill=bill,
        H=H,
    )
