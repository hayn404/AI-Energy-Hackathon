"""
2026 Q1 load forecasting + LP-MPC dispatch — judges' 2nd dataset (different house).

Dataset: data/2nd_DataSet_All_Features.xlsx
  Columns: timestamp, load_p, pv_p, battery_p, grid_p, Selling_price_eur_kwh
  Period:  2024-11-25 to 2026-03-31

Sign convention in this dataset (OPPOSITE to original house):
  grid_p    < 0  =  importing from grid (costs money)
  grid_p    > 0  =  exporting to grid   (earns revenue)
  battery_p < 0  =  charging
  battery_p > 0  =  discharging

Task: train on dataset up to 2026-02-28, forecast and dispatch March 2026.
Outputs:
  results/march_2026_forecast.csv      timestamp, actual, forecast, error
  results/march_2026_controller.csv    full dispatch: P_battery, P_grid, SoC
  results/march_2026_results.json      all metrics
  results/march_2026_forecast.png      forecast vs actual
  results/march_2026_dispatch.png      5-panel controller dispatch
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from src.data_pipeline import (
    fetch_weather, build_features, get_feature_cols, tou_price,
    DT, LAT, LON, SOC_INIT
)
from src.model import train as train_lgbm, retrain_full, predict as lgbm_predict, evaluate
from src.optimizer import run_simulation

RESULTS = Path('results')
RESULTS.mkdir(exist_ok=True)

# System parameters for this house (inferred from data percentiles)
P_BAT_MAX  = 10.0   # kW  (p95 of |battery_p| = 9.4 kW)
P_GRID_MAX = 20.0   # kW  (p95 of |grid_p|    = 18.4 kW)
C_BAT      = 16.0   # kWh (hackathon spec — no SoC column to estimate from)
SOC_MIN    = 0.05
SOC_MAX    = 0.95


# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("[1] Loading 2nd dataset (new house, all features) …")

df_raw = pd.read_excel('data/2nd_DataSet_All_Features.xlsx')
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
df_raw = df_raw.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

# Add TOU buy price (not in dataset)
df_raw['buy_price'] = df_raw['timestamp'].apply(tou_price)

# Verify energy balance: load = pv + battery_p - grid_p  (new sign convention)
df_raw['balance_err'] = (df_raw['load_p'] - df_raw['pv_p']
                         - df_raw['battery_p'] + df_raw['grid_p']).abs()

print(f"  Rows:  {len(df_raw)}")
print(f"  Range: {df_raw['timestamp'].min().date()} → {df_raw['timestamp'].max().date()}")
print(f"  Energy balance error — mean={df_raw['balance_err'].mean():.4f}  "
      f"max={df_raw['balance_err'].max():.4f} kW")
print(f"  Sell price — mean={df_raw['Selling_price_eur_kwh'].mean():.4f}  "
      f"min={df_raw['Selling_price_eur_kwh'].min():.4f}  "
      f"max={df_raw['Selling_price_eur_kwh'].max():.4f} €/kWh")

# Splits
TRAIN_END = '2026-02-28 23:45'
TEST_START = '2026-03-01'
VAL_START  = '2026-02-01'

df_combined   = df_raw.copy()
df_train_all  = df_combined[df_combined['timestamp'] <= TRAIN_END]
df_test_raw   = df_combined[df_combined['timestamp'] >= TEST_START].reset_index(drop=True)
print(f"  Train (≤ Feb 2026): {len(df_train_all):,}   Test (Mar 2026): {len(df_test_raw):,}")


# ── 2. Weather ────────────────────────────────────────────────────────────────
print("\n[2] Loading weather …")
weather_df = fetch_weather('data/weather_sondrio.csv')

if len(weather_df) == 0 or weather_df['timestamp'].max() < pd.Timestamp('2026-03-31'):
    print("  Fetching 2026 Q1 weather …")
    try:
        import openmeteo_requests, requests_cache
        from retry_requests import retry

        cs = requests_cache.CachedSession('.omcache', expire_after=-1)
        rs = retry(cs, retries=5, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=rs)
        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": "2026-01-01", "end_date": "2026-03-31",
            "hourly": ["temperature_2m", "apparent_temperature",
                       "cloud_cover", "precipitation", "wind_speed_10m"],
            "timezone": "Europe/Rome",
        }
        resp = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )[0]
        hourly = resp.Hourly()
        ts = pd.date_range(
            start=pd.Timestamp(hourly.Time(), unit="s"),
            periods=hourly.Variables(0).ValuesAsNumpy().shape[0],
            freq="1h",
        )
        df_w = pd.DataFrame({
            "timestamp":            ts,
            "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
            "apparent_temperature": hourly.Variables(1).ValuesAsNumpy(),
            "cloud_cover":          hourly.Variables(2).ValuesAsNumpy(),
            "precipitation":        hourly.Variables(3).ValuesAsNumpy(),
            "wind_speed_10m":       hourly.Variables(4).ValuesAsNumpy(),
        })
        df_w = df_w.set_index('timestamp').resample('15min').interpolate('linear').reset_index()
        weather_df = (pd.concat([weather_df, df_w], ignore_index=True)
                      .drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True))
        weather_df.to_csv('data/weather_sondrio.csv', index=False)
        print(f"  Cached ({len(df_w)} rows)")
    except Exception as e:
        print(f"  Fetch failed ({e}) — proceeding without 2026 weather")


# ── 3. Feature engineering ────────────────────────────────────────────────────
print("\n[3] Building features …")
df_feat_all = build_features(df_combined, weather_df)
feature_cols = get_feature_cols(df_feat_all[df_feat_all['timestamp'] <= TRAIN_END])
print(f"  Features: {len(feature_cols)}")

df_feat_train = df_feat_all[df_feat_all['timestamp'] <= TRAIN_END].dropna(subset=feature_cols)
df_feat_test  = df_feat_all[df_feat_all['timestamp'] >= TEST_START].copy()
for c in feature_cols:
    if c in df_feat_test.columns:
        df_feat_test[c] = df_feat_test[c].ffill().bfill()
    else:
        df_feat_test[c] = 0.0
df_feat_test = df_feat_test.dropna(subset=feature_cols)

print(f"  Train rows: {len(df_feat_train):,}   Test rows: {len(df_feat_test):,}")


# ── 4. Train model (two-phase) ────────────────────────────────────────────────
print("\n[4] Training LightGBM (two-phase) …")
df_ph1_tr  = df_feat_train[df_feat_train['timestamp'] <  VAL_START]
df_ph1_val = df_feat_train[df_feat_train['timestamp'] >= VAL_START]
print(f"  Phase-1: train={len(df_ph1_tr):,}  val={len(df_ph1_val):,} (Feb 2026)")

model_probe = train_lgbm(df_ph1_tr, df_ph1_val, feature_cols)
best_iter = model_probe.best_iteration_
print(f"  Best iteration: {best_iter}")

model = retrain_full(df_feat_train, feature_cols, best_iter)
joblib.dump(model, RESULTS / 'lgbm_model_march2026.pkl')
joblib.dump(feature_cols, RESULTS / 'feature_cols_march2026.pkl')
print(f"  Saved: results/lgbm_model_march2026.pkl")


# ── 5. Forecast March 2026 ────────────────────────────────────────────────────
print("\n[5] Forecasting March 2026 …")
forecast = lgbm_predict(model, df_feat_test, feature_cols)
actual   = df_feat_test['load_p'].values
ts_test  = pd.to_datetime(df_feat_test['timestamp'].values)

print(f"  Actual   — mean={actual.mean():.3f}  min={actual.min():.3f}  max={actual.max():.3f} kW")
print(f"  Forecast — mean={forecast.mean():.3f}  min={forecast.min():.3f}  max={forecast.max():.3f} kW")
print()
metrics = evaluate(actual, forecast, label='March 2026 (new house)')


# ── 6. Controller inputs ──────────────────────────────────────────────────────
print("\n[6] Preparing controller inputs …")

# Use actual sell prices from the dataset
df_ctrl = df_test_raw[['timestamp', 'load_p', 'pv_p',
                        'battery_p', 'grid_p',
                        'buy_price', 'Selling_price_eur_kwh']].copy().reset_index(drop=True)

# run_simulation expects Selling_price_eur_kwh and buy_price columns — both present
print(f"  Sell price — mean={df_ctrl['Selling_price_eur_kwh'].mean():.4f}  "
      f"min={df_ctrl['Selling_price_eur_kwh'].min():.4f}  "
      f"max={df_ctrl['Selling_price_eur_kwh'].max():.4f} €/kWh")
print(f"  Buy price  — mean={df_ctrl['buy_price'].mean():.4f} €/kWh (TOU tariff)")


# ── 7. Baseline A — historical controller (actual battery dispatch) ────────────
print("\n[7] Computing Baseline A (historical controller, actual battery_p / grid_p) …")
# Sign convention in this dataset: grid_p < 0 = import, grid_p > 0 = export
grid_a    = df_ctrl['grid_p'].values
buy_arr   = df_ctrl['buy_price'].values
sell_arr  = df_ctrl['Selling_price_eur_kwh'].values
imp_mask  = grid_a < 0   # importing
exp_mask  = grid_a > 0   # exporting
bill_a    = (np.abs(grid_a[imp_mask]) * buy_arr[imp_mask] * DT).sum() \
          - (grid_a[exp_mask]         * sell_arr[exp_mask] * DT).sum()
print(f"  Baseline A (historical) bill: €{bill_a:.4f}")


# ── 8. Baseline B — no battery ────────────────────────────────────────────────
print("\n[8] Computing Baseline B (no battery) …")
# grid_p_nobat = pv - load  (positive = export, negative = import — same new convention)
pv_arr   = df_ctrl['pv_p'].values
load_arr = df_ctrl['load_p'].values
pg_b     = np.clip(pv_arr - load_arr, -P_GRID_MAX, P_GRID_MAX)
imp_b    = pg_b < 0
exp_b    = pg_b > 0
bill_b   = (np.abs(pg_b[imp_b]) * buy_arr[imp_b]  * DT).sum() \
         - (pg_b[exp_b]         * sell_arr[exp_b] * DT).sum()
print(f"  Baseline B (no battery) bill:  €{bill_b:.4f}")


# ── 9. LP-MPC + Oracle ───────────────────────────────────────────────────────
# run_simulation computes billing internally with positive-grid=import convention,
# using load_true and pv from the DataFrame and Selling_price_eur_kwh for sell price.
# We pass a modified copy with P_GRID_MAX and P_BAT_MAX updated for this house.
import src.optimizer as _opt
import src.data_pipeline as _dp

_orig_pgmax = _dp.P_GRID_MAX
_orig_pbmax = _dp.P_BAT_MAX
_opt_orig_pgmax = _opt.P_GRID_MAX if hasattr(_opt, 'P_GRID_MAX') else None

# Temporarily patch module-level constants
_dp.P_GRID_MAX = P_GRID_MAX
_dp.P_BAT_MAX  = P_BAT_MAX
# Rebuild bounds with new limits by monkey-patching run_simulation's _bounds call
from src.optimizer import _build_Aeq, _bounds as _orig_bounds_fn, mpc_step
from scipy.optimize import linprog
from tqdm import tqdm

SOC_MN, SOC_MX = SOC_MIN, SOC_MAX

def _run_sim_custom(df, load_forecast, H=96, desc='MPC'):
    """run_simulation with custom P_BAT_MAX / P_GRID_MAX for this house."""
    N        = len(df)
    pv       = df['pv_p'].values
    buy      = df['buy_price'].values
    sell     = df['Selling_price_eur_kwh'].ffill().values
    load_true= df['load_p'].values

    Aeq = _build_Aeq(H)
    bounds_h = ([(0, P_BAT_MAX)] * H +
                [(0, P_BAT_MAX)] * H +
                [(0, P_GRID_MAX)] * H +
                [(0, P_GRID_MAX)] * H +
                [(SOC_MN, SOC_MX)] * (H + 1))

    P_battery = np.zeros(N)
    P_grid    = np.zeros(N)
    soc_traj  = np.zeros(N + 1)
    soc_traj[0] = SOC_INIT
    bill = 0.0
    ETA  = _dp.ETA
    C    = C_BAT

    for t in tqdm(range(N), desc=f'  {desc} H={H}', leave=False, ncols=80):
        end      = min(t + H, N)
        h_actual = end - t

        soc = soc_traj[t]
        lf  = load_forecast[t:end]
        pv_ = pv[t:end]
        bu_ = buy[t:end]
        se_ = sell[t:end]

        c = np.zeros(4 * H + (H + 1))
        for i in range(h_actual):
            c[2 * H + i] =  bu_[i] * DT
            c[3 * H + i] = -se_[i] * DT

        beq = np.zeros(2 * H + 1)
        for i in range(h_actual):
            beq[i] = float(lf[i]) - float(pv_[i])
        beq[2 * H] = soc

        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds_h, method='highs')

        if res.status != 0:
            bounds_rlx = list(bounds_h)
            for i in range(h_actual):
                bounds_rlx[2 * H + i] = (0, max(P_GRID_MAX, float(lf[i]) + 1))
            res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds_rlx, method='highs')

        if res.status != 0:
            P_bat, new_soc = 0.0, soc
        else:
            x = res.x
            P_bat   = float(x[H] - x[0])
            new_soc = float(np.clip(x[4 * H + 1], SOC_MN, SOC_MX))

        # Physical clamp
        if P_bat > 0:
            max_dis = (soc - SOC_MN) * C * ETA / DT
            P_bat   = float(np.clip(P_bat, 0, max(0, max_dis)))
        else:
            max_chg = (SOC_MX - soc) * C / (ETA * DT)
            P_bat   = float(np.clip(P_bat, -max(0, max_chg), 0))

        P_battery[t] = P_bat

        if P_bat < 0:
            soc_traj[t+1] = soc + abs(P_bat) * ETA * DT / C
        else:
            soc_traj[t+1] = soc - P_bat / ETA * DT / C
        soc_traj[t+1] = float(np.clip(soc_traj[t+1], 0.0, 1.0))

        # Billing with actual load
        pg = load_true[t] - pv[t] - P_bat
        pg = float(np.clip(pg, -P_GRID_MAX, P_GRID_MAX))
        P_grid[t] = pg

        if pg > 0:
            bill += pg * buy[t] * DT
        else:
            bill -= abs(pg) * sell[t] * DT

    return dict(P_battery=P_battery, P_grid=P_grid, SoC=soc_traj, bill=bill, H=H)


print("\n[9] Running LP-MPC controller (H=96) …")
res_mpc = _run_sim_custom(df_ctrl, forecast, H=96, desc='LP-MPC March-2026')
print(f"  LP-MPC bill: €{res_mpc['bill']:.4f}")

print("\n  Running Oracle (perfect forecast) …")
res_oracle = _run_sim_custom(df_ctrl, actual, H=96, desc='Oracle March-2026')
print(f"  Oracle bill: €{res_oracle['bill']:.4f}")

# Savings
sav_vs_b_eur = bill_b - res_mpc['bill']
sav_vs_b_pct = sav_vs_b_eur / bill_b * 100
sav_vs_a_eur = bill_a - res_mpc['bill']
sav_vs_a_pct = sav_vs_a_eur / bill_a * 100
oracle_gap   = res_mpc['bill'] - res_oracle['bill']


# ── 10. Full summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTS SUMMARY — March 2026 (New House, Full Features)")
print("=" * 65)
print(f"\n  Forecast Quality:")
print(f"    RMSE   = {metrics['RMSE']:.4f} kW")
print(f"    MAE    = {metrics['MAE']:.4f} kW")
print(f"    NRMSE  = {metrics['NRMSE']:.2f}%")
print(f"\n  Bill Comparison (March 2026):")
print(f"    Baseline A (historical controller)  : €{bill_a:.4f}")
print(f"    Baseline B (no battery)             : €{bill_b:.4f}")
print(f"    LP-MPC (our controller)             : €{res_mpc['bill']:.4f}")
print(f"    Oracle (perfect forecast)           : €{res_oracle['bill']:.4f}")
print(f"\n  Savings vs Baseline A : €{sav_vs_a_eur:.4f}  ({sav_vs_a_pct:.2f}%)")
print(f"  Savings vs Baseline B : €{sav_vs_b_eur:.4f}  ({sav_vs_b_pct:.2f}%)")
print(f"  Oracle gap            : €{oracle_gap:.4f}  (cost of forecast error)")
print(f"\n  Energy Totals (March):")
pg_mpc = res_mpc['P_grid']
print(f"    Load      : {actual.sum()*DT:.2f} kWh")
print(f"    PV        : {pv_arr.sum()*DT:.2f} kWh")
print(f"    Import    : {pg_mpc[pg_mpc>0].sum()*DT:.2f} kWh")
print(f"    Export    : {abs(pg_mpc[pg_mpc<0].sum())*DT:.2f} kWh")
pb = res_mpc['P_battery']
print(f"    Charged   : {abs(pb[pb<0].sum())*DT:.2f} kWh")
print(f"    Discharged: {pb[pb>0].sum()*DT:.2f} kWh")
print(f"    SoC end   : {res_mpc['SoC'][-1]*100:.1f}%")
print("=" * 65)


# ── 11. Daily breakdown ───────────────────────────────────────────────────────
print(f"\n  {'Day':<14} {'Load':>10} {'PV':>10} {'Bat.net':>10} {'Grid':>10} {'Bill':>10}")
print("  " + "─" * 60)
for d in range(31):
    s, e = d * 96, (d + 1) * 96
    if e > len(actual):
        break
    day    = pd.Timestamp(ts_test[s]).strftime('%a %d %b')
    lk     = actual[s:e].sum() * DT
    pk     = pv_arr[s:e].sum() * DT
    bk     = pb[s:e].sum() * DT
    pg_d   = pg_mpc[s:e]
    bu_d   = buy_arr[s:e]
    se_d   = sell_arr[s:e]
    db     = (pg_d[pg_d>0] * bu_d[pg_d>0] * DT).sum() \
           - (abs(pg_d[pg_d<0]) * se_d[pg_d<0] * DT).sum()
    print(f"  {day:<14} {lk:>10.2f} {pk:>10.2f} {bk:>10.2f} {pg_d.sum()*DT:>10.2f} {db:>10.4f}")


# ── 12. Save outputs ──────────────────────────────────────────────────────────
print("\n[10] Saving outputs …")

pd.DataFrame({
    'timestamp':     ts_test,
    'actual_load':   actual.round(4),
    'forecast_load': forecast.round(4),
    'error':         (forecast - actual).round(4),
}).to_csv(RESULTS / 'march_2026_forecast.csv', index=False)
print("  Saved: results/march_2026_forecast.csv")

out_ctrl = df_ctrl[['timestamp', 'load_p', 'pv_p', 'buy_price', 'Selling_price_eur_kwh']].copy()
out_ctrl['load_forecast'] = forecast.round(4)
out_ctrl['P_battery']     = res_mpc['P_battery'].round(4)
out_ctrl['P_grid']        = res_mpc['P_grid'].round(4)
out_ctrl['SoC']           = res_mpc['SoC'][1:].round(4)
out_ctrl.to_csv(RESULTS / 'march_2026_controller.csv', index=False)
print("  Saved: results/march_2026_controller.csv")

summary = {
    'house':        'New house (2nd dataset, all features)',
    'period':       'March 2026 (2026-03-01 to 2026-03-31)',
    'train_period': f'{df_feat_train["timestamp"].min().date()} to {df_feat_train["timestamp"].max().date()}',
    'test_rows':    int(len(df_feat_test)),
    'best_iteration': int(best_iter),
    'features':     int(len(feature_cols)),
    'system_params': {
        'P_BAT_MAX_kW':  P_BAT_MAX,
        'P_GRID_MAX_kW': P_GRID_MAX,
        'C_BAT_kWh':     C_BAT,
        'note': 'P_BAT_MAX/P_GRID_MAX inferred from data p95; C_BAT from hackathon spec',
    },
    'forecast_metrics': {
        'RMSE':  round(float(metrics['RMSE']), 4),
        'MAE':   round(float(metrics['MAE']),  4),
        'NRMSE': round(float(metrics['NRMSE']), 2),
    },
    'controller': {
        'baseline_a_bill_eur':  round(bill_a,               4),
        'baseline_b_bill_eur':  round(bill_b,               4),
        'lp_mpc_bill_eur':      round(res_mpc['bill'],      4),
        'oracle_bill_eur':      round(res_oracle['bill'],   4),
        'savings_vs_a_eur':     round(sav_vs_a_eur,         4),
        'savings_vs_a_pct':     round(sav_vs_a_pct,         2),
        'savings_vs_b_eur':     round(sav_vs_b_eur,         4),
        'savings_vs_b_pct':     round(sav_vs_b_pct,         2),
        'oracle_gap_eur':       round(oracle_gap,           4),
        'soc_start_pct':        round(SOC_INIT * 100,       1),
        'soc_end_pct':          round(float(res_mpc['SoC'][-1]) * 100, 1),
    },
    'energy_totals_kwh': {
        'load':      round(float(actual.sum() * DT), 2),
        'pv':        round(float(pv_arr.sum() * DT), 2),
        'import':    round(float(pg_mpc[pg_mpc > 0].sum() * DT), 2),
        'export':    round(float(abs(pg_mpc[pg_mpc < 0].sum()) * DT), 2),
        'charge':    round(float(abs(pb[pb < 0].sum()) * DT), 2),
        'discharge': round(float(pb[pb > 0].sum() * DT), 2),
    },
}
with open(RESULTS / 'march_2026_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved: results/march_2026_results.json")


# ── 13. Plots ─────────────────────────────────────────────────────────────────
# Forecast vs actual
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle(
    f'Load Forecast vs Actual — March 2026 (New House)\n'
    f'RMSE={metrics["RMSE"]:.4f} kW   MAE={metrics["MAE"]:.4f} kW   NRMSE={metrics["NRMSE"]:.2f}%',
    fontsize=12, fontweight='bold'
)
axes[0].plot(ts_test, actual,   lw=0.7, color='steelblue', label='Actual load')
axes[0].plot(ts_test, forecast, lw=0.7, color='tomato', alpha=0.85, label='Forecast')
axes[0].set_ylabel('Load (kW)'); axes[0].legend(); axes[0].set_ylim(bottom=0)
err = forecast - actual
axes[1].fill_between(ts_test, err, 0, where=err>0, alpha=0.6, color='tomato',    label='Over-forecast')
axes[1].fill_between(ts_test, err, 0, where=err<0, alpha=0.6, color='steelblue', label='Under-forecast')
axes[1].axhline(0, color='k', lw=0.5); axes[1].set_ylabel('Error (kW)'); axes[1].legend()
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.grid(True, alpha=0.3)
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(RESULTS / 'march_2026_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/march_2026_forecast.png")

# Dispatch (5 panels)
fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
fig.suptitle(
    f'LP-MPC Dispatch — March 2026 (New House)\n'
    f'Baseline A: €{bill_a:.2f}  |  Baseline B: €{bill_b:.2f}  |  '
    f'LP-MPC: €{res_mpc["bill"]:.2f}  |  Savings vs A: €{sav_vs_a_eur:.2f} ({sav_vs_a_pct:.1f}%)',
    fontsize=11, fontweight='bold'
)
axes[0].plot(ts_test, actual,   lw=0.8, color='steelblue', label='Actual load')
axes[0].plot(ts_test, forecast, lw=0.8, color='tomato', alpha=0.7, label='Forecast', ls='--')
axes[0].set_ylabel('Load (kW)'); axes[0].legend(); axes[0].set_ylim(bottom=0)

axes[1].fill_between(ts_test, pv_arr, alpha=0.7, color='gold', label='PV generation')
axes[1].set_ylabel('PV (kW)'); axes[1].legend(); axes[1].set_ylim(bottom=0)

axes[2].fill_between(ts_test, np.where(pb<0, pb, 0), 0, alpha=0.7, color='tomato',    label='Charging')
axes[2].fill_between(ts_test, np.where(pb>0, pb, 0), 0, alpha=0.7, color='limegreen', label='Discharging')
axes[2].axhline(0, color='k', lw=0.5); axes[2].set_ylabel('P_battery (kW)'); axes[2].legend()

axes[3].fill_between(ts_test, np.where(pg_mpc>0, pg_mpc, 0), 0, alpha=0.6, color='gray',   label='Import')
axes[3].fill_between(ts_test, np.where(pg_mpc<0, pg_mpc, 0), 0, alpha=0.6, color='orange', label='Export')
axes[3].axhline(0, color='k', lw=0.5); axes[3].set_ylabel('P_grid (kW)'); axes[3].legend()

soc_pct = res_mpc['SoC'][:len(ts_test)] * 100
axes[4].fill_between(ts_test, soc_pct, alpha=0.6, color='mediumseagreen', label='SoC')
axes[4].set_ylim(0, 100); axes[4].set_ylabel('SoC (%)'); axes[4].legend()

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.grid(True, alpha=0.3)
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(RESULTS / 'march_2026_dispatch.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/march_2026_dispatch.png")

print("\nDone.")
