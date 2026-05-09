"""Baseline A (historical controller) and Baseline B (zero-intelligence)."""

import numpy as np
import pandas as pd
from src.data_pipeline import P_GRID_MAX, DT


def compute_baseline_b(df: pd.DataFrame) -> dict:
    """Baseline B: PV serves load directly; no battery; surplus exported.

    P_grid = load - pv  (positive = import, negative = export)
    """
    sell = df['Selling_price_eur_kwh'].ffill().values
    buy  = df['buy_price'].values
    load = df['load_p'].values
    pv   = df['pv_p'].values

    pg_raw = load - pv
    pg     = np.clip(pg_raw, -P_GRID_MAX, P_GRID_MAX)

    import_mask  = pg > 0
    export_mask  = pg < 0
    import_cost  = (pg[import_mask] * buy[import_mask] * DT).sum()
    export_rev   = (np.abs(pg[export_mask]) * sell[export_mask] * DT).sum()
    bill = import_cost - export_rev

    return dict(bill=bill, P_grid=pg, label='Baseline B (no battery)')


def compute_baseline_a(df: pd.DataFrame) -> dict:
    """Baseline A: use the actual recorded battery dispatch (p_battery_kw).

    The energy balance is satisfied per timestep in the dataset, so
    we can bill directly from the recorded grid_p column.
    """
    sell = df['Selling_price_eur_kwh'].ffill().values
    buy  = df['buy_price'].values
    pg   = df['grid_p'].values   # actual grid power from on-site sensors
    pb   = df['battery_p'].values

    import_mask = pg > 0
    export_mask = pg < 0
    import_cost = (pg[import_mask] * buy[import_mask] * DT).sum()
    export_rev  = (np.abs(pg[export_mask]) * sell[export_mask] * DT).sum()
    bill = import_cost - export_rev

    return dict(bill=bill, P_grid=pg, P_battery=pb, label='Baseline A (historical)')
