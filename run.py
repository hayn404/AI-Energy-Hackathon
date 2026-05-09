"""
Energy AI Hackathon 2026 — Complete Pipeline
Run: python run.py
"""

import time
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_pipeline import load_raw, fetch_weather, build_features, get_feature_cols
from src.model        import train as train_lgbm, predict as lgbm_predict, evaluate
from src.optimizer    import run_simulation
from src.baselines    import compute_baseline_a, compute_baseline_b
from src.evaluation   import (print_results, plot_march_week3, plot_forecast_sample,
                               plot_savings, plot_horizon_sensitivity, plot_soc_overview,
                               forecast_metrics)

DATA_PATH = 'data/ENERGY_Hackathon_DataSet.csv'

HORIZONS_TO_TEST = [4, 24, 48, 96]   # for extension analysis
PRIMARY_HORIZON  = 96


def main():
    t0 = time.time()

    # ── 1. LOAD & CLEAN DATA ─────────────────────────────────────────────────
    print("\n[1/8] Loading and cleaning data …")
    df = load_raw(DATA_PATH)
    print(f"  Loaded {len(df):,} rows  |  {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Missing sell prices filled: 8")
    print(f"  Balance error > 0.05 kW: {(df['balance_err'] > 0.05).sum()} timesteps")

    df_2024 = df[df['timestamp'].dt.year == 2024].reset_index(drop=True)
    df_2025 = df[df['timestamp'].dt.year == 2025].reset_index(drop=True)
    print(f"  Train (2024): {len(df_2024):,} rows  |  Test (2025): {len(df_2025):,} rows")

    # ── 2. FETCH WEATHER ─────────────────────────────────────────────────────
    print("\n[2/8] Fetching weather (Sondrio, Italy) …")
    weather_df = fetch_weather('data/weather_sondrio.csv')
    has_weather = len(weather_df) > 0

    # ── 3. FEATURE ENGINEERING ───────────────────────────────────────────────
    print("\n[3/8] Building features …")
    # Build features on the FULL dataset so lags span the 2024→2025 boundary
    df_full = build_features(df, weather_df if has_weather else None)

    df_full_2024 = df_full[df_full['timestamp'].dt.year == 2024].reset_index(drop=True)
    df_full_2025 = df_full[df_full['timestamp'].dt.year == 2025].reset_index(drop=True)

    feature_cols = get_feature_cols(df_full_2024)
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Drop rows with NaN in features (first ~7 days lost to lag_672)
    TRAIN_END = '2024-09-30 23:45:00'
    VAL_START = '2024-10-01'

    df_tr  = df_full_2024[df_full_2024['timestamp'] <= TRAIN_END].dropna(subset=feature_cols)
    df_val = df_full_2024[df_full_2024['timestamp'] >= VAL_START].dropna(subset=feature_cols)
    df_te  = df_full_2025.dropna(subset=feature_cols)

    print(f"  Train: {len(df_tr):,}  Val: {len(df_val):,}  Test (2025): {len(df_te):,}")

    # ── 4. TRAIN LGBM FORECASTER ─────────────────────────────────────────────
    MODEL_PATH = Path('results/lgbm_model.pkl')
    FEATURES_PATH = Path('results/feature_cols.pkl')
    print("\n[4/8] Training LightGBM forecaster …")
    t_train = time.time()
    model = train_lgbm(df_tr, df_val, feature_cols)
    print(f"  Training done in {time.time()-t_train:.1f}s  |  best_iteration={model.best_iteration_}")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"  Model saved to {MODEL_PATH}  |  Features saved to {FEATURES_PATH}")

    # Evaluate on 2025 test set
    # We need predictions for ALL 2025 rows (including those that have NaN lags)
    # Fill NaN lags with forward-fill so we have predictions everywhere
    df_te_filled = df_full_2025.copy()
    for c in feature_cols:
        if c in df_te_filled.columns:
            df_te_filled[c] = df_te_filled[c].ffill().bfill()
    df_te_filled = df_te_filled.dropna(subset=feature_cols)

    forecast_2025   = lgbm_predict(model, df_te_filled, feature_cols)
    y_true_2025     = df_te_filled['load_p'].values

    # Align forecast to the full 2025 index
    full_forecast_2025 = np.zeros(len(df_2025))
    valid_idx = df_te_filled.index.values  # original indices in df_full_2025
    for i, idx in enumerate(df_te_filled.index):
        full_forecast_2025[idx] = forecast_2025[i]
    # Fill any remaining zeros (beginning of year, NaN lags) with rolling mean
    for i in range(len(full_forecast_2025)):
        if full_forecast_2025[i] == 0:
            full_forecast_2025[i] = full_forecast_2025[max(0, i-1)] if i > 0 else df_2025['load_p'].mean()

    print("\n  Forecasting metrics on 2025 test set:")
    fc_met = evaluate(y_true_2025, forecast_2025, label='2025 test')
    fc_met_full = forecast_metrics(df_2025['load_p'].values, full_forecast_2025)

    # Validation metrics
    val_pred = lgbm_predict(model, df_val, feature_cols)
    print("  Forecasting metrics on Oct–Dec 2024 validation:")
    _ = evaluate(df_val['load_p'].values, val_pred, label='val Oct-Dec 2024')

    plot_forecast_sample(df_2025, full_forecast_2025, weeks=3)

    # ── 5. BASELINES ─────────────────────────────────────────────────────────
    print("\n[5/8] Computing baselines …")
    res_a = compute_baseline_a(df_2025)
    res_b = compute_baseline_b(df_2025)
    print(f"  Baseline B (no battery):   EUR {res_b['bill']:.2f}")
    print(f"  Baseline A (historical):   EUR {res_a['bill']:.2f}")

    # ── 6. LP-MPC CONTROLLER ─────────────────────────────────────────────────
    print(f"\n[6/8] Running LP-MPC controller (H={PRIMARY_HORIZON}) …")
    t_mpc = time.time()
    res_mpc = run_simulation(df_2025, full_forecast_2025, H=PRIMARY_HORIZON, desc='LP-MPC')
    print(f"  Done in {time.time()-t_mpc:.1f}s  |  Bill: EUR {res_mpc['bill']:.2f}")

    # Oracle controller: perfect load forecast
    print("\n  Running Oracle controller (perfect forecast) …")
    t_oracle = time.time()
    res_oracle = run_simulation(
        df_2025, df_2025['load_p'].values, H=PRIMARY_HORIZON, desc='Oracle'
    )
    print(f"  Done in {time.time()-t_oracle:.1f}s  |  Bill: EUR {res_oracle['bill']:.2f}")

    # ── 7. HORIZON SENSITIVITY ───────────────────────────────────────────────
    print("\n[7/8] Horizon sensitivity analysis …")
    horizon_results = []
    for H in HORIZONS_TO_TEST:
        if H == PRIMARY_HORIZON:
            horizon_results.append(dict(H=H, bill=res_mpc['bill']))
            continue
        t_h = time.time()
        r = run_simulation(df_2025, full_forecast_2025, H=H, desc=f'H={H}')
        elapsed = time.time() - t_h
        print(f"  H={H:3d} ({H*0.25:4.1f}h): EUR {r['bill']:.2f}  "
              f"saves EUR {res_a['bill']-r['bill']:.2f} vs A  ({elapsed:.1f}s)")
        horizon_results.append(dict(H=H, bill=r['bill'], time_s=elapsed))
    # Sort by H
    horizon_results.sort(key=lambda x: x['H'])

    # ── 8. PLOTS & RESULTS ────────────────────────────────────────────────────
    print("\n[8/8] Generating plots …")

    plot_march_week3(df_2025, res_mpc['P_battery'], res_mpc['P_grid'],
                     res_mpc['SoC'], label='LP-MPC')

    plot_march_week3(df_2025, res_oracle['P_battery'], res_oracle['P_grid'],
                     res_oracle['SoC'], label='Oracle')

    plot_savings(res_b['bill'], res_a['bill'], res_mpc['bill'], res_oracle['bill'])
    plot_horizon_sensitivity(horizon_results, res_a['bill'])
    plot_soc_overview(df_2025, res_mpc['SoC'], label='LP-MPC')

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print_results(
        bill_b=res_b['bill'],
        bill_a=res_a['bill'],
        bill_ours=res_mpc['bill'],
        bill_oracle=res_oracle['bill'],
        fc_metrics=fc_met,
        horizon_results=horizon_results,
    )

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    print("All outputs saved to: results/")


if __name__ == '__main__':
    main()
