"""
Inference script for the judges' 2026 test data.

Input:  data/Test_Data.xlsx  — columns: Timestamps, Load
Output: results/test_predictions.csv  + results/test_results.json

Strategy:
  - Lag features are computed by appending the test rows to the 2025 history,
    so every lag references real past observations (no imputation needed).
  - pv_p is set to 0 (correct: midnight / early morning, 1 Jan 2026).
  - sell_price is forward-filled from the last known 2025 value.
  - buy_price is computed from the Italian TOU tariff function.
  - Weather is fetched from the Open-Meteo historical API (same source as training).
  - The LP-MPC controller runs for each of the 7 test timesteps with H=96.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.data_pipeline import (
    load_raw, fetch_weather, build_features, get_feature_cols,
    tou_price, ETA, C_BAT, P_BAT_MAX, P_GRID_MAX, SOC_INIT, DT
)
from src.optimizer import run_simulation

RESULTS = Path('results')
RESULTS.mkdir(exist_ok=True)


# ── 1. Load historical context (2025) ────────────────────────────────────────
print("[1] Loading 2025 historical data for lag context …")
df_hist = load_raw('data/ENERGY_Hackathon_DataSet.csv')
df_2025 = df_hist[df_hist['timestamp'].dt.year == 2025].reset_index(drop=True)
last_sell_price = df_2025['Selling_price_eur_kwh'].dropna().iloc[-1]
print(f"  Last known sell price: {last_sell_price:.4f} €/kWh")


# ── 2. Load and normalise test data ──────────────────────────────────────────
print("[2] Loading test data …")
df_test_raw = pd.read_excel('data/Test_Data.xlsx')
df_test_raw.columns = df_test_raw.columns.str.strip()

# Rename to match pipeline column names
df_test = pd.DataFrame({
    'timestamp':               pd.to_datetime(df_test_raw['Timestamps']),
    'load_p':                  df_test_raw['Load'].astype(float),
    'pv_p':                    0.0,           # nighttime — no solar generation
    'grid_p':                  0.0,           # unknown; will be computed by controller
    'battery_p':               0.0,           # unknown; will be decided by controller
    'Selling_price_eur_kwh':   last_sell_price,
    'balance_err':             0.0,
})
df_test['buy_price'] = df_test['timestamp'].apply(tou_price)

print(f"  Test rows: {len(df_test)}")
print(f"  Period:    {df_test['timestamp'].iloc[0]}  →  {df_test['timestamp'].iloc[-1]}")
print(f"  Load:      min={df_test['load_p'].min():.2f}  max={df_test['load_p'].max():.2f}  "
      f"mean={df_test['load_p'].mean():.2f} kW")


# ── 3. Fetch 2026 weather ─────────────────────────────────────────────────────
print("[3] Fetching 2026 weather …")
weather_df = fetch_weather('data/weather_sondrio.csv')

# Extend weather cache to cover 2026 if needed
if len(weather_df) == 0 or weather_df['timestamp'].max() < df_test['timestamp'].max():
    print("  Cache doesn't cover 2026 — fetching fresh …")
    try:
        import openmeteo_requests, requests_cache
        from retry_requests import retry
        from src.data_pipeline import LAT, LON

        cache_session = requests_cache.CachedSession('.omcache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": "2026-01-01", "end_date": "2026-01-07",
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
        df_w2026 = pd.DataFrame({
            "timestamp":            ts,
            "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
            "apparent_temperature": hourly.Variables(1).ValuesAsNumpy(),
            "cloud_cover":          hourly.Variables(2).ValuesAsNumpy(),
            "precipitation":        hourly.Variables(3).ValuesAsNumpy(),
            "wind_speed_10m":       hourly.Variables(4).ValuesAsNumpy(),
        })
        df_w2026 = df_w2026.set_index('timestamp').resample('15min').interpolate('linear').reset_index()
        weather_df = pd.concat([weather_df, df_w2026], ignore_index=True).drop_duplicates('timestamp')
        weather_df.to_csv('data/weather_sondrio.csv', index=False)
        print(f"  Fetched and cached 2026 weather ({len(df_w2026)} rows)")
    except Exception as e:
        print(f"  Weather fetch failed ({e}) — will proceed without 2026 weather")


# ── 4. Build features using combined 2025 + test data ─────────────────────────
print("[4] Building features …")
# Concatenate so lag features for 2026 rows reference real 2025 values
df_combined = pd.concat([df_2025, df_test], ignore_index=True)
df_combined_feat = build_features(df_combined, weather_df)

# Extract only the test rows (last N rows)
n_test = len(df_test)
df_test_feat = df_combined_feat.tail(n_test).reset_index(drop=True)

# Load saved model and feature list
model        = joblib.load('results/lgbm_model.pkl')
feature_cols = joblib.load('results/feature_cols.pkl')

# Fill any remaining NaNs in features (should be minimal — lags covered by 2025)
df_test_filled = df_test_feat.copy()
for c in feature_cols:
    if c in df_test_filled.columns:
        df_test_filled[c] = df_test_filled[c].ffill().bfill()

missing_feats = [c for c in feature_cols if c not in df_test_filled.columns]
if missing_feats:
    print(f"  WARNING: missing features (will be set to 0): {missing_feats}")
    for c in missing_feats:
        df_test_filled[c] = 0.0

print(f"  NaN remaining in features: {df_test_filled[feature_cols].isnull().sum().sum()}")


# ── 5. Forecast load ──────────────────────────────────────────────────────────
print("[5] Forecasting load …")
load_forecast = np.clip(model.predict(df_test_filled[feature_cols]), 0, None)

print(f"\n  {'Timestamp':<22} {'Actual (kW)':>12} {'Forecast (kW)':>14} {'Error (kW)':>11}")
print("  " + "-" * 63)
for i, row in df_test_feat.iterrows():
    ts  = df_test['timestamp'].iloc[i]
    act = df_test['load_p'].iloc[i]
    fct = load_forecast[i]
    print(f"  {str(ts):<22} {act:>12.3f} {fct:>14.3f} {fct-act:>+11.3f}")

actual = df_test['load_p'].values
rmse  = float(np.sqrt(np.mean((actual - load_forecast) ** 2)))
mae   = float(np.mean(np.abs(actual - load_forecast)))
nrmse = float(rmse / actual.mean() * 100) if actual.mean() > 0 else 0.0
print(f"\n  RMSE={rmse:.4f} kW  MAE={mae:.4f} kW  NRMSE={nrmse:.2f}%")


# ── 6. Run LP-MPC controller ──────────────────────────────────────────────────
print("\n[6] Running LP-MPC controller on test window …")

# Use the last known SoC from 2025 end-of-year simulation, default to SOC_INIT
# (SoC at start of 2026 is unknown — use 50%)
soc_start = SOC_INIT
print(f"  Starting SoC: {soc_start*100:.1f}%")

# Build a mini test DataFrame compatible with run_simulation
df_ctrl = df_test.copy()
df_ctrl['pv_p']                  = 0.0   # nighttime
df_ctrl['Selling_price_eur_kwh'] = last_sell_price

res = run_simulation(df_ctrl, load_forecast, H=min(96, len(df_ctrl)), desc='Test')

print(f"\n  {'Timestamp':<22} {'Load':>7} {'PV':>6} {'P_bat':>8} {'P_grid':>8} {'SoC%':>7} {'Action'}")
print("  " + "-" * 72)
for i in range(len(df_test)):
    ts    = df_test['timestamp'].iloc[i]
    load  = df_test['load_p'].iloc[i]
    pbat  = res['P_battery'][i]
    pgrd  = res['P_grid'][i]
    soc   = res['SoC'][i+1] * 100
    action = 'DISCHARGE' if pbat > 0.01 else ('CHARGE' if pbat < -0.01 else 'IDLE')
    print(f"  {str(ts):<22} {load:>7.2f} {0:>6.2f} {pbat:>8.3f} {pgrd:>8.3f} {soc:>6.1f}%  {action}")

print(f"\n  Bill for test window: EUR {res['bill']:.4f}")


# ── 7. Save results ───────────────────────────────────────────────────────────
print("\n[7] Saving results …")

out_df = df_test[['timestamp', 'load_p', 'buy_price']].copy()
out_df['pv_p']           = 0.0
out_df['load_forecast']  = load_forecast
out_df['P_battery']      = res['P_battery']
out_df['P_grid']         = res['P_grid']
out_df['SoC']            = res['SoC'][1:]
out_df['sell_price']     = last_sell_price
out_df['forecast_error'] = load_forecast - actual

out_df.to_csv(RESULTS / 'test_predictions.csv', index=False)
print(f"  Saved: results/test_predictions.csv")

summary = {
    'test_period': {
        'start': str(df_test['timestamp'].iloc[0]),
        'end':   str(df_test['timestamp'].iloc[-1]),
        'n_timesteps': len(df_test),
    },
    'forecast_metrics': {
        'RMSE': round(rmse, 4),
        'MAE':  round(mae, 4),
        'NRMSE': round(nrmse, 2),
    },
    'controller': {
        'bill_eur':     round(res['bill'], 6),
        'soc_start':    round(soc_start * 100, 1),
        'soc_end':      round(float(res['SoC'][len(df_test)]) * 100, 1),
        'total_discharge_kwh': round(float(res['P_battery'][res['P_battery'] > 0].sum() * DT), 4),
        'total_charge_kwh':    round(float(abs(res['P_battery'][res['P_battery'] < 0].sum()) * DT), 4),
    },
    'model_info': {
        'features': len(feature_cols),
        'model':    'LightGBM retrained on full 2024',
        'NRMSE_2025_test': 46.14,
    }
}
with open(RESULTS / 'test_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: results/test_results.json")
print("\nDone.")
