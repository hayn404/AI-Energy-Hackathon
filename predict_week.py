"""
Full first-week-of-2026 prediction and battery dispatch.

Strategy
--------
* Lag features for 2026 are resolved using:
    - 2025 actual data    (lag_96 to lag_1344 for most of the week)
    - 2026 actual test    (first 7 rows from Test_Data.xlsx)
    - Recursive predictions (short lags once we move past the 1.5h actual window)
* PV proxy: same calendar week from 2025 (identical season, same location).
* sell_price: forward-filled from last known 2025 value.
* buy_price: computed from Italian TOU tariff.
* Weather: fetched/cached from Open-Meteo for 2026-01-01 to 01-07.
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
    load_raw, fetch_weather, build_features, get_feature_cols,
    tou_price, DT, SOC_INIT
)
from src.optimizer import run_simulation

RESULTS = Path('results')


# ── 1. Load historical data ───────────────────────────────────────────────────
print("[1] Loading historical data …")
df_hist = load_raw('data/ENERGY_Hackathon_DataSet.csv')
df_2025 = df_hist[df_hist['timestamp'].dt.year == 2025].reset_index(drop=True)

last_sell_price = df_2025['Selling_price_eur_kwh'].dropna().iloc[-1]

# PV proxy: first 7 days of 2025 → use as 2026 proxy (same season)
pv_proxy_2025 = df_2025[
    (df_2025['timestamp'] >= '2025-01-01') &
    (df_2025['timestamp'] <  '2025-01-08')
]['pv_p'].values   # 672 values


# ── 2. Load test data (actual first 1.5 hours) ────────────────────────────────
print("[2] Loading test data (actual 2026-01-01 00:00–01:30) …")
df_test_raw = pd.read_excel('data/Test_Data.xlsx')
df_test_raw.columns = df_test_raw.columns.str.strip()
actual_ts   = pd.to_datetime(df_test_raw['Timestamps']).values
actual_load = df_test_raw['Load'].astype(float).values
n_actual    = len(actual_load)   # 7
print(f"  Actual rows: {n_actual}  (load = {actual_load.mean():.2f} kW constant)")


# ── 3. Build the full 672-row week skeleton ───────────────────────────────────
print("[3] Building 2026 week skeleton …")
week_ts = pd.date_range('2026-01-01 00:00', periods=672, freq='15min')

df_week = pd.DataFrame({
    'timestamp':             week_ts,
    'load_p':                np.nan,          # to be filled by prediction
    'pv_p':                  pv_proxy_2025,   # seasonal proxy from 2025
    'grid_p':                0.0,
    'battery_p':             0.0,
    'Selling_price_eur_kwh': last_sell_price,
    'balance_err':           0.0,
})
df_week['buy_price'] = df_week['timestamp'].apply(tou_price)

# Seed with known actuals (first n_actual rows)
df_week.loc[:n_actual - 1, 'load_p'] = actual_load


# ── 4. Fetch weather for 2026 week ────────────────────────────────────────────
print("[4] Loading weather …")
weather_df = fetch_weather('data/weather_sondrio.csv')


# ── 5. Recursive prediction ───────────────────────────────────────────────────
print("[5] Recursive load prediction for all 672 timesteps …")
model        = joblib.load('results/lgbm_model_2026.pkl')  # trained on 2024+2025
feature_cols = joblib.load('results/feature_cols.pkl')

# We predict in chunks to avoid rebuilding the full feature matrix every step.
# Chunk size = 96 (1 day): after each day, rebuild features for the next day.
# Within a chunk, short lags reference previously predicted values in df_week.

load_predictions = actual_load.tolist()   # start with actuals

for i in range(n_actual, 672):
    # Fill the current row's load_p with a temporary best guess
    # (use the prediction so far; lag_1 etc. will reference this)
    df_week.loc[i, 'load_p'] = np.nan

    # Concatenate 2025 history + week-so-far (with predictions filled in)
    df_week_so_far = df_week.copy()
    df_week_so_far.loc[:i - 1, 'load_p'] = load_predictions  # fill in predictions

    df_combined = pd.concat([df_2025, df_week_so_far.iloc[:i + 1]], ignore_index=True)
    df_feat = build_features(df_combined, weather_df)

    # Take the last row (current timestep)
    row = df_feat.tail(1).copy()
    for c in feature_cols:
        if c in row.columns:
            row[c] = row[c].ffill().bfill()
        else:
            row[c] = 0.0

    pred = float(np.clip(model.predict(row[feature_cols])[0], 0, None))
    load_predictions.append(pred)

    if i % 96 == 0:
        day = (i // 96) + 1
        print(f"  Day {day}/7 predicted … "
              f"mean={np.mean(load_predictions[i-95:i+1]):.2f} kW")

load_predictions = np.array(load_predictions)
df_week['load_p'] = load_predictions
print(f"  Full week predicted. Mean={load_predictions.mean():.2f} kW  "
      f"Min={load_predictions.min():.2f}  Max={load_predictions.max():.2f}")


# ── 6. Forecast quality on the known actual rows ──────────────────────────────
pred_actual = load_predictions[:n_actual]
rmse  = float(np.sqrt(np.mean((actual_load - pred_actual) ** 2)))
mae   = float(np.mean(np.abs(actual_load - pred_actual)))
nrmse = float(rmse / actual_load.mean() * 100)
print(f"\n  Forecast vs actual (first {n_actual} rows):")
print(f"  RMSE={rmse:.4f} kW  MAE={mae:.4f} kW  NRMSE={nrmse:.2f}%")


# ── 7. LP-MPC controller over the full week ───────────────────────────────────
print("\n[6] Running LP-MPC controller over full week …")
res = run_simulation(df_week, load_predictions, H=96, desc='Week-2026')
print(f"  Bill for week: EUR {res['bill']:.4f}")
print(f"  SoC start: {res['SoC'][0]*100:.1f}%  →  end: {res['SoC'][-1]*100:.1f}%")


# ── 8. Daily summary ──────────────────────────────────────────────────────────
print("\n  Daily summary:")
print(f"  {'Day':<12} {'Load (kWh)':>12} {'PV (kWh)':>10} {'Battery net':>13} {'Grid (kWh)':>12}")
print("  " + "-" * 65)
for d in range(7):
    s, e = d * 96, (d + 1) * 96
    ts_day     = week_ts[s].strftime('%a %d %b')
    load_kwh   = float(load_predictions[s:e].sum() * DT)
    pv_kwh     = float(pv_proxy_2025[s:e].sum() * DT)
    bat_kwh    = float(res['P_battery'][s:e].sum() * DT)   # +ve = net discharge
    grid_kwh   = float(res['P_grid'][s:e].sum() * DT)      # +ve = net import
    print(f"  {ts_day:<12} {load_kwh:>12.2f} {pv_kwh:>10.2f} {bat_kwh:>13.2f} {grid_kwh:>12.2f}")


# ── 9. Save outputs ───────────────────────────────────────────────────────────
print("\n[7] Saving outputs …")
out_df = df_week[['timestamp', 'pv_p', 'buy_price']].copy()
out_df['load_forecast'] = load_predictions
out_df['P_battery']     = res['P_battery']
out_df['P_grid']        = res['P_grid']
out_df['SoC']           = res['SoC'][1:]
out_df['sell_price']    = last_sell_price
out_df['is_actual']     = ([True] * n_actual + [False] * (672 - n_actual))

out_df.to_csv(RESULTS / 'week_2026_predictions.csv', index=False)
print(f"  Saved: results/week_2026_predictions.csv")

# Clean forecast-only CSV (judges' submission format)
forecast_df = pd.DataFrame({
    'timestamp':     week_ts,
    'load_forecast': load_predictions.round(4),
})
forecast_df.to_csv(RESULTS / 'week_2026_forecast.csv', index=False)
print(f"  Saved: results/week_2026_forecast.csv")

summary = {
    'period': {'start': '2026-01-01 00:00', 'end': '2026-01-07 23:45', 'timesteps': 672},
    'actual_rows_available': n_actual,
    'forecast_on_actuals': {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'NRMSE': round(nrmse, 2)},
    'controller': {
        'bill_eur':    round(res['bill'], 4),
        'soc_start':   round(float(res['SoC'][0]) * 100, 1),
        'soc_end':     round(float(res['SoC'][-1]) * 100, 1),
    },
    'week_totals': {
        'load_kwh':   round(float(load_predictions.sum() * DT), 2),
        'pv_kwh':     round(float(pv_proxy_2025.sum() * DT), 2),
        'net_import_kwh': round(float(res['P_grid'][res['P_grid'] > 0].sum() * DT), 2),
        'net_export_kwh': round(float(abs(res['P_grid'][res['P_grid'] < 0].sum()) * DT), 2),
    }
}
with open(RESULTS / 'week_2026_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: results/week_2026_results.json")


# ── 10. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(16, 13), sharex=True)
fig.suptitle('LP-MPC Dispatch — First Week of 2026 (Jan 1–7)', fontsize=13, fontweight='bold')

axes[0].plot(week_ts, load_predictions, lw=0.9, color='steelblue', label='Load forecast')
axes[0].scatter(week_ts[:n_actual], actual_load, s=20, color='red', zorder=5, label='Actual (given)')
axes[0].set_ylabel('Load (kW)'); axes[0].legend(loc='upper right'); axes[0].set_ylim(bottom=0)

axes[1].fill_between(week_ts, pv_proxy_2025, alpha=0.7, color='gold', label='PV (2025 proxy)')
axes[1].set_ylabel('PV (kW)'); axes[1].legend(loc='upper right'); axes[1].set_ylim(bottom=0)

pbat = res['P_battery']
axes[2].fill_between(week_ts, np.where(pbat < 0, pbat, 0), 0, alpha=0.7, color='tomato',   label='Charging')
axes[2].fill_between(week_ts, np.where(pbat > 0, pbat, 0), 0, alpha=0.7, color='limegreen', label='Discharging')
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_ylabel('P_battery (kW)'); axes[2].legend(loc='upper right')

pgrd = res['P_grid']
axes[3].fill_between(week_ts, np.where(pgrd > 0, pgrd, 0), 0, alpha=0.6, color='gray',   label='Import')
axes[3].fill_between(week_ts, np.where(pgrd < 0, pgrd, 0), 0, alpha=0.6, color='orange', label='Export')
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_ylabel('P_grid (kW)'); axes[3].legend(loc='upper right')

soc_pct = res['SoC'][:672] * 100
axes[4].fill_between(week_ts, soc_pct, alpha=0.6, color='mediumseagreen', label='SoC')
axes[4].set_ylim(0, 100); axes[4].set_ylabel('SoC (%)'); axes[4].legend(loc='upper right')

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.grid(True, alpha=0.3)

plt.xticks(rotation=20)
plt.tight_layout()
path = RESULTS / 'week_2026_dispatch.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")
print("\nDone.")
