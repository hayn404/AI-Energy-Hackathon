"""Metrics, result printing, and all required plots."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

RESULTS = Path('results')
RESULTS.mkdir(exist_ok=True)

# ── Metrics ───────────────────────────────────────────────────────────────────

def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    nrmse = float(rmse / y_true.mean() * 100)
    return dict(RMSE=rmse, MAE=mae, NRMSE=nrmse)


def print_results(bill_b: float, bill_a: float, bill_ours: float,
                  bill_oracle: float, fc_metrics: dict,
                  horizon_results: list[dict]) -> dict:
    """Print the full results table and return a dict for JSON export."""
    sav_ours_vs_a   = bill_a - bill_ours
    sav_ours_pct_a  = sav_ours_vs_a / bill_a * 100
    sav_ours_vs_b   = bill_b - bill_ours
    sav_ours_pct_b  = sav_ours_vs_b / bill_b * 100

    sav_oracle_vs_a = bill_a - bill_oracle
    oracle_gap      = bill_ours - bill_oracle
    oracle_gap_pct  = oracle_gap / max(sav_oracle_vs_a, 1e-9) * 100

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Controller':<30} {'Bill (EUR)':>12} {'vs A (EUR)':>12} {'vs A (%)':>10}")
    print("-" * 68)
    print(f"{'Baseline B (no battery)':<30} {bill_b:>12.2f} {bill_b - bill_a:>+12.2f} {(bill_b - bill_a)/bill_a*100:>+10.1f}")
    print(f"{'Baseline A (historical)':<30} {bill_a:>12.2f} {'reference':>12} {'reference':>10}")
    print(f"{'Our LP-MPC controller':<30} {bill_ours:>12.2f} {-sav_ours_vs_a:>+12.2f} {-sav_ours_pct_a:>+10.1f}")
    print(f"{'Oracle (perfect forecast)':<30} {bill_oracle:>12.2f} {-sav_oracle_vs_a:>+12.2f} {-sav_oracle_vs_a/bill_a*100:>+10.1f}")

    print(f"\nOracle gap: EUR {oracle_gap:.2f}  ({oracle_gap_pct:.1f}% of oracle savings)")

    print("\n--- Forecasting Metrics (2025 test set) ---")
    print(f"  RMSE  = {fc_metrics['RMSE']:.4f} kW")
    print(f"  MAE   = {fc_metrics['MAE']:.4f} kW")
    print(f"  NRMSE = {fc_metrics['NRMSE']:.2f}%")

    if horizon_results:
        print("\n--- Horizon Sensitivity (Extension) ---")
        print(f"  {'H':>6}  {'Horizon':>10}  {'Bill (EUR)':>12}  {'vs A (EUR)':>12}  {'vs A (%)':>10}")
        for r in horizon_results:
            h = r['H']
            b = r['bill']
            sa = bill_a - b
            print(f"  {h:>6}  {h*0.25:>8.1f}h  {b:>12.2f}  {sa:>+12.2f}  {sa/bill_a*100:>+10.1f}")

    print("=" * 60)

    out = dict(
        baseline_b_bill=bill_b,
        baseline_a_bill=bill_a,
        our_bill=bill_ours,
        oracle_bill=bill_oracle,
        savings_vs_a_eur=sav_ours_vs_a,
        savings_vs_a_pct=sav_ours_pct_a,
        savings_vs_b_eur=sav_ours_vs_b,
        savings_vs_b_pct=sav_ours_pct_b,
        oracle_gap_eur=oracle_gap,
        oracle_gap_pct=oracle_gap_pct,
        forecast=fc_metrics,
        horizon_sensitivity=horizon_results,
    )
    with open(RESULTS / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved metrics to {RESULTS/'metrics.json'}")
    return out


# ── March Week 3 Dispatch Plot ─────────────────────────────────────────────────

def plot_march_week3(df_2025: pd.DataFrame,
                     P_battery: np.ndarray,
                     P_grid:    np.ndarray,
                     SoC:       np.ndarray,
                     label:     str = 'LP-MPC') -> None:
    """5-panel dispatch plot for March 15–21, 2025 (Week 3 of March)."""
    mask = (
        (df_2025['timestamp'] >= '2025-03-15') &
        (df_2025['timestamp'] <  '2025-03-22')
    )
    idx  = df_2025.index[mask]
    ts   = df_2025.loc[mask, 'timestamp'].values

    load = df_2025.loc[mask, 'load_p'].values
    pv   = df_2025.loc[mask, 'pv_p'].values
    pbat = P_battery[idx - idx[0]]
    pgrd = P_grid[idx - idx[0]]
    soc  = SoC[idx - idx[0]] * 100   # %

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'March Week 3 (15–21 Mar 2025) — {label}', fontsize=13, fontweight='bold')

    axes[0].fill_between(ts, load, alpha=0.7, color='steelblue', label='Load')
    axes[0].set_ylabel('Load (kW)');  axes[0].legend(loc='upper right'); axes[0].set_ylim(bottom=0)

    axes[1].fill_between(ts, pv, alpha=0.7, color='gold', label='PV')
    axes[1].set_ylabel('PV (kW)');    axes[1].legend(loc='upper right'); axes[1].set_ylim(bottom=0)

    axes[2].fill_between(ts, np.where(pbat < 0, pbat, 0), 0,
                         alpha=0.7, color='tomato',   label='Charging')
    axes[2].fill_between(ts, np.where(pbat > 0, pbat, 0), 0,
                         alpha=0.7, color='limegreen', label='Discharging')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set_ylabel('P_battery (kW)'); axes[2].legend(loc='upper right')

    axes[3].fill_between(ts, np.where(pgrd > 0, pgrd, 0), 0,
                         alpha=0.6, color='gray',   label='Import')
    axes[3].fill_between(ts, np.where(pgrd < 0, pgrd, 0), 0,
                         alpha=0.6, color='orange', label='Export')
    axes[3].axhline(0, color='k', lw=0.5)
    axes[3].set_ylabel('P_grid (kW)'); axes[3].legend(loc='upper right')

    axes[4].fill_between(ts, soc, alpha=0.6, color='mediumseagreen', label='SoC')
    axes[4].set_ylim(0, 100); axes[4].set_ylabel('SoC (%)'); axes[4].legend(loc='upper right')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()
    path = RESULTS / f'march_week3_{label.replace(" ", "_")}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Forecast quality plot ──────────────────────────────────────────────────────

def plot_forecast_sample(df_2025: pd.DataFrame,
                         forecast: np.ndarray,
                         weeks: int = 2) -> None:
    """Plot first `weeks` weeks of 2025: actual vs forecast."""
    n = weeks * 7 * 96
    ts   = df_2025['timestamp'].values[:n]
    act  = df_2025['load_p'].values[:n]
    pred = forecast[:n]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts, act,  lw=0.8, color='steelblue', label='Actual load')
    ax.plot(ts, pred, lw=0.8, color='tomato',    label='Forecast', alpha=0.8)
    ax.set_title(f'Load Forecast vs Actual — first {weeks} weeks of 2025')
    ax.set_ylabel('Load (kW)'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS / 'forecast_vs_actual.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Savings comparison bar chart ───────────────────────────────────────────────

def plot_savings(bill_b: float, bill_a: float,
                 bill_ours: float, bill_oracle: float) -> None:
    labels = ['Baseline B\n(no battery)', 'Baseline A\n(historical)',
              'LP-MPC\n(our)', 'Oracle\n(perfect)']
    values = [bill_b, bill_a, bill_ours, bill_oracle]
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#27ae60']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5,
                f'€{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Annual Electricity Bill (€)')
    ax.set_title('2025 Electricity Bill Comparison')
    ax.set_ylim(0, bill_b * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = RESULTS / 'savings_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Horizon sensitivity plot ───────────────────────────────────────────────────

def plot_horizon_sensitivity(horizon_results: list[dict], bill_a: float) -> None:
    if not horizon_results:
        return
    Hs     = [r['H']    for r in horizon_results]
    bills  = [r['bill'] for r in horizon_results]
    savings= [bill_a - b for b in bills]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h * 0.25 for h in Hs], savings, 'o-', color='steelblue', lw=2, ms=7)
    for h, s in zip(Hs, savings):
        ax.annotate(f'H={h}\n€{s:.0f}', (h * 0.25, s),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('Savings vs Baseline A (€)')
    ax.set_title('Controller Savings vs Forecast Horizon')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS / 'horizon_sensitivity.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── SoC trajectory overview ───────────────────────────────────────────────────

def plot_soc_overview(df_2025: pd.DataFrame, SoC: np.ndarray, label: str = 'LP-MPC') -> None:
    ts  = df_2025['timestamp'].values
    soc = SoC[:len(ts)] * 100

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(ts, soc, alpha=0.5, color='mediumseagreen')
    ax.plot(ts, soc, lw=0.4, color='darkgreen')
    ax.set_ylim(0, 100); ax.set_ylabel('SoC (%)'); ax.set_title(f'Battery SoC — {label} 2025')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = RESULTS / f'soc_overview_{label.replace(" ","_")}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
