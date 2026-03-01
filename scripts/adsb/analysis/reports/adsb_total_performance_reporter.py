import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from arena.lib.paths import OUTPUT_DIR, ADSB_DAILY_SUMMARY, ADSB_DAILY_SUMMARY_V2, ADSB_SIGNAL_RANGE_SUMMARY
from arena.lib.phase_config import get_config

AUC_CSV = str(ADSB_DAILY_SUMMARY_V2 if ADSB_DAILY_SUMMARY_V2.exists() else ADSB_DAILY_SUMMARY)
SIG_CSV = str(ADSB_SIGNAL_RANGE_SUMMARY)
REPORT_IMG = str(Path(OUTPUT_DIR) / "tsuchiura_master_log_report.png")


def generate_report():
    if not os.path.exists(AUC_CSV) or not os.path.exists(SIG_CSV):
        print(" CSV not found。")
        return

    cfg = get_config()

    df_auc = pd.read_csv(AUC_CSV)
    df_sig = pd.read_csv(SIG_CSV)
    df_auc['date'] = pd.to_datetime(df_auc['date'])
    df_sig['date'] = pd.to_datetime(df_sig['date'])

    df = pd.merge(df_auc, df_sig, on='date', how='inner').sort_values('date')
    if df.empty:
        print("  Merged data is empty; skipping report.")
        return

    df = df[df['date'] >= pd.Timestamp(cfg.report_start_date)].copy()
    df = df[df['date'] < df['date'].max()]

    phases = cfg.master_log_phases

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    ax1.plot(df['date'], df['auc_n_used'], color='#2c3e50', lw=2)
    ax1.fill_between(df['date'], df['auc_n_used'], color='#2c3e50', alpha=0.1)
    ax1.set_ylabel("Quantity: Daily Packets (AUC)", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(df['date'], df['sig_150_175'], color='#16a085', lw=2)
    ax2.set_ylabel("Quality: Signal 150-175km (dBFS)", fontweight='bold')
    sig_min = df['sig_150_175'].min()
    sig_max = df['sig_150_175'].max()
    if pd.notna(sig_min) and pd.notna(sig_max):
        ax2.set_ylim(sig_min - 3, sig_max + 8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    for i, p in enumerate(phases):
        d = pd.Timestamp(p['date'])
        for ax in [ax1, ax2]:
            ax.axvline(d, color=p['color'], linestyle='--', lw=1.5, alpha=0.7)

        y_pos = ax1.get_ylim()[1] - (i % 3) * (ax1.get_ylim()[1] * 0.1)
        ax1.text(d, y_pos, f" {p['name']}", color=p['color'],
                 fontweight='bold', va='top', ha='left', rotation=30, fontsize=9)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.suptitle("Tsuchiura ADS-B Station Master Log Analysis", fontsize=18, fontweight='bold')

    plt.savefig(REPORT_IMG, bbox_inches='tight', dpi=150)
    print(f" Saved report image: {REPORT_IMG}")

    print("\n" + "="*85)
    print(f"{'Phase Configuration':<25} | {'Days':<5} | {'Avg Pkts/Day':<12} | {'Signal(dBFS)'}")
    print("-" * 85)

    for i, p in enumerate(phases):
        mask = (df['date'] >= pd.Timestamp(p['date']))
        if i+1 < len(phases):
            mask &= (df['date'] < pd.Timestamp(phases[i+1]['date']))

        phase_df = df[mask]
        days = len(phase_df)

        if days == 0:
            print(f"{p['name']:<25} | {days:>4}d | {'---':>12} | {'---'}")
            continue

        avg_auc = phase_df['auc_n_used'].mean()
        avg_sig = phase_df['sig_150_175'].mean()

        sig_str = f"{avg_sig:>8.2f}" if not pd.isna(avg_sig) else "---"
        print(f"{p['name']:<25} | {days:>4}d | {int(avg_auc):>12,} | {sig_str:>10}")
    print("="*85)

if __name__ == "__main__":
    generate_report()
