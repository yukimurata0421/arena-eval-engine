import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure local src package is preferred when this script is executed directly.
_ROOT = Path(__file__).resolve().parents[4]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from arena.lib.paths import OUTPUT_DIR, ADSB_DAILY_SUMMARY, ADSB_DAILY_SUMMARY_V2, ADSB_SIGNAL_RANGE_SUMMARY
from arena.lib.phase_config import get_config
from arena.lib.config import get_quality_thresholds

from arena.log import get_script_logger


log = get_script_logger(__name__)
AUC_CSV = str(ADSB_DAILY_SUMMARY_V2 if ADSB_DAILY_SUMMARY_V2.exists() else ADSB_DAILY_SUMMARY)
SIG_CSV = str(ADSB_SIGNAL_RANGE_SUMMARY)
REPORT_IMG = str(Path(OUTPUT_DIR) / "integrated_master_log_report.png")


def generate_report():
    if not os.path.exists(AUC_CSV) or not os.path.exists(SIG_CSV):
        log.info("CSV not found.")
        return

    cfg = get_config()

    df_auc = pd.read_csv(AUC_CSV)
    df_sig = pd.read_csv(SIG_CSV)
    df_auc['date'] = pd.to_datetime(df_auc['date'])
    df_sig['date'] = pd.to_datetime(df_sig['date'])
    
    df = pd.merge(df_auc, df_sig, on='date', how='inner').sort_values('date')
    if df.empty:
        log.info("Skipping report because join data is empty.")
        return
    
    df = df[df['date'] >= pd.Timestamp(cfg.report_start_date)].copy()
    df_plot = df.copy()
    lowq_plot_mask = pd.Series(False, index=df_plot.index)
    min_auc, min_minutes = get_quality_thresholds()
    if 'minutes_covered' in df.columns:
        lowq = df[(df['minutes_covered'] < min_minutes) | (df['auc_n_used'] < min_auc)].copy()
        phase_dates = {pd.Timestamp(p['date']) for p in cfg.master_log_phases}
        retained_lowq = lowq[lowq['date'].isin(phase_dates)].copy()
        lowq_plot_mask = (df_plot['minutes_covered'] < min_minutes) | (df_plot['auc_n_used'] < min_auc)
        if not lowq.empty:
            log.info(f" Exclude low quality days: {len(lowq)} days (minutes<{min_minutes} or auc_n_used<{min_auc})")
            for _, r in lowq.tail(5).iterrows():
                log.info(f"    - {r['date'].date()} (minutes={int(r['minutes_covered'])}, auc_n_used={int(r['auc_n_used'])})")
        if not retained_lowq.empty:
            log.info(f" Retained for phase boundary date: {len(retained_lowq)} days")
            for _, r in retained_lowq.iterrows():
                log.info(f"    + {r['date'].date()} (minutes={int(r['minutes_covered'])}, auc_n_used={int(r['auc_n_used'])})")
        quality_ok = (df['minutes_covered'] >= min_minutes) & (df['auc_n_used'] >= min_auc)
        keep_phase_boundary = df['date'].isin(phase_dates)
        df = df[quality_ok | keep_phase_boundary].copy()
        if df.empty:
            log.info("There are no dates to draw after applying the quality threshold.")
            return

    phases = cfg.master_log_phases

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    ax1.plot(df_plot['date'], df_plot['auc_n_used'], color='#2c3e50', lw=2, label='Daily Packets (all merged days)')
    ax1.fill_between(df_plot['date'], df_plot['auc_n_used'], color='#2c3e50', alpha=0.1)
    if lowq_plot_mask.any():
        ax1.scatter(
            df_plot.loc[lowq_plot_mask, 'date'],
            df_plot.loc[lowq_plot_mask, 'auc_n_used'],
            marker='x',
            s=45,
            color='#d35400',
            alpha=0.85,
            label='Low-quality day'
        )
    ax1.set_ylabel("Quantity: Daily Packets (AUC)", fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(df_plot['date'], df_plot['sig_150_175'], color='#16a085', lw=2, label='Signal 150-175km (all merged days)')
    if lowq_plot_mask.any():
        ax2.scatter(
            df_plot.loc[lowq_plot_mask, 'date'],
            df_plot.loc[lowq_plot_mask, 'sig_150_175'],
            marker='x',
            s=45,
            color='#d35400',
            alpha=0.85,
            label='Low-quality day'
        )
    ax2.set_ylabel("Quality: Signal 150-175km (dBFS)", fontweight='bold')
    sig_min = df_plot['sig_150_175'].min()
    sig_max = df_plot['sig_150_175'].max()
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
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper left', fontsize=9)
    plt.suptitle("Tsuchiura ADS-B Station Master Log Analysis", fontsize=18, fontweight='bold')
    
    plt.savefig(REPORT_IMG, bbox_inches='tight', dpi=150)
    log.info(f" Report image saved: {REPORT_IMG}")

    log.info("\n" + "="*85)
    log.info(f"{'Phase configuration':<25} | {'Number of days':<5} | {'Average packets/day':<12} | {'Signal (dBFS)'}")
    log.info("-" * 85)

    for i, p in enumerate(phases):
        mask = (df['date'] >= pd.Timestamp(p['date']))
        if i+1 < len(phases):
            mask &= (df['date'] < pd.Timestamp(phases[i+1]['date']))
        
        phase_df = df[mask]
        days = len(phase_df)
        
        if days == 0:
            log.info(f"{p['name']:<25} | {days:>4}d | {'---':>12} | {'---'}")
            continue

        avg_auc = phase_df['auc_n_used'].mean()
        avg_sig = phase_df['sig_150_175'].mean()
        
        sig_str = f"{avg_sig:>8.2f}" if not pd.isna(avg_sig) else "---"
        log.info(f"{p['name']:<25} | {days:>4}d | {int(avg_auc):>12,} | {sig_str:>10}")
    log.info("="*85)

if __name__ == "__main__":
    generate_report()
