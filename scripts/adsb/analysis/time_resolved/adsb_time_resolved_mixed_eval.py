import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# statsmodels の収束警告のみ抑制
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")


from arena.lib.paths import OUTPUT_DIR
from arena.log import get_script_logger

log = get_script_logger(__name__)

CSV_PATH = os.path.join(str(OUTPUT_DIR), "time_resolved", "adsb_timebin_summary.csv")
OUTPUT_DIR = str(Path(OUTPUT_DIR) / "performance")

def run_mixed_analysis():
    if not os.path.exists(CSV_PATH):
        log.info("❌ CSV が見つかりません。先にアグリゲータを実行してください。")
        return

    df = pd.read_csv(CSV_PATH)
    df['date_factor'] = df['date'].astype(str)
    
    df['traffic_proxy'] = df['traffic_proxy'].replace(0, 1)

    log.info(f">>> 混合モデル（GEE）解析を開始（サンプル数: {len(df)}）")

    formula = "auc_sum ~ post + C(time_bin) + np.log(traffic_proxy)"
    
    try:
        model = smf.gee(
            formula, 
            data=df, 
            groups=df['date_factor'], 
            family=sm.families.NegativeBinomial(),
            cov_struct=sm.cov_struct.Exchangeable()
        ).fit()

        log.info("\n" + "="*60)
        log.info("      ADS-B 時間分解パフォーマンスレポート (GEE-NB)")
        log.info("="*60)
        log.info(model.summary().tables[1])
        
        if 'post' not in model.params:
            log.info("  [WARN] モデルパラメータに 'post' が存在しません。"
                  " データに pre/post の変動がない可能性があります。")
            return
        post_coeff = model.params['post']
        p_val = model.pvalues['post']
        improvement = (np.exp(post_coeff) - 1) * 100

        log.info(f"\n✅ 全体改善（純粋なハードウェア効果）: {improvement:+.2f} %")
        log.info(f"✅ 有意性 (P値)                    : {p_val:.10e}")
        
        if p_val < 0.05:
            log.info("\n判定: 統計的に有意な改善が確認されました。")
            log.info("時間帯や交通密度を考慮しても有意差が残ります。")
        else:
            log.info("\n判定: 統計的に有意な変化は検出されませんでした。")
        log.info("="*60)

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df, x='time_bin', y='auc_sum', hue='post', palette="Set2")
        plt.title("AUC by Time Bin: Before vs After (Tsuchiura Station)", fontsize=14)
        plt.xlabel("Time Bin (JST)", fontsize=12)
        plt.ylabel("AUC (Aggregate n_used)", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "time_resolved_performance.png")
        plt.savefig(save_path)
        log.info(f"\n可視化レポートを保存しました: {save_path}")

    except Exception as e:
        import traceback
        log.info(f"❌ 解析中にエラーが発生しました: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_mixed_analysis()
