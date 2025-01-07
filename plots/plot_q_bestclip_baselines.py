import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ========== USER DEFINED VARIABLES ==========
ENV_NAME = "Ant-v5"
METHODS = ["maxentirl", "rkl"]  # You can add other methods (e.g. maxentirl, etc.)
BASELINES = {
    "gail": "/home/viel/SOAR-IL/logs/Ant-v5/exp-16/gail/2025_01_05_15_56_35/progress.csv",
    # "bc":   "/path/to/baseline_bc.csv",
}
# Where all processed CSV files are located
PROCESSED_DATA_DIR = "processed_data"

SHOW_CONFIDENCE = False  # Whether to show ±std (or ±stderr) shading

# ========== HELPER FUNCTIONS ==========

def load_processed_data_for_method(env_name: str, method_name: str, data_dir: str) -> pd.DataFrame:
    """
    Finds a CSV in data_dir matching something like:
      {env_name}_exp-*_{method_name}_data.csv
    Returns the loaded DataFrame, or None if not found.
    If multiple CSVs match, it picks the first one (or you could handle differently).
    """
    pattern = os.path.join(data_dir, f"{env_name}_exp-*_{method_name}_data.csv")
    matched_files = glob.glob(pattern)
    if not matched_files:
        print(f"[Warning] No CSV found for {env_name} with method={method_name} under {data_dir}")
        return None
    
    # If multiple, you might pick the "longest" or "latest" or something. For now, pick the first.
    csv_file = matched_files[0]
    print(f"  Found CSV for {method_name}: {csv_file}")
    df = pd.read_csv(csv_file)
    return df

def compute_mean_return_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (q, clip, episode), compute mean and std across seeds.
    Returns a DataFrame with columns: q, clip, episode, mean_return, std_return, n_seeds.
    """
    grouped = df.groupby(['q','clip','episode'])
    df_agg = grouped['Real Det Return'].agg(['mean','std','count']).reset_index()
    df_agg.rename(
        columns={
            'mean': 'mean_return',
            'std': 'std_return',
            'count': 'n_seeds'
        }, 
        inplace=True
    )
    return df_agg

def compute_auc(df_mean_std: pd.DataFrame, use_trapz: bool = False) -> pd.DataFrame:
    """
    Compute area under curve (AUC) for each (q, clip).
    Optionally use trapezoidal rule (more accurate) or simple sum.
    """
    # Sort to ensure episode is ascending
    df_sorted = df_mean_std.sort_values(by=['q','clip','episode'])
    
    auc_records = []
    for (q_val, clip_val), group_df in df_sorted.groupby(['q','clip']):
        x = group_df['episode'].values
        y = group_df['mean_return'].values
        
        if use_trapz:
            # trapezoidal rule
            auc_val = np.trapz(y, x)
        else:
            # discrete sum
            auc_val = y.sum()
        
        auc_records.append({
            'q': q_val,
            'clip': clip_val,
            'auc': auc_val
        })
    
    return pd.DataFrame(auc_records)

def find_best_clipping(auc_df: pd.DataFrame, skip_q=1.0) -> tuple:
    """
    Find the (q,clip) with the highest AUC, ignoring skip_q if desired.
    Returns (best_q, best_clip).
    If there's no candidate other than skip_q, returns (None, None).
    """
    filtered = auc_df[auc_df['q'] != skip_q]
    if filtered.empty:
        return (None, None)
    
    best_row = filtered.iloc[filtered['auc'].argmax()]
    return (best_row['q'], best_row['clip'])

def load_baseline_data(baseline_csv_path: str) -> pd.DataFrame:
    """
    Loads a baseline CSV with columns [Iteration, Real Det Return, ...].
    Converts 'Iteration' to 'episode' by multiplying by 5000 (if you do that).
    """
    df = pd.read_csv(baseline_csv_path)
    if 'Iteration' in df.columns:
        df['episode'] = df['Iteration'] * 5000
    return df

# ========== MAIN LOGIC ==========

def main():
    """
    Create a single plot with:
      - For each method in METHODS: q=1 line, best clip line
      - All baselines in BASELINES
    """
    # Dictionary to store the processed DataFrame (grouped means) for each method
    method_results = {}
    # Dictionary to store best (q, clip) for each method
    method_best = {}
    
    print(f"=== Processing environment: {ENV_NAME} ===")
    # 1) For each method, load the corresponding CSV and compute mean returns, then AUC.
    for method in METHODS:
        df_raw = load_processed_data_for_method(ENV_NAME, method, PROCESSED_DATA_DIR)
        if df_raw is None:
            continue  # skip if no file found
        
        df_mean_std = compute_mean_return_across_seeds(df_raw)
        auc_df = compute_auc(df_mean_std, use_trapz=True)  # or False if you prefer discrete sum
        best_q, best_clip = find_best_clipping(auc_df, skip_q=1.0)
        
        method_results[method] = df_mean_std
        method_best[method] = (best_q, best_clip)
        
        print(f"    Method={method}, best clip => q={best_q}, clip={best_clip}")
    
    # 2) Load each baseline
    baseline_dfs = {}
    for baseline_name, baseline_path in BASELINES.items():
        if not os.path.isfile(baseline_path):
            print(f"[Warning] Baseline file not found: {baseline_path}")
            continue
        df_base = load_baseline_data(baseline_path)
        baseline_dfs[baseline_name] = df_base
    
    # 3) Create the plot
    plt.figure(figsize=(10, 6))
    color_idx = 0
    
    for method, df_mean_std in method_results.items():
        # (a) Plot q=1 line
        q1_data = df_mean_std[df_mean_std['q'] == 1.0].sort_values('episode')
        if not q1_data.empty:
            plt.plot(
                q1_data['episode'],
                q1_data['mean_return'],
                label=f"{method} (q=1)",
            )
            if SHOW_CONFIDENCE:
                upper = q1_data['mean_return'] + q1_data['std_return']
                lower = q1_data['mean_return'] - q1_data['std_return']
                plt.fill_between(q1_data['episode'], lower, upper, alpha=0.2)
        
        # (b) Plot best clip line (if found)
        best_q, best_clip = method_best[method]
        if best_q is not None:
            best_data = df_mean_std[
                (df_mean_std['q'] == best_q) & (df_mean_std['clip'] == best_clip)
            ].sort_values('episode')
            if not best_data.empty:
                label = f"{method} (q={best_q}, clip={best_clip})"
                plt.plot(best_data['episode'], best_data['mean_return'], label=label)
                if SHOW_CONFIDENCE:
                    upper = best_data['mean_return'] + best_data['std_return']
                    lower = best_data['mean_return'] - best_data['std_return']
                    plt.fill_between(best_data['episode'], lower, upper, alpha=0.2)
    
    # (c) Plot baselines
    for baseline_name, df_base in baseline_dfs.items():
        df_base = df_base.sort_values('Itration')
        plt.plot(
            df_base['Itration'] * 5000, 
            df_base['Real Det Return'], 
            label=f"Baseline: {baseline_name}", 
            linestyle='--'
        )
    
    # 4) Decorate and save
    plt.title(f"{ENV_NAME} Comparison: Q=1 vs Best Clip (Multiple Methods) + Baselines")
    plt.xlabel("Episode")
    plt.ylabel("Real Det Return")
    plt.legend()
    plt.grid(True)

    # Save figure
    method_str = "_".join(METHODS) if METHODS else "noMethod"
    out_dir = Path("plots") / ENV_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{ENV_NAME}_{method_str}_q1_vs_bestclip_vs_baselines.png"
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Plot saved at {out_path}")

if __name__ == "__main__":
    main()

