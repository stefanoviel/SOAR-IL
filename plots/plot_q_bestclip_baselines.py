import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========== USER DEFINED VARIABLES ==========

# List of environments to process
ENVIRONMENTS = [
    "Ant-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    "HalfCheetah-v5",
]

# Methods (need to plot q=1 and best clipping)
# METHODS = ["maxentirl", "rkl"]
METHODS = ["cisl", "maxentirl_sa"]

# Baselines (only one line per baseline)
BASELINES = ["gail", "sqil", "opt-AIL"]

# Manually specify colors for each method
METHOD_COLORS = {
    "maxentirl": "blue",        # or '#1f77b4'
    "rkl": "red",               # or '#d62728'
    "cisl": "green",            # or '#2ca02c'
    "maxentirl_sa": "purple",   # or '#9467bd'
}

# Manually specify colors for each baseline
BASELINE_COLORS = {
    "gail": "orange",
    "sqil": "brown",
    "opt-AIL": "cyan",
}

PROCESSED_DATA_DIR = "processed_data"
SHOW_CONFIDENCE = False
SMOOTHING_WINDOW = 15

# Dictionary specifying max episodes for each environment
MAX_EPISODES_DICT = {
    "Hopper-v5": 1e6,
    "Walker2d-v5": 1.5e6,
    "Ant-v5": 1.2e6,
    "Humanoid-v5": 1e6,
    "HalfCheetah-v5": 1.5e6,
    # Add more environments as needed
}

# (Optional) mapping from environment to the exact expert .txt file
EXPERT_FILE_DICT = {
    "Ant-v5": "expert_data/meta/AntFH-v0_airl.txt",
    "Hopper-v5": "expert_data/meta/Hopper-v5_1.txt",
    "Walker2d-v5": "expert_data/meta/Walker2d-v5_1.txt",
    "Humanoid-v5": "expert_data/meta/Humanoid-v5_1.txt",
    "HalfCheetah-v5": "expert_data/meta/HalfCheetah-v5_1.txt",
}

# ========== HELPER FUNCTIONS ==========

def load_processed_data_for_method(env_name: str, method_name: str, data_dir: str) -> pd.DataFrame:
    """
    Finds a CSV in data_dir matching something like:
      {env_name}_exp-*_{method_name}_data.csv
    Returns the loaded DataFrame, or None if not found.
    If multiple CSVs match, it picks the first one.
    """
    pattern = os.path.join(data_dir, f"{env_name}_exp-*_{method_name}_data.csv")
    matched_files = glob.glob(pattern)
    if not matched_files:
        print(f"[Warning] No CSV found for {env_name} with method={method_name} under {data_dir}")
        return None
    
    csv_file = matched_files[0]
    print(f"  Found CSV for {method_name}: {csv_file}")
    df = pd.read_csv(csv_file)
    return df

def compute_mean_return_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (q, clip, episode), compute mean and std across seeds.
    Returns a DataFrame with columns: q, clip, episode, mean_return, std_return, n_seeds.
    This version ensures we keep rows where clip is NaN (e.g., for baselines).
    """
    grouped = df.groupby(['q', 'clip', 'episode'], dropna=False)
    df_agg = grouped['Real Det Return'].agg(['mean', 'std', 'count']).reset_index()
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

def parse_expert_det_return(expert_txt_path: str) -> float:
    """
    Parse the file looking for a line like:
      Expert(Det) Return Avg: 4061.41, std: 730.58
    Returns the float (e.g. 4061.41).
    If not found, returns None.
    """
    if not os.path.isfile(expert_txt_path):
        print(f"[Warning] Expert file not found: {expert_txt_path}")
        return None
    
    with open(expert_txt_path, "r") as f:
        for line in f:
            match = re.search(r"Expert\(Det\) Return Avg:\s*([\d.]+)", line)
            if match:
                return float(match.group(1))
    return None

def process_environment(env_name: str):
    """
    Creates a single plot for each environment, including:
      - For each method in METHODS: Q=1 (dashed) + best clipping (solid).
      - For each baseline in BASELINES: single line (solid or dash-dotted, as you prefer).
      - Horizontal line for the expert return (if available).
    Then saves the figure in `plots/{env_name}` directory.
    """
    print(f"=== Processing environment: {env_name} ===")
    
    # Max episodes for environment
    max_ep = MAX_EPISODES_DICT.get(env_name, None)
    
    # Dictionary to store data & best clip results for methods
    method_results = {}
    method_best = {}
    
    # 1) For each method, load CSV, compute means, then AUC, then best clip
    for method in METHODS:
        df_raw = load_processed_data_for_method(env_name, method, PROCESSED_DATA_DIR)
        if df_raw is None:
            continue  # skip if no file found
        
        df_mean_std = compute_mean_return_across_seeds(df_raw)
        
        # If there's a max_ep limit, filter out episodes
        if max_ep is not None:
            df_mean_std = df_mean_std[df_mean_std['episode'] <= max_ep]
        
        auc_df = compute_auc(df_mean_std, use_trapz=True)
        best_q, best_clip = find_best_clipping(auc_df, skip_q=1.0)
        
        method_results[method] = df_mean_std
        method_best[method] = (best_q, best_clip)
        
        print(f"    Method={method}, best clip => q={best_q}, clip={best_clip}")
    
    # 2) For each baseline, load CSV, compute means
    baseline_results = {}
    for baseline in BASELINES:
        df_raw = load_processed_data_for_method(env_name, baseline, PROCESSED_DATA_DIR)
        if df_raw is None:
            continue
        
        df_mean_std = compute_mean_return_across_seeds(df_raw)
        if max_ep is not None:
            df_mean_std = df_mean_std[df_mean_std['episode'] <= max_ep]
        
        baseline_results[baseline] = df_mean_std
    
    # 3) Parse expert deterministic return (horizontal line)
    expert_return = None
    expert_txt_path = EXPERT_FILE_DICT.get(env_name, None)
    if expert_txt_path:
        expert_return = parse_expert_det_return(expert_txt_path)
        if expert_return is not None:
            print(f"Expert(Det) Return for {env_name}: {expert_return:.2f}")
    
    # 4) Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each method with two lines: q=1 (dashed), best clip (solid)
    for method, df_mean_std in method_results.items():
        color = METHOD_COLORS.get(method, None)
        if color is None:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        # (a) Q=1 line
        q1_data = df_mean_std[df_mean_std['q'] == 1.0].sort_values('episode')
        if not q1_data.empty:
            # smoothing
            q1_data['smoothed_mean_return'] = (
                q1_data['mean_return'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            )
            plt.plot(
                q1_data['episode'],
                q1_data['smoothed_mean_return'],
                label=f"{method} (q=1)",
                color=color,
                linestyle='--'
            )
            if SHOW_CONFIDENCE:
                upper = q1_data['mean_return'] + q1_data['std_return']
                lower = q1_data['mean_return'] - q1_data['std_return']
                plt.fill_between(q1_data['episode'], lower, upper, alpha=0.2, color=color)
        
        # (b) Best (q, clip) line
        best_q, best_clip = method_best[method]
        if best_q is not None:
            best_data = df_mean_std[
                (df_mean_std['q'] == best_q) & (df_mean_std['clip'] == best_clip)
            ].sort_values('episode')
            
            if not best_data.empty:
                best_data['smoothed_mean_return'] = (
                    best_data['mean_return'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
                )
                plt.plot(
                    best_data['episode'],
                    best_data['smoothed_mean_return'],
                    label=f"{method} (q={best_q}, clip={best_clip})",
                    color=color,
                    linestyle='-'
                )
                if SHOW_CONFIDENCE:
                    upper = best_data['mean_return'] + best_data['std_return']
                    lower = best_data['mean_return'] - best_data['std_return']
                    plt.fill_between(best_data['episode'], lower, upper, alpha=0.2, color=color)
    
    # Plot each baseline as a single line
    for baseline, df_mean_std in baseline_results.items():
        color = BASELINE_COLORS.get(baseline, None)
        if color is None:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        df_mean_std = df_mean_std.sort_values('episode')
        df_mean_std['smoothed_mean_return'] = (
            df_mean_std['mean_return'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        )
        
        plt.plot(
            df_mean_std['episode'],
            df_mean_std['smoothed_mean_return'],
            label=f"{baseline}",
            color=color,
            linestyle='-.'
        )
        if SHOW_CONFIDENCE:
            upper = df_mean_std['mean_return'] + df_mean_std['std_return']
            lower = df_mean_std['mean_return'] - df_mean_std['std_return']
            plt.fill_between(
                df_mean_std['episode'],
                lower,
                upper,
                alpha=0.2,
                color=color
            )
    
    # (d) Plot expert horizontal line (if found)
    if expert_return is not None:
        plt.axhline(
            y=expert_return, 
            color='black', 
            linestyle=':',
            label=f"Expert(Det) {expert_return:.2f}"
        )
    
    # 5) Decorate and save
    plt.title(f"{env_name} Comparison: Methods (q=1 & best clip) + Baselines + Expert")
    plt.xlabel("Episode")
    plt.ylabel("Real Det Return")
    plt.legend()
    plt.grid(True)

    # Save figure
    all_methods = METHODS + BASELINES
    method_str = "_".join(all_methods) if all_methods else "noMethod"
    out_dir = Path("plots") / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{env_name}_{method_str}_comparison.png"
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Plot saved at {out_path}")

# ========== MAIN LOGIC ==========

def main():
    for env_name in ENVIRONMENTS:
        process_environment(env_name)

if __name__ == "__main__":
    main()
