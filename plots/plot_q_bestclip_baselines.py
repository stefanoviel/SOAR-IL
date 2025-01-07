import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========== USER DEFINED VARIABLES ==========

# List of environments to process
ENVIRONMENTS = ["Ant-v5", "Hopper-v5", "Walker2d-v5", "Humanoid-v5", "HalfCheetah-v5"]  

# Methods to plot
METHODS = ["cisl", "maxentirl_sa"]

# Manually specify colors for each method
METHOD_COLORS = {
    "maxentirl": "blue",
    "rkl": "orange",
    "cisl": "blue",
    "maxentirl_sa": "orange",
    # Add more if needed
}


BASELINES_DICT = {
    "Ant-v5": {
        "gail": "logs/Ant-v5/exp-16/gail/2025_01_05_15_56_35/progress.csv",
        # "bc":   "/path/to/baseline_bc.csv",
    },
    "Hopper-v5": {
        "gail": "logs/Hopper-v5/exp-16/gail/2025_01_05_15_56_35/progress.csv",
        # add more if you have them
    },
    "Walker2d-v5": {
        "gail": "logs/Walker2d-v5/exp-16/gail/2025_01_05_15_56_35/progress.csv",
        # add more if you have them
    },
    "Humanoid-v5": {
        "gail": "logs/Humanoid-v5/exp-16/gail/2025_01_05_15_56_35/progress.csv",
        # add more if you have them
    },
    "HalfCheetah-v5": {
        "gail": "logs/HalfCheetah-v5/exp-16/gail/2025_01_05_15_56_35/progress.csv",
        # add more if you have them
    },
}

# Manually specify colors for each baseline
BASELINE_COLORS = {
    "gail": "green",
    # "bc": "purple",
    # Add more if needed
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
    If multiple CSVs match, it picks the first one (or you could handle differently).
    """
    pattern = os.path.join(data_dir, f"{env_name}_exp-*_{method_name}_data.csv")
    matched_files = glob.glob(pattern)
    if not matched_files:
        print(f"[Warning] No CSV found for {env_name} with method={method_name} under {data_dir}")
        return None
    
    # If multiple, pick the first or implement custom selection logic
    csv_file = matched_files[0]
    print(f"  Found CSV for {method_name}: {csv_file}")
    df = pd.read_csv(csv_file)
    return df

def compute_mean_return_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (q, clip, episode), compute mean and std across seeds.
    Returns a DataFrame with columns: q, clip, episode, mean_return, std_return, n_seeds.
    This version ensures we keep rows where clip is NaN (e.g., for q=1).
    """
    # Use dropna=False to include NaN values as their own group
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

def load_baseline_data(baseline_csv_path: str) -> pd.DataFrame:
    """
    Loads a baseline CSV with columns [Iteration, Real Det Return, ...].
    Converts 'Iteration' to 'episode' by multiplying by 5000 (if that's your convention).
    """
    df = pd.read_csv(baseline_csv_path)
    if 'Iteration' in df.columns:
        df['episode'] = df['Iteration'] * 5000
    return df

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
            # Look for "Expert(Det) Return Avg: <number>"
            match = re.search(r"Expert\(Det\) Return Avg:\s*([\d.]+)", line)
            if match:
                return float(match.group(1))
    return None

def process_environment(env_name: str):
    """
    Creates a plot for a single environment, using:
      - Q=1 line (dashed) + best clipping line (solid) for each method
      - Baselines for that environment
      - Horizontal line for the expert return
    Then saves the figure in `plots/{env_name}` directory.
    """

    # Retrieve baseline dictionary for this environment (if any)
    baselines_for_env = BASELINES_DICT.get(env_name, {})

    # Max episodes for environment
    max_ep = MAX_EPISODES_DICT.get(env_name, None)
    
    # Dictionary to store data & best clip results
    method_results = {}
    method_best = {}
    
    print(f"=== Processing environment: {env_name} ===")

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
    
    # 2) Load each baseline, filter by max_ep
    baseline_dfs = {}
    for baseline_name, baseline_path in baselines_for_env.items():
        if not os.path.isfile(baseline_path):
            print(f"[Warning] Baseline file not found: {baseline_path}")
            continue
        df_base = load_baseline_data(baseline_path)
        if max_ep is not None:
            # Filter if the 'episode' (or 'Iteration') column is beyond max_ep
            # (Note: the original code had a minor typo 'Itration'—ensure you adjust for your data)
            df_base = df_base[df_base['Itration'] <= max_ep]
        baseline_dfs[baseline_name] = df_base
    
    # 3) Parse expert deterministic return (horizontal line)
    expert_return = None
    expert_txt_path = EXPERT_FILE_DICT.get(env_name, None)
    if expert_txt_path:
        expert_return = parse_expert_det_return(expert_txt_path)
        if expert_return is not None:
            print(f"Expert(Det) Return for {env_name}: {expert_return:.2f}")
    
    # 4) Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each method
    for method, df_mean_std in method_results.items():
        # Get color if specified, else fallback to next in the cycle
        color = METHOD_COLORS.get(method, None)
        if color is None:
            # fallback color from the default cycle
            color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        # (a) Plot q=1 line (dashed)
        q1_data = df_mean_std[df_mean_std['q'] == 1.0].sort_values('episode')
        if not q1_data.empty:
            # -- Smoothing step (rolling average) --
            q1_data['smoothed_mean_return'] = (
                q1_data['mean_return'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            )
            plt.plot(
                q1_data['episode'],
                q1_data['smoothed_mean_return'],  # << use smoothed data
                label=f"{method} (q=1)",
                color=color,
                linestyle='--'
            )
            if SHOW_CONFIDENCE:
                upper = q1_data['mean_return'] + q1_data['std_return']
                lower = q1_data['mean_return'] - q1_data['std_return']
                plt.fill_between(q1_data['episode'], lower, upper, alpha=0.2, color=color)
        
        # (b) Plot best clip line (solid)
        best_q, best_clip = method_best[method]
        if best_q is not None:
            best_data = df_mean_std[
                (df_mean_std['q'] == best_q) & (df_mean_std['clip'] == best_clip)
            ].sort_values('episode')
            if not best_data.empty:
                # -- Smoothing step (rolling average) --
                best_data['smoothed_mean_return'] = (
                    best_data['mean_return'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
                )
                label = f"{method} (q={best_q}, clip={best_clip})"
                plt.plot(
                    best_data['episode'],
                    best_data['smoothed_mean_return'],  # << use smoothed data
                    label=label,
                    color=color,
                    linestyle='-'
                )
                if SHOW_CONFIDENCE:
                    upper = best_data['mean_return'] + best_data['std_return']
                    lower = best_data['mean_return'] - best_data['std_return']
                    plt.fill_between(best_data['episode'], lower, upper, alpha=0.2, color=color)
    
    # (c) Plot baselines
    for baseline_name, df_base in baseline_dfs.items():
        color = BASELINE_COLORS.get(baseline_name, None)
        if color is None:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
        
        df_base = df_base.sort_values('Itration')
        
        # -- Smoothing step (rolling average) --
        df_base['smoothed_return'] = (
            df_base['Real Det Return'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        )
        
        plt.plot(
            df_base['Itration'] * 5000,  # << convert to episode
            df_base['smoothed_return'],  # << use smoothed data
            label=f"Baseline: {baseline_name}", 
            linestyle='-.',
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
    plt.title(f"{env_name} Comparison: Q=1 vs Best Clip (Multiple Methods) + Baselines + Expert")
    plt.xlabel("Episode")
    plt.ylabel("Real Det Return")
    plt.legend()
    plt.grid(True)

    # Save figure
    method_str = "_".join(METHODS) if METHODS else "noMethod"
    out_dir = Path("plots") / env_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{env_name}_{method_str}_q1_vs_bestclip_vs_baselines_expert.png"
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Plot saved at {out_path}")

# ========== MAIN LOGIC ==========

def main():
    """
    Loops over all environments in ENVIRONMENTS and creates one plot per environment.
    """
    for env_name in ENVIRONMENTS:
        process_environment(env_name)

if __name__ == "__main__":
    main()
