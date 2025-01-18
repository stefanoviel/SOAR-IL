import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib as mpl

# Reuse the matplotlib parameters from the original script
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['legend.fontsize'] = 14

# Environment and method settings
ENVIRONMENTS = [
    "Ant-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
]

METHODS = ["cisl", "maxentirl_sa", "maxentirl", "rkl"]

# Add display name mapping for methods
METHOD_DISPLAY_NAMES = {
    "maxentirl": "ML-IRL",
    "maxentirl_sa": "ML-IRL (SA)",
    "cisl": "CISL",
    "rkl": "RKL"
}

METHOD_COLORS = {
    "maxentirl": "blue",
    "rkl": "red",
    "cisl": "green",          
    "maxentirl_sa": "purple",
}

PROCESSED_DATA_DIR = "processed_data"
SMOOTHING_WINDOW = 15

# Reuse the expert file dictionary and max episodes dictionary from original script
EXPERT_FILE_DICT = {
    "Ant-v5": "expert_data/meta/AntFH-v0_airl.txt",
    "Hopper-v5": "expert_data/meta/Hopper-v5_1.txt",
    "Walker2d-v5": "expert_data/meta/Walker2d-v5_1.txt",
    "Humanoid-v5": "expert_data/meta/Humanoid-v5_1.txt",
}


MAX_EPISODES_DICT = {
    "Hopper-v5": 1e6,
    "Walker2d-v5": 1.5e6,
    "Ant-v5": 1.2e6,
    "Humanoid-v5": 1e6,
}

# Reuse the helper functions from the original script
def load_processed_data_for_method(env_name, method_name, data_dir):
    pattern = os.path.join(data_dir, f"{env_name}_exp-*_{method_name}_data.csv")
    matched_files = glob.glob(pattern)
    if not matched_files:
        print(f"[Warning] No CSV found for {env_name} with method={method_name} under {data_dir}")
        return None
    
    csv_file = matched_files[0]
    print(f"  Found CSV for {method_name}: {csv_file}")
    df = pd.read_csv(csv_file)
    return df

def compute_mean_return_across_seeds(df):
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

def compute_auc(df_mean_std, use_trapz=False):
    df_sorted = df_mean_std.sort_values(by=['q','clip','episode'])
    auc_records = []
    for (q_val, clip_val), group_df in df_sorted.groupby(['q','clip']):
        x = group_df['episode'].values
        y = group_df['mean_return'].values
        
        if use_trapz:
            auc_val = np.trapz(y, x)
        else:
            auc_val = y.sum()
        
        auc_records.append({
            'q': q_val,
            'clip': clip_val,
            'auc': auc_val
        })
    
    return pd.DataFrame(auc_records)

def find_best_clipping(auc_df, skip_q=1.0):
    filtered = auc_df[auc_df['q'] != skip_q]
    if filtered.empty:
        return (None, None)
    
    best_row = filtered.iloc[filtered['auc'].argmax()]
    return (best_row['q'], best_row['clip'])

def parse_expert_det_return(expert_txt_path):
    if not os.path.isfile(expert_txt_path):
        print(f"[Warning] Expert file not found: {expert_txt_path}")
        return None
    
    with open(expert_txt_path, "r") as f:
        for line in f:
            match = re.search(r"Expert\(Det\) Return Avg:\s*([\d.]+)", line)
            if match:
                return float(match.group(1))
    return None

def get_normalized_data_for_method_env(env_name, method_name):
    """Get normalized data for a specific method and environment."""
    df_raw = load_processed_data_for_method(env_name, method_name, PROCESSED_DATA_DIR)
    if df_raw is None:
        return None, None
    
    df_mean_std = compute_mean_return_across_seeds(df_raw)
    max_ep = MAX_EPISODES_DICT.get(env_name, None)
    if max_ep is not None:
        df_mean_std = df_mean_std[df_mean_std['episode'] <= max_ep]
    
    # Normalize by expert return
    expert_txt_path = EXPERT_FILE_DICT.get(env_name)
    if not expert_txt_path:
        return None, None
    expert_return = parse_expert_det_return(expert_txt_path)
    if expert_return is None or expert_return <= 0:
        return None, None
    
    df_mean_std['std_return'] = df_mean_std['mean_return'] / expert_return
    
    # Find best parameters
    auc_df = compute_auc(df_mean_std, use_trapz=True)
    best_q, best_clip = find_best_clipping(auc_df, skip_q=1.0)
    
    return df_mean_std, (best_q, best_clip)

# [Previous imports and constant definitions remain the same...]
def plot_method_average(method, ax):
    """Plot average performance across environments for a specific method."""
    color = METHOD_COLORS.get(method, "black")
    display_name = METHOD_DISPLAY_NAMES.get(method, method.upper())
    
    # Collect data for all environments
    q1_data_list = []
    best_data_list = []
    
    for env_name in ENVIRONMENTS:
        df_mean_std, (best_q, best_clip) = get_normalized_data_for_method_env(env_name, method)
        if df_mean_std is None:
            continue
            
        # Get q=1 data
        q1_data = df_mean_std[df_mean_std['q'] == 1.0].sort_values('episode')
        if not q1_data.empty:
            q1_data_list.append(q1_data)
            
        # Get best data
        if best_q is not None:
            best_data = df_mean_std[
                (df_mean_std['q'] == best_q) & (df_mean_std['clip'] == best_clip)
            ].sort_values('episode')
            if not best_data.empty:
                best_data_list.append(best_data)
    
    lines = []
    labels = []
    
    # Plot averaged data
    if q1_data_list:
        # Interpolate and average q1 data
        min_ep = max(df['episode'].min() for df in q1_data_list)
        max_ep = min(df['episode'].max() for df in q1_data_list)
        ep_range = np.linspace(min_ep, max_ep, 100)
        
        avg_returns = []
        for ep in ep_range:
            returns = []
            for df in q1_data_list:
                idx = (df['episode'] - ep).abs().idxmin()
                returns.append(df.loc[idx, 'std_return'])
            avg_returns.append(np.mean(returns))
            
        smoothed_returns = pd.Series(avg_returns).rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        line, = ax.plot(ep_range, smoothed_returns, linestyle='--', color=color, 
                       label=f'{display_name} (Base)')
    
    if best_data_list:
        # Interpolate and average best data
        min_ep = max(df['episode'].min() for df in best_data_list)
        max_ep = min(df['episode'].max() for df in best_data_list)
        ep_range = np.linspace(min_ep, max_ep, 100)
        
        avg_returns = []
        for ep in ep_range:
            returns = []
            for df in best_data_list:
                idx = (df['episode'] - ep).abs().idxmin()
                returns.append(df.loc[idx, 'std_return'])
            avg_returns.append(np.mean(returns))
            
        smoothed_returns = pd.Series(avg_returns).rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        ax.plot(ep_range, smoothed_returns, linestyle='-', color=color,
               label=f'{display_name} + SOAR')
    
    # Add expert line with label only for first method
    if method == METHODS[-1]:
        ax.plot([min_ep, max_ep], [1.0, 1.0], color='black', linestyle=':', label='Expert')
    else:
        ax.plot([min_ep, max_ep], [1.0, 1.0], color='black', linestyle=':', label=None)
    
    # Add decorations with the display name instead of the method name
    ax.set_title(f"{display_name}", y=1.02)
    ax.set_xlabel("Episode")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

def main():
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    
    for i, method in enumerate(METHODS):
        plot_method_average(method, axes[i])
        if i == 0:  # Only add y-label to the first subplot
            axes[i].set_ylabel("Average Normalized Return")
    
    # Adjust layout
    fig.subplots_adjust(
        top=0.83,
        bottom=0.25,
        left=0.06,
        right=0.98
    )
    
    # Add super title
    fig.suptitle(
        "Average Performance Across Environments",
        y=0.96,
        fontsize=23
    )
    
    # Collect all legend handles and labels
    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    # Remove duplicate expert entries
    unique_handles = []
    unique_labels = []
    seen_labels = set()
    for h, l in zip(handles, labels):
        if l not in seen_labels:
            unique_handles.append(h)
            unique_labels.append(l)
            seen_labels.add(l)
    
    # Create combined legend
    fig.legend(
        unique_handles,
        unique_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=5  # Adjust this number based on how many methods you have
    )
    
    # Save the figure
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "method_averages_comparison.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved combined figure to {out_path}")



if __name__ == "__main__":
    main()