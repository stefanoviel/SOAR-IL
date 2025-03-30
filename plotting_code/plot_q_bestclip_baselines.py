import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# 1) Increase font sizes for ticks, axis labels, etc.
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 16    # Size of "Episode" or "Return" labels
mpl.rcParams['axes.titlesize'] = 16    # Size of each subplot title
mpl.rcParams['legend.fontsize'] = 14

# ========== USER DEFINED VARIABLES ==========

EXPERIMENT_NUMBER = 16  # Change this to 1 or 16 as needed

ENVIRONMENTS = [
    "Ant-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    # "HalfCheetah-v5",
]

# Methods (two lines each: q=1 [dashed], best q [solid])
METHODS = ["cisl", "maxentirl_sa", "hype"]
# METHODS = ["maxentirl", "rkl"]

# Baselines (one line each: dash-dot)
# BASELINES = []
BASELINES = ["gail", "sqil", "opt-AIL_sa"]
# BASELINES = ["opt-AIL"]

# Unique colors for each method
METHOD_COLORS = {
    "maxentirl": "blue",
    "rkl": "red",
    "cisl": "green",          
    "maxentirl_sa": "purple",
    "hype": "crimson"
}

# Dictionary to map internal method names to display names
METHOD_DISPLAY_NAMES = {
    "maxentirl": "ML-IRL",
    "maxentirl_sa": "ML-IRL (SA)",
    "cisl": "csil",
    "rkl": "rkl",
    "hype": "HYPE",
}

BASELINE_DISPLAY_NAMES = {
    "opt-AIL_sa": "opt-AIL (SA)",
}

# Unique colors for baselines
BASELINE_COLORS = {
    "gail": "orange",
    "sqil": "brown",
    "opt-AIL": "cyan",
    "opt-AIL_sa": "magenta",
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

def load_processed_data_for_method(env_name: str, method_name: str, data_dir: str, exp_num: int) -> pd.DataFrame:
    """
    Finds a CSV in data_dir matching the specific experiment number:
      {env_name}_exp-{exp_num}_{method_name}_data.csv
    Returns the loaded DataFrame, or None if not found.
    """
    pattern = os.path.join(data_dir, f"{env_name}_exp-{exp_num}_{method_name}_data.csv")
    matched_files = glob.glob(pattern)
    if not matched_files:
        print(f"[Warning] No CSV found for {env_name} with method={method_name}, exp={exp_num} under {data_dir}")
        return None
    
    csv_file = matched_files[0]
    print(f"  Found CSV for {method_name} (exp-{exp_num}): {csv_file}")
    df = pd.read_csv(csv_file)
    return df


def compute_mean_return_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (q, clip, episode), compute mean and std across seeds.
    Returns columns: q, clip, episode, mean_return, std_return, n_seeds.
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
    Compute area under the curve for each (q, clip).
    Optionally use trapezoidal rule.
    """
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

def find_best_clipping(auc_df: pd.DataFrame, skip_q=1.0) -> tuple:
    """
    Find (q, clip) with highest AUC, ignoring skip_q if desired.
    Returns (best_q, best_clip) or (None, None) if no candidate found.
    """
    filtered = auc_df[auc_df['q'] != skip_q]
    if filtered.empty:
        return (None, None)
    
    best_row = filtered.iloc[filtered['auc'].argmax()]
    return (best_row['q'], best_row['clip'])

def parse_expert_det_return(expert_txt_path: str) -> float:
    """
    Parse the file for a line like:
      Expert(Det) Return Avg: 4061.41, std: 730.58
    Returns the float (e.g., 4061.41).
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


def plot_environment(env_name: str, ax: plt.Axes, idx: int, exp_num: int):
    """
    Plots data for a single environment onto a given Axes.
    """
    print(f"=== Processing environment: {env_name} (exp-{exp_num}) ===")

    # Expert Return
    expert_txt_path = EXPERT_FILE_DICT.get(env_name)
    if not expert_txt_path:
        print(f"  No expert file for {env_name}, skipping plot.")
        return
    expert_return = parse_expert_det_return(expert_txt_path)
    if expert_return is None or expert_return <= 0:
        print(f"  Invalid expert return for {env_name}, skipping plot.")
        return

    # For x-limits
    max_ep = MAX_EPISODES_DICT.get(env_name, None)
    
    # Gather data
    method_results = {}
    method_best = {}
    
    # 1) Methods
    for method in METHODS:
        df_raw = load_processed_data_for_method(env_name, method, PROCESSED_DATA_DIR, exp_num)
        if df_raw is None:
            continue
        
        df_mean_std = compute_mean_return_across_seeds(df_raw)
        if max_ep is not None:
            df_mean_std = df_mean_std[df_mean_std['episode'] <= max_ep]
        
        # standardize
        df_mean_std['std_return'] = df_mean_std['mean_return'] / expert_return
        
        # find best
        auc_df = compute_auc(df_mean_std, use_trapz=True)
        best_q, best_clip = find_best_clipping(auc_df, skip_q=1.0)

        print(f"  {method}: Best q={best_q}, clip={best_clip}")
        
        method_results[method] = df_mean_std
        method_best[method] = (best_q, best_clip)
    
    # 2) Baselines
    baseline_results = {}
    for baseline in BASELINES:
        df_raw = load_processed_data_for_method(env_name, baseline, PROCESSED_DATA_DIR, exp_num)
        if df_raw is None:
            continue
        
        df_mean_std = compute_mean_return_across_seeds(df_raw)
        if max_ep is not None:
            df_mean_std = df_mean_std[df_mean_std['episode'] <= max_ep]
        
        # standardize
        df_mean_std['std_return'] = df_mean_std['mean_return'] / expert_return
        baseline_results[baseline] = df_mean_std
    
    # ========== Plot ==========

    # (a) Methods
    for method, df_mean_std in method_results.items():
        color = METHOD_COLORS.get(method, "black")
        
        # q=1 line
        q1_data = df_mean_std[df_mean_std['q'] == 1.0].sort_values('episode')
        if not q1_data.empty:
            q1_data['smoothed'] = q1_data['std_return'].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
            
            best_q, best_clip = method_best[method]
            # Use display name instead of internal method name
            line_label = METHOD_DISPLAY_NAMES.get(method, method)
            
            
            ax.plot(
                q1_data['episode'],
                q1_data['smoothed'],
                linestyle='--',
                color=color,
                label=line_label
            )
        
        # best line
        best_q, best_clip = method_best[method]
        if best_q is not None:
            best_data = df_mean_std[
                (df_mean_std['q'] == best_q) & (df_mean_std['clip'] == best_clip)
            ].sort_values('episode')
            if not best_data.empty:
                best_data['smoothed'] = best_data['std_return'].rolling(
                    SMOOTHING_WINDOW, min_periods=1
                ).mean()
                
                # If best_q=4 => "method + SOAR", else => "_nolegend_"
                if best_q == 4:
                    display_name = METHOD_DISPLAY_NAMES.get(method, method)
                    best_label = f"{display_name} + SOAR (Ours)"
                else:
                    best_label = "_nolegend_"
                
                ax.plot(
                    best_data['episode'],
                    best_data['smoothed'],
                    linestyle='-',
                    color=color,
                    label=best_label
                )
    
    # (b) Baselines
    for baseline, df_mean_std in baseline_results.items():
        color = BASELINE_COLORS.get(baseline, "black")
        df_mean_std = df_mean_std.sort_values('episode')
        df_mean_std['smoothed'] = df_mean_std['std_return'].rolling(
            SMOOTHING_WINDOW, min_periods=1
        ).mean()
        
        # Use display name instead of internal method name
        line_label = BASELINE_DISPLAY_NAMES.get(baseline, baseline)
        print(f"  {baseline}: line_label={line_label}")

        ax.plot(
            df_mean_std['episode'],
            df_mean_std['smoothed'],
            linestyle='-.',
            color=color,
            label=line_label
        )
    
    # (c) Expert line at y=1.0
    ax.axhline(y=1.0, color='black', linestyle=':', label='Expert')
    
    # (d) Decorate
    ax.set_title(env_name, y=1.02)  # move title a bit higher
    if idx == 0:
        ax.set_ylabel("Return")
    else:
        ax.set_ylabel("")  # no label for others
    ax.set_xlabel("Episode")
    
    # Give some margin above/below 0..1
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

def plot_best_clipping_return(file_path: str, skip_q: float = 1.0, use_trapz: bool = False):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Compute mean and std return across seeds
    df_mean_std = compute_mean_return_across_seeds(df)

    # Compute AUC for each (q, clip)
    auc_df = compute_auc(df_mean_std, use_trapz=use_trapz)

    # Find the best clipping value for each q
    best_clippings = {}
    for q_val in auc_df['q'].unique():
        filtered_auc = auc_df[auc_df['q'] == q_val]
        if not filtered_auc.empty:
            best_row = filtered_auc.iloc[filtered_auc['auc'].argmax()]
            best_clippings[q_val] = best_row['clip']

    # Plot return over time for the best clipping value for each q
    plt.figure(figsize=(10, 6))
    for q_val, best_clip in best_clippings.items():
        # Filter the data for the best clipping value for this q
        subset = df_mean_std[(df_mean_std['q'] == q_val) & (df_mean_std['clip'] == best_clip)]

        # Sort by episode for consistent plotting
        subset = subset.sort_values(by='episode')

        # Truncate episodes at 1.2 million
        subset = subset[subset['episode'] <= 1.2e6]

        # Apply smoothing
        subset['smoothed'] = subset['mean_return'].rolling(SMOOTHING_WINDOW, min_periods=1).mean()

        # Plot
        if q_val == 1.0:
            plt.plot(subset['episode'], subset['smoothed'], label=f'Networks={q_val}')
        else:
            plt.plot(subset['episode'], subset['smoothed'], label=f'Networks={q_val}, clip={best_clip}')

    env_name = "Ant-v5"
    expert_txt_path = EXPERT_FILE_DICT.get(env_name)
    if not expert_txt_path:
        print(f"  No expert file for {env_name}, skipping plot.")
        return
    expert_return = parse_expert_det_return(expert_txt_path)

    # Plot expert return
    plt.axhline(y=expert_return, color='black', linestyle=':', label='Expert')

    # Add labels, legend, and title
    plt.xlabel('Episode')
    plt.ylabel('Mean Return')
    # plt.title('Mean Return vs Episode for Best Clipping Values')
    plt.legend()
    plt.grid(True)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot
    plt.savefig('plots/best_clipping_returns.png', dpi=300, bbox_inches='tight')
    plt.close()



def main():
    n_envs = len(ENVIRONMENTS)
    fig, axes = plt.subplots(
        1, n_envs,
        figsize=(5 * n_envs, 6),
        sharey=True
    )
    
    if n_envs == 1:
        axes = [axes]
    
    for i, env_name in enumerate(ENVIRONMENTS):
        plot_environment(env_name, ax=axes[i], idx=i, exp_num=EXPERIMENT_NUMBER)
    
    # Gather legend info
    handles_labels = [ax.get_legend_handles_labels() for ax in axes]
    handles = []
    labels = []
    for hlist, llist in handles_labels:
        handles.extend(hlist)
        labels.extend(llist)
    
    # Deduplicate
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    
    # Place legend
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=4
    )
    
    # Adjust spacing
    fig.subplots_adjust(
        top=0.92,
        bottom=0.3,
        left=0.06,
        right=0.98
    )

    # fig.suptitle(
    #     f"State‐Action Methods (Expert Trajectories {EXPERIMENT_NUMBER})",
    #     y=0.96,
    #     fontsize=23
    # )

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"all_envs_single_row_standardized_exp{EXPERIMENT_NUMBER}.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved combined figure to {out_path}")


if __name__ == "__main__":
    main()

    # plot_best_clipping_return('processed_data/Ant-v5_exp-16_testing_number_nn_data.csv')