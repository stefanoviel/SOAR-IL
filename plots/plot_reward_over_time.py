import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from datetime import datetime
import numpy as np
import os
from scipy import stats
import argparse

def extract_q(folder_name):
    match = re.search(r'q(\d+)_', folder_name)
    return int(match.group(1)) if match else None

def extract_clip(folder_name):
    match = re.search(r'qstd(\d+\.\d+)', folder_name)
    return float(match.group(1)) if match else None

def load_data(base_path, max_steps):
    folders = glob.glob(f'{base_path}2024_*_seed*')
    q_clip_results = {}

    print(f"base_path: {base_path}")

    for folder in folders:
        q = extract_q(folder)
        clip = extract_clip(folder)

        print(f"Loading data from folder: {folder}, q={q}, clip={clip}")
        
        if q is not None:
            key = (q, clip)
            try:
                df = pd.read_csv(f'{folder}/progress.csv')
                if key not in q_clip_results:
                    q_clip_results[key] = []
                
                df = df[df['Itration']*5000 <= max_steps]

                q_clip_results[key].append(df['Real Det Return'].values)
            except Exception as e:
                print(f"Error reading file in folder: {folder}")
                print(f"Error: {e}")

    return q_clip_results

def filter_results(q_clip_results, q_filter=None, clip_filter=None):
    filtered_results = q_clip_results.copy()
    
    if q_filter is not None:
        filtered_results = {k: v for k, v in filtered_results.items() if k[0] in q_filter}
    
    if clip_filter is not None:
        filtered_results = {k: v for k, v in filtered_results.items() 
                            if (k[1] in clip_filter) or 
                            (k[0] == 1)}
    
    return filtered_results

def plot_data(base_path, q_clip_results, show_confidence_interval, q_filter=None, clip_filter=None):
    # Extract environment name from base path
    env_name = base_path.split('/')[1]
    method_type = base_path.split('/')[3]
    
    # Create output directory for the plot if it doesn't exist
    plot_dir = os.path.join("plots", env_name, method_type)
    os.makedirs(plot_dir, exist_ok=True)

    # Define a set of distinct colors using tableau colors
    DISTINCT_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    # Create distinct visual combinations for each line
    LINE_STYLES = ['-', '--', ':', '-.']

    plt.figure(figsize=(12, 8))
    filtered_results = filter_results(q_clip_results, q_filter, clip_filter)
    
    def create_style_mapping(q_clip_results):
        style_mapping = {}
        unique_combos = sorted(q_clip_results.keys())
        for idx, combo in enumerate(unique_combos):
            color_idx = idx % len(DISTINCT_COLORS)
            style_idx = idx // len(DISTINCT_COLORS)
            style = LINE_STYLES[style_idx % len(LINE_STYLES)]
            style_mapping[combo] = (DISTINCT_COLORS[color_idx], style)
        return style_mapping

    style_mapping = create_style_mapping(filtered_results)

    for idx, ((q, clip), series_list) in enumerate(sorted(filtered_results.items())):
        print([len(series) for series in series_list])
        min_length = min(len(series) for series in series_list)
        aligned_series = [series[:min_length] for series in series_list]
        data = np.array(aligned_series)
        mean = np.mean(data, axis=0)
        episodes = np.arange(min_length) * 5000
        color, line_style = style_mapping[(q, clip)]
        label = f'num_of_nns={q}, ' + ('no_clipping' if clip is None or q == 1 else f'clip={clip}')
        plt.plot(episodes, mean, label=label, linestyle=line_style, color=color)

        if show_confidence_interval:
            stderr = stats.sem(data, axis=0)
            conf_int = stderr * stats.t.ppf((1 + 0.95) / 2, data.shape[0] - 1)
            plt.fill_between(episodes, mean - conf_int, mean + conf_int, alpha=0.2, color=color)

    plt.xlabel('Episodes')
    plt.ylabel('Average Real Det Return')
    plt.title(f'{env_name}: Average Real Det Return by num of NNs and clip value')
    plt.legend()
    plt.grid(True)

    # Generate plot filename based on parameters
    q_values_str = '_'.join(map(str, sorted(q_filter))) if q_filter else 'all_q'
    clip_values_str = '_'.join(map(str, sorted(clip_filter))) if clip_filter else 'all_clip'
    conf_int_str = 'with_conf_int' if show_confidence_interval else 'without_conf_int'
    plot_filename = f'{env_name}_average_real_det_return_{q_values_str}_{conf_int_str}_{clip_values_str}.png'
    plot_path = os.path.join(plot_dir, plot_filename)

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved as: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot Average Real Det Return')
    parser.add_argument('--show_confidence_interval', action='store_true', help='Show confidence intervals in the plot')
    parser.add_argument('--q', type=int, nargs='+', help='Specify which q values to plot')
    parser.add_argument('--clip', type=float, nargs='+', help='Specify which clip values to plot')
    parser.add_argument('--base_path', type=str, help='Specify the base path to load data from')
    parser.add_argument('--max_steps', type=float, default=3e6, help='Maximum number of environment steps to plot (default: 3e6)')
    args = parser.parse_args()

    q_clip_results = load_data(args.base_path, args.max_steps)
    plot_data(args.base_path, q_clip_results, args.show_confidence_interval, args.q, args.clip)

    print("\nNumber of runs for each q and clip value:")
    for (q, clip) in sorted(q_clip_results.keys()):
        print(f"q={q}, clip={clip}: {len(q_clip_results[(q, clip)])} runs")

if __name__ == "__main__":

    # Base path
    # "logs/Ant-v5/exp-16/rkl/"
    # "logs/Walker2d-v5/exp-16/rkl/"
    # "logs/Hopper-v5/exp-4/rkl/"
    # "logs/HalfCheetah-v5/exp-16/rkl/"

    main()


# python plots/plot_reward_over_time.py --q 1 4 --clip 1 2 3 5 10 20 50 0.8 0.5 0.1 100 200 300 500 1000 --base_path logs/HalfCheetah-v5/exp-16/rkl/
# python plots/plot_reward_over_time.py --q 1 4 --clip 1 2 3 5 10 20 50 0.8 0.5 0.1 100 --base_path logs/Walker2d-v5/exp-16/rkl/
# python plots/plot_reward_over_time.py --q 1 4 --clip 1 2 3 5 10 20 50 0.8 0.5 0.1 100 --base_path logs/Ant-v5/exp-16/rkl/
# python plots/plot_reward_over_time.py --q 1 4 --clip 0.1 1 10 100 500 1000 --base_path logs/Humanoid-v5/exp-16/rkl/ --max_steps 1e6


# python plots/plot_reward_over_time.py --q 1 4 --clip 0.1 0.5 1 5 10 50 100 500 1000 --base_path logs/Humanoid-v5/exp-16/maxentirl_sa/