import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
from pathlib import Path

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load and process a single CSV file."""
    return pd.read_csv(file_path)

def create_returns_plot(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    q_values: List[float],
    clip_values: List[float],
    env_name: str,
    method: str,
    show_confidence: bool = True,
    max_episodes: Union[None, int] = None
) -> None:
    """Create and save a plot of returns for specified q and clip values."""
    
    plt.figure(figsize=(10, 6))
    
    # Convert single DataFrame to list for consistent processing
    if isinstance(data, pd.DataFrame):
        data = [data]
    
    # Combine all DataFrames
    df = pd.concat(data, ignore_index=True)
    
    # Truncate data if max_episodes is specified
    if max_episodes is not None:
        df = df[df['episode'] <= max_episodes]
    
    # Plot q=1 first (without clip value in label)
    if 1.0 in q_values:
        q1_data = df[df['q'] == 1.0]
        if len(q1_data) > 0:
            grouped = q1_data.groupby('episode')['Real Det Return'].agg(['mean', 'std']).reset_index()
            plt.plot(grouped['episode'], grouped['mean'], label=f'q=1')
            
            if show_confidence:
                n_seeds = len(q1_data['seed'].unique())
                std_error = grouped['std'] / np.sqrt(n_seeds)
                plt.fill_between(
                    grouped['episode'],
                    grouped['mean'] - std_error,
                    grouped['mean'] + std_error,
                    alpha=0.2
                )
    
    # Plot other q values with their clip values
    for q in q_values:
        if q == 1.0:  # Skip q=1 as it's already plotted
            continue
            
        if "dynamic_clipping" in method:
            # For dynamic clipping, only filter by q value
            data_subset = df[df['q'] == q]
            label = f'q={q}'
            
            if len(data_subset) > 0:
                grouped = data_subset.groupby('episode')['Real Det Return'].agg(['mean', 'std']).reset_index()
                plt.plot(grouped['episode'], grouped['mean'], label=label)
                
                if show_confidence:
                    n_seeds = len(data_subset['seed'].unique())
                    std_error = grouped['std'] / np.sqrt(n_seeds)
                    plt.fill_between(
                        grouped['episode'],
                        grouped['mean'] - std_error,
                        grouped['mean'] + std_error,
                        alpha=0.2
                    )
        else:
            # For static clipping, filter by both q and clip values
            for clip in clip_values:
                data_subset = df[(df['q'] == q) & (df['clip'] == clip)]
                label = f'q={q}, clip={clip}'
                

                if len(data_subset) > 0:
                    grouped = data_subset.groupby('episode')['Real Det Return'].agg(['mean', 'std']).reset_index()
                    plt.plot(grouped['episode'], grouped['mean'], label=label)
                    
                    if show_confidence:
                        n_seeds = len(data_subset['seed'].unique())
                        std_error = grouped['std'] / np.sqrt(n_seeds)
                        plt.fill_between(
                            grouped['episode'],
                            grouped['mean'] - std_error,
                            grouped['mean'] + std_error,
                            alpha=0.2
                        )
    
    plt.xlabel('Episode')
    plt.ylabel('Real Det Return')
    plt.title(f'Average Returns vs Episodes for {env_name} ({method})')
    plt.legend()
    plt.grid(True)
    
    # Create filename
    conf_str = "" if show_confidence else "without_conf_int_"
    q_str = "_".join(str(q) for q in sorted(q_values) if q != 1.0)
    clip_str = "_".join(str(clip) for clip in sorted(clip_values))
    filename = f"{env_name}_average_real_det_return_{q_str}_{conf_str}{clip_str}.png"
    
    # Create directory structure: plots/env_name/method/
    save_path = Path("plots") / env_name / method
    save_path.mkdir(parents=True, exist_ok=True)
    print('save_path', save_path)
    
    # Save plot
    plt.savefig(save_path / filename)
    plt.close()

def plot_single_file(
    file_path: str,
    q_values: List[float],
    clip_values: List[float],
    show_confidence: bool = True,
    max_episodes_dict: dict = None,
    filter_dynamic_clipping: bool = False
) -> None:
    """Create plot from a single CSV file."""
    df = load_and_process_data(file_path)
    
    # Extract env_name (only until first underscore) and method from file path
    path_parts = Path(file_path).parts
    full_env_name = next(part for part in path_parts if "v" in part)
    env_name = full_env_name.split('_')[0]  # Take only the part before first underscore
    filename = path_parts[-1]
    method = filename.split('_')[2]  # Assuming 'cisl' is always the third component

    method = "_".join(filename.split('_')[2:-1])

    print("full_env_name", full_env_name, "env_name", env_name, "method", method)
    
    # Get max episodes for this environment
    max_episodes = max_episodes_dict.get(env_name, None) if max_episodes_dict else None
    
    create_returns_plot(df, q_values, clip_values, env_name, method, show_confidence, max_episodes)

def plot_multiple_files(
    folder_path: str,
    q_values: List[float],
    clip_values: List[float],
    show_confidence: bool = True,
    max_episodes_dict: dict = None,
    filter_dynamic_clipping: bool = False
) -> None:
    """Create plot from all CSV files in the specified folder."""
    # Convert string path to Path object
    folder = Path(folder_path)
    
    # Get all CSV files in the folder
    file_paths = list(folder.glob("**/*.csv"))  # ** means search recursively through subfolders
    
    if not file_paths:
        print(f"No CSV files found in {folder_path}")
        return
        
    # Process each file individually
    for file_path in file_paths:
        print("Processing", file_path)
        df = load_and_process_data(str(file_path))
        
        # Extract env_name and method from file path
        full_env_name = next(part for part in file_path.parts if "v" in part)
        env_name = full_env_name.split('_')[0]  # Take only the part before first underscore
        method = "_".join(file_path.name.split('_')[2:]).replace("_data.csv", "")  # Assuming method is always the third component
        
        # Get max episodes for this environment
        max_episodes = max_episodes_dict.get(env_name, None) if max_episodes_dict else None
        
        # Create individual plot for this file
        create_returns_plot([df], q_values, clip_values, env_name, method, show_confidence, max_episodes)

# Example usage:
if __name__ == "__main__":
    # Example parameters
    q_values = [1.0, 4.0]
    clip_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    # clip_values = [0.5]
    
    # Dictionary specifying max episodes for each environment
    max_episodes_dict = {
        "Hopper-v5": 1e6,
        "Walker2d-v5": 1.5e6,
        "Ant-v5": 1.2e6,
        "Humanoid-v5": 1e6,
        "HalfCheetah-v5": 1.5e6,
        # Add more environments as needed
    }
    
    # Single file example
    # plot_single_file(
    #     "plots/cached_data/Walker2d-v5_exp-16_maxentirl_sa_data.csv",
    #     q_values,
    #     clip_values,
    #     show_confidence=False,
    #     max_episodes_dict=max_episodes_dict,
    #     filter_dynamic_clipping=True
    # )
    
    # Multiple files example - now using folder path
    plot_multiple_files(
        "processed_data",  # Just specify the folder path
        q_values,
        clip_values,
        show_confidence=False,
        max_episodes_dict=max_episodes_dict,
        filter_dynamic_clipping=True
    )
