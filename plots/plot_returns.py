import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

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
    
    plt.figure(figsize=(10, 7))
    
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
    # Place legend outside the plot on the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Create filename
    conf_str = "" if show_confidence else "without_conf_int_"
    q_str = "_".join(str(q) for q in sorted(q_values) if q != 1.0)
    clip_str = "_".join(str(clip) for clip in sorted(clip_values))
    filename = f"{env_name}_{method}_{q_str}_{conf_str}{clip_str}.png"
    
    # Create directory structure: plots/env_name/method/
    save_path = Path("plots") / "gridsearch" 
    save_path.mkdir(parents=True, exist_ok=True)
    print('save_path', save_path)
    
    # Save plot with adjusted figure size to accommodate legend
    plt.savefig(save_path / filename, bbox_inches='tight')
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


def create_method_comparison_figure(
    folder_path: str,
    method: str,
    environments: List[str] = ['Hopper-v5', 'Walker2d-v5', 'Humanoid-v5', 'Ant-v5'],
    clip_values: List[float] = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    max_episodes_dict: dict = None,
    show_confidence: bool = True
) -> None:
    """
    Create a figure with subplots comparing different environments for a specific method.
    Each subplot shows the performance across different clipping values.
    """
    folder = Path(folder_path)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Set font sizes
    TITLE_SIZE = 16
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12
    
    # Dictionary to store all unique clip values found in the data
    all_clip_values = set()
    env_clip_data = {}
    
    # First pass: collect all clip values and prepare data
    for env in environments:
        pattern = f"{env}_*_{method}_data.csv"
        files = list(folder.glob(pattern))
        
        if not files:
            print(f"No files found for {env} with method {method}")
            continue
            
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            dfs.append(df)
            
        if not dfs:
            continue
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if max_episodes_dict and env in max_episodes_dict:
            combined_df = combined_df[combined_df['episode'] <= max_episodes_dict[env]]
        
        env_clip_values = combined_df['clip'].unique()
        all_clip_values.update(env_clip_values)
        env_clip_data[env] = combined_df
    
    all_clip_values = sorted(all_clip_values)
    color_map = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(all_clip_values)))
    clip_to_color = dict(zip(all_clip_values, color_map))
    
    legend_elements = []
    
    for idx, env in enumerate(environments):
        ax = axes[idx]
        
        if env not in env_clip_data:
            continue
            
        combined_df = env_clip_data[env]
        
        for clip in all_clip_values:
            data_subset = combined_df[combined_df['clip'] == clip]
            if len(data_subset) > 0:
                grouped = data_subset.groupby('episode')['Real Det Return'].agg(['mean', 'std']).reset_index()
                line = ax.plot(grouped['episode'], grouped['mean'], 
                             color=clip_to_color[clip], 
                             label=f'clip={clip}')
                
                if show_confidence:
                    n_seeds = len(data_subset['seed'].unique())
                    std_error = grouped['std'] / np.sqrt(n_seeds)
                    ax.fill_between(
                        grouped['episode'],
                        grouped['mean'] - std_error,
                        grouped['mean'] + std_error,
                        color=clip_to_color[clip],
                        alpha=0.2
                    )
                
                if not any(f'clip={clip}' == label.get_label() for label in legend_elements):
                    legend_elements.extend(line)
        
        # Customize subplot with larger fonts
        ax.set_title(env.replace('-v5', ''), fontsize=TITLE_SIZE)
        ax.set_xlabel('Episode', fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Average Return', fontsize=AXIS_LABEL_SIZE)
        ax.grid(True)
        
        # Reduce tick frequency and increase font size
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce number of x-ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce number of y-ticks
    
    # Adjust layout first
    plt.tight_layout(rect=[0, 0, 1, 1])  # Leave space at top for title
    
    # Add overall title with larger font - now after tight_layout
    # fig.suptitle(f'Performance Comparison for {method.upper()}', 
    #             fontsize=TITLE_SIZE + 2,
    #             y=0.98)  # Move title up slightly
    
    # Add shared legend below the plots
    legend = fig.legend(legend_elements, 
                       [f'clip={clip}' for clip in all_clip_values],
                       loc='center',
                       bbox_to_anchor=(0.5, -0.05),
                       ncol=4,
                       fontsize=LEGEND_SIZE)
    
    # Save figure with extra bottom margin for legend
    save_path = Path("plots") / "method_comparisons"
    save_path.mkdir(parents=True, exist_ok=True)
    confidence_suffix = "_with_ci" if show_confidence else "_without_ci"
    plt.savefig(save_path / f'{method}_comparison{confidence_suffix}.png', 
                bbox_inches='tight',
                dpi=300,
                bbox_extra_artists=(legend,))
    plt.close()

# The create_all_method_comparisons function remains the same
def create_all_method_comparisons(
    folder_path: str,
    methods: List[str] = ['maxentirl', 'rkl', 'cisl', 'maxentirl_sa'],
    max_episodes_dict: dict = {
        'Hopper-v5': 1e6,
        'Walker2d-v5': 1.5e6,
        'Ant-v5': 1.2e6,
        'Humanoid-v5': 1e6,
        'HalfCheetah-v5': 1.5e6,
    },
    show_confidence: bool = True
):
    for method in methods:
        print(f"Creating comparison figure for {method}...")
        create_method_comparison_figure(
            folder_path, 
            method, 
            max_episodes_dict=max_episodes_dict,
            show_confidence=show_confidence
        )
        print(f"Completed {method}")


# Example usage:
# if __name__ == "__main__":
#     # Example parameters
#     q_values = [1.0, 4.0]
#     clip_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    
#     # Dictionary specifying max episodes for each environment
#     max_episodes_dict = {
#         "Hopper-v5": 1e6,
#         "Walker2d-v5": 1.5e6,
#         "Ant-v5": 1.2e6,
#         "Humanoid-v5": 1e6,
#         "HalfCheetah-v5": 1.5e6,
#         # Add more environments as needed
#     }
    
#     # Multiple files example - now using folder path
#     plot_multiple_files(
#         "processed_data",  # Just specify the folder path
#         q_values,
#         clip_values,
#         show_confidence=False,
#         max_episodes_dict=max_episodes_dict,
#         filter_dynamic_clipping=True
#     )

# Usage example:
if __name__ == "__main__":
    # Create figures with confidence intervals
    # create_all_method_comparisons("processed_data", show_confidence=True)
    # Create figures without confidence intervals
    create_all_method_comparisons("processed_data", show_confidence=False)
    