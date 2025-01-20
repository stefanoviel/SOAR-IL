import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_ema(data, alpha=0.1):
    """Calculate exponential moving average"""
    return data.ewm(alpha=alpha, adjust=False).mean()

def plot_aggregated_data(
    df,
    x_column='Running Env Steps',
    y_column='Real Sto Return',
    group_column='q',
    show_confidence=True,
    smoothing_alpha=0.1  # Added smoothing parameter
):
    """
    Create a plot with one line per number of neural networks, aggregated across seeds
    """
    # Constants for plot styling
    TITLE_SIZE = 14
    AXIS_LABEL_SIZE = 12
    TICK_SIZE = 10
    LEGEND_SIZE = 10
    AXIS_UNIT_SIZE = 10
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Convert group column to numeric if it's not already
    df[group_column] = pd.to_numeric(df[group_column])
    
    # Get unique network counts
    network_counts = sorted(df[group_column].unique())
    
    # Create color map
    color_map = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(network_counts)))
    count_to_color = dict(zip(network_counts, color_map))
    
    for count in network_counts:
        # Filter data for current network count
        data_subset = df[df[group_column] == count]
        
        # Group by steps and calculate statistics
        grouped = data_subset.groupby(x_column)[y_column].agg(['mean', 'std']).reset_index()
        
        # Apply smoothing to the mean values
        smoothed_mean = calculate_ema(grouped['mean'], alpha=smoothing_alpha)
        
        # Plot smoothed mean line
        line = plt.plot(grouped[x_column], smoothed_mean,
                       color=count_to_color[count],
                       label=f'Networks={int(count)}',
                       linewidth=2)
        
        if show_confidence:
            # Calculate standard error
            n_seeds = len(data_subset['seed'].unique())
            std_error = grouped['std'] / np.sqrt(n_seeds)
            
            # Smooth the confidence intervals
            smoothed_upper = calculate_ema(grouped['mean'] + std_error, alpha=smoothing_alpha)
            smoothed_lower = calculate_ema(grouped['mean'] - std_error, alpha=smoothing_alpha)
            
            # Add confidence intervals
            plt.fill_between(
                grouped[x_column],
                smoothed_lower,
                smoothed_upper,
                color=count_to_color[count],
                alpha=0.2
            )
    
    # Customize plot
    plt.xlabel('Episode', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Average Return', fontsize=AXIS_LABEL_SIZE)
    plt.grid(True)
    
    # Adjust ticks
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    # Format axes - use scientific notation for both axes
    plt.gca().ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    # Set offset text size for both axes
    plt.gca().xaxis.get_offset_text().set_fontsize(AXIS_UNIT_SIZE)
    plt.gca().yaxis.get_offset_text().set_fontsize(AXIS_UNIT_SIZE)
    
    # Add legend inside the plot
    legend = plt.legend(fontsize=LEGEND_SIZE,
                       loc='lower right',
                       framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

# Example usage:
if __name__ == "__main__":
    
    environments = ["Humanoid", "Walker2d", "Ant", "Hopper"]
    
    for env in environments:
        df = pd.read_csv(f'processed_data/{env}-v5_exp-16_testing_number_nn_data.csv')
        
        # Create plot without confidence intervals, with smoothing
        fig = plot_aggregated_data(df, 
                                 show_confidence=False, 
                                 smoothing_alpha=0.1)  # Adjust alpha value as needed
        fig.savefig(f'plots/aggregated_plot_without_ci_{env}.png',
                    bbox_inches='tight',
                    dpi=300)
        plt.close()