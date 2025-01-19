import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_ema(data, alpha=0.1):
    """Calculate exponential moving average"""
    return data.ewm(alpha=alpha, adjust=False).mean()

def plot_aggregated_data(df, x_column='Running Env Steps', y_column='Real Sto Return', 
                        group_column='q', alpha=0.1):
    """
    Create a plot with one line per q value, aggregated across seeds
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    x_column : str
        Column name for x-axis
    y_column : str
        Column name for y-axis
    group_column : str
        Column name for grouping (q values)
    alpha : float
        Smoothing factor for EMA
    """
    # Convert q to numeric if it's not already
    df[group_column] = pd.to_numeric(df[group_column])
    
    # Create figure with larger size
    plt.figure(figsize=(12, 6))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Get unique q values
    q_values = sorted(df[group_column].unique())
    
    # Create color palette
    colors = sns.color_palette("husl", n_colors=len(q_values))
    
    for q_val, color in zip(q_values, colors):
        # Filter data for current q value
        q_data = df[df[group_column] == q_val]
        
        # Group by environment steps and calculate mean across seeds
        grouped = q_data.groupby(x_column)[y_column].mean().reset_index()
        
        # Apply EMA smoothing
        smoothed_y = calculate_ema(grouped[y_column], alpha=alpha)
        
        # Plot the line
        plt.plot(grouped[x_column], smoothed_y, label=f'Networks: {int(q_val)}', 
                color=color, linewidth=2)
    
    # Customize the plot
    plt.xlabel('Environment Steps', fontsize=12)
    plt.ylabel('Return', fontsize=12)
    plt.legend(title='Number of Neural Networks', bbox_to_anchor=(1.05, 1), 
              loc='upper left', fontsize=10)
    plt.tight_layout()
    
    return plt.gcf()

# Example usage:
df = pd.read_csv('/home/viel/SOAR-IL/processed_data/Hopper-v5_exp-4_hopper_final_data.csv')
fig = plot_aggregated_data(df)
fig.savefig('plots/aggregated_plot.png')