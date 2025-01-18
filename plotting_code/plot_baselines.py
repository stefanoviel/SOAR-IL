import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_over_time(input_folder: str, output_folder: str):
    """
    Reads a 'progress.csv' file from 'input_folder', plots the reward ("Real Det Return") 
    over time ("Itration"), and saves the plot to 'output_folder'.
    """
    # Path to the CSV file
    csv_path = os.path.join(input_folder, 'progress.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract columns
    x = df['Itration'] * 5000
    y = df['Real Det Return']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Real Det Return')
    plt.title('Reward over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Real Det Return')
    plt.legend()
    plt.grid(True)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the plot
    plot_filename = os.path.join(output_folder, 'reward_over_time.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()

if __name__ == '__main__':
    # Example usage (hard-coded paths for demonstration):
    # Adjust these paths as appropriate for your system
    input_folder = '/home/viel/SOAR-IL/logs/Walker2d-v5/exp-16/gail/2025_01_05_15_56_35'
    output_folder = '/home/viel/SOAR-IL/plots/Walker2d-v5/gail'
    
    plot_reward_over_time(input_folder, output_folder)
    print(f"Plot saved successfully in {output_folder}")
