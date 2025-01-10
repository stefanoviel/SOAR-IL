#!/usr/bin/env bash

# Create an outputs directory if it doesn't exist
mkdir -p outputs

# List of environments
envs=(
    "Hopper-v5"
    "HalfCheetah-v5"
    "Humanoid-v5"
    "Walker2d-v5"
    "Ant-v5"
)

# List of algorithms
algorithms=(
    # "gail"
    # "airl"
    "sqil"
    # Add more algorithms here if needed
)

# Number of expert trajectories
num_expert_trajs=16

# Loop over each algorithm
for algorithm in "${algorithms[@]}"; do
    # Loop over each environment and launch the algorithm in the background
    for env in "${envs[@]}"; do
        # Create environment-specific output directory (optional)
        mkdir -p "outputs/${env}"

        echo "Running $algorithm for environment: $env with $num_expert_trajs expert trajectories"
        
        # Redirect both stdout and stderr to a log file
        python -u -m baselines."${algorithm}" \
            --env_name "$env" \
            --num_expert_trajs "$num_expert_trajs" \
            > "outputs/${env}/${algorithm}.log" 2>&1 &

    done
done

# Wait for all background processes to complete
wait

echo "All experiments completed!"
