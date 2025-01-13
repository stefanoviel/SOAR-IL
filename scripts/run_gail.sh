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
algorithms="gail"

# Number of expert trajectories
num_expert_trajs=16

# List of seeds
seeds=(0 1 2 3 4)

# Loop over each algorithm
for algorithm in "${algorithms[@]}"; do
    # Loop over each environment
    for env in "${envs[@]}"; do
        # Create environment-specific output directory
        mkdir -p "outputs/${env}"

        # Loop over each seed
        for seed in "${seeds[@]}"; do
            echo "Running $algorithm for environment: $env with $num_expert_trajs expert trajectories, seed=$seed"

            # Redirect both stdout and stderr to a log file
            python -u -m baselines.gail \
                --env_name "$env" \
                --num_expert_trajs "$num_expert_trajs" \
                --seed "$seed" \
                > "outputs/${env}/${algorithm}_seed${seed}.log" 2>&1 &
            
            # Sleep for a bit to avoid overloading the system
            sleep 0.5
        done
    done
done

# Wait for all background processes to complete
wait

echo "All experiments completed!"
