#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Array of q values
q_values=(1 4)

# Array of seeds
seeds=(42 123 456 789 1024)

# Array of environments
environments=(ant, halfcheetah, hopper, humanoid, walker2d)

# Add delay between process launches
delay_seconds=1

# Loop through environments
for env in "${environments[@]}"; do
    # Create environment-specific output directory
    mkdir -p "outputs/${env}"
    
    # Loop through q values
    for q in "${q_values[@]}"; do
        # Loop through seeds
        for seed in "${seeds[@]}"; do
            echo "Starting process with env=${env}, q_pairs=${q} and seed=${seed}"
            
            # Run process and wait for the specified delay
            (python -m irl_methods.irl_samples_ml_dynamic_clipping --config configs/samples/agents/${env}.yml \
                --num_q_pairs "${q}" \
                --seed "${seed}" \
                > "outputs/${env}/run_q${q}_seed${seed}.log" 2>&1) &
            
            sleep $delay_seconds
        done
    done
done

wait

echo "All experiments launched and completed!"