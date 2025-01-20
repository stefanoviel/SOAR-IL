#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Array of number of neural networks to use
q_pairs=(3 5 10)

# Array of seeds
seeds=(42 123 456)

# Array of q_std_clip values
q_std_clip=(0.1 0.5 1 5 10 50 100 500)

# Add delay between process launches
delay_seconds=1

# Set environment to ant
env="ant"

# Create environment-specific output directory if it doesn't exist
mkdir -p "outputs/${env}"

# Run grid search over number of neural networks and clipping values
for q in "${q_pairs[@]}"; do
    for clip in "${q_std_clip[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Starting process with q_pairs=${q} and seed=${seed} and q_std_clip=${clip}"
            
            # Run process and wait for the specified delay
            (python -m irl_methods.irl_samples_ml_irl --config configs/samples/agents/${env}.yml \
                --num_q_pairs "${q}" \
                --seed "${seed}" \
                --q_std_clip "${clip}" \
                > "outputs/${env}/run_q${q}_seed${seed}_clip${clip}.log" 2>&1) &
            
            sleep $delay_seconds
        done
    done
done

# Wait for all background processes to complete
wait

echo "All experiments launched and completed!"