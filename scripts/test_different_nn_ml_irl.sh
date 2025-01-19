#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Configuration
env="walker2d"
q_std_clip=5  # Fixed clipping value
delay_seconds=1

# Arrays for parameters
seeds=(42 123 456)  # 3 different seeds
q_pairs=(1 3 4 5 10)  # Different numbers of neural networks

# Create environment-specific output directory if it doesn't exist
mkdir -p "outputs/${env}"

# Loop through each number of neural networks
for q in "${q_pairs[@]}"; do
    echo "Starting experiments with ${q} neural networks..."
    
    # Loop through seeds for each q value
    for seed in "${seeds[@]}"; do
        echo "Starting process with q_pairs=${q} and seed=${seed} and q_std_clip=${q_std_clip}"
        
        # Run process and wait for the specified delay
        (python -m irl_methods.irl_samples_ml_irl \
            --config configs/samples/agents/${env}.yml \
            --num_q_pairs "${q}" \
            --seed "${seed}" \
            --q_std_clip "${q_std_clip}" \
            > "outputs/${env}/run_q${q}_seed${seed}_clip${q_std_clip}.log" 2>&1) &
        
        sleep $delay_seconds
    done
done

# Wait for all background processes to complete
wait

echo "All experiments launched and completed!"