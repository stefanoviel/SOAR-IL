#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Array of environments
environments=("ant" "hopper" "humanoid" "walker2d")

# Fixed q value
q=4

# Array of seeds
seeds=(42 123 456)

# Array of q_std_clip values
q_std_clip=(0.1 0.5 1 10 50) 

# Add delay between process launches
delay_seconds=0.5

# Iterate over all environments
for env in "${environments[@]}"; do
    # Create environment-specific output directory if it doesn't exist
    mkdir -p "outputs/${env}"

    for clip in "${q_std_clip[@]}"; do
        for seed in "${seeds[@]}"; do

            echo "Starting process for environment=${env} with q_pairs=${q}, seed=${seed}, and q_std_clip=${clip}"
            
            # Run process and wait for the specified delay
            (python -m irl_methods.irl_samples_cisl --config configs/samples/agents/${env}.yml \
                --num_q_pairs "${q}" \
                --seed "${seed}" \
                --q_std_clip "${clip}" \
                > "outputs/${env}/run_q${q}_seed${seed}_clip${clip}.log" 2>&1) &
            
            sleep $delay_seconds

        done
    done

done

q=1

for env in "${environments[@]}"; do
    # Create environment-specific output directory if it doesn't exist
    mkdir -p "outputs/${env}"

    for seed in "${seeds[@]}"; do

        echo "Starting process for environment=${env} with q_pairs=${q} and seed=${seed}"
        
        # Run process and wait for the specified delay
        (python -m irl_methods.irl_samples_cisl --config configs/samples/agents/${env}.yml \
            --num_q_pairs "${q}" \
            --seed "${seed}" \
            > "outputs/${env}/run_q${q}_seed${seed}.log" 2>&1) &
        
        sleep $delay_seconds

    done

done

wait

echo "All experiments launched and completed!"
