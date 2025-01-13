#!/usr/bin/env bash

# Define the algorithm name (taken from your command "python -m baselines.new_sqil")
algorithm="sqil"

# Define all environments to iterate over
environments=("ant" "halfcheetah" "hopper" "humanoid" "walker2d")

# Define the seeds to iterate over
seeds=(0 1 2 3 4)

# Optionally create an output directory if it doesn't exist
mkdir -p outputs

# Loop over each environment
for env in "${environments[@]}"; do
  # Create a subdirectory for each environment
  mkdir -p "outputs/${env}"

  config_path="configs/samples/agents/${env}.yml"

  # Loop over each seed
  for seed in "${seeds[@]}"; do
    echo "Running ${algorithm} for env=${env} with seed=${seed} ..."

    # Run the command in the background (&), redirect output to .log
    python -u -m baselines.${algorithm} \
        --config "${config_path}" \
        --seed "${seed}" \
        > "outputs/${env}/${algorithm}_seed${seed}.log" 2>&1 &

    sleep 0.5 
  done
done

# Wait for all background tasks to complete
wait

echo "All runs completed."
