#!/usr/bin/env bash

# Define all environments to iterate over
environments=("ant" "hopper" "humanoid" "walker2d")

# Optionally create an output directory if it doesn't exist
mkdir -p outputs

# Loop over each environment
for env in "${environments[@]}"; do
  # Create a subdirectory for each environment
  mkdir -p "outputs/${env}"

  config_path="configs/samples/agents/${env}.yml"

  echo "Running ${algorithm} for env=${env} ..."

  # Run for 4 different seeds
  for seed in 0 1 2 3 4; do
    echo "  - Using seed=${seed} ..."
    python -u -m irl_methods.irl_samples_ml_irl \
        --config "${config_path}" \
        --seed "${seed}" \
        > "outputs/${env}/opt_ail_seed${seed}.log" 2>&1 &

    # Sleep for a bit to avoid overloading the system
    sleep 0.5
  done
done

# Wait for all background tasks to complete
wait

echo "All runs completed."
