#!/usr/bin/env bash

# Define all environments to iterate over
environments=("ant" "halfcheetah" "hopper" "humanoid" "walker2d")


# Optionally create an output directory if it doesn't exist
mkdir -p outputs

# Loop over each environment
for env in "${environments[@]}"; do
  # Create a subdirectory for each environment
  mkdir -p "outputs/${env}"

  config_path="configs/samples/agents/${env}.yml"

  echo "Running ${algorithm} for env=${env} ..."

  # Run the command in the background (&), redirect output to .log
  python -u -m irl_methods.irl_samples_ml_irl \
      --config "${config_path}" \
      --seed 0 \
      > "outputs/${env}/opt_ail.log" 2>&1 &

done

# Wait for all background tasks to complete
wait

echo "All runs completed."
