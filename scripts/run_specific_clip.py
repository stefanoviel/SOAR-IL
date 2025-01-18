#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from typing import Dict, List

# Define optimal clipping values for each method and environment
METHOD_CLIPS = {
    'maxentirl': {
        'ant': 10.0,
        'hopper': 50.0,
        'walker2d': 0.5,
        'humanoid': 5.0
    },
    'rkl': {
        'ant': 0.8,
        'hopper': 50.0,
        'walker2d': 30.0,
        'humanoid': 100.0
    },
    'cisl': {
        'ant': 10.0,
        'hopper': 5.0,
        'walker2d': 0.5,
        'humanoid': 0.1
    },
    'maxentirl_sa': {
        'ant': 5.0,
        'hopper': 10.0,
        'walker2d': 0.5,
        'humanoid': 50.0
    }
}

METHOD_MODULES = {
    'maxentirl': 'irl_samples_ml_irl',
    'rkl': 'irl_samples_f_irl',
    'cisl': 'irl_samples_cisl',
    'maxentirl_sa': 'irl_samples_ml_irl'
}

SEEDS = [42, 123, 456, 789, 1024]
DELAY_SECONDS = 1

def setup_directories() -> None:
    """Create necessary output directories."""
    os.makedirs('outputs', exist_ok=True)
    for env in ['ant', 'hopper', 'walker2d', 'humanoid']:
        os.makedirs(f'outputs/{env}', exist_ok=True)

def run_experiment(method: str, env: str, q_pairs: int, seed: int, clip_value: float = None) -> None:
    """
    Run a single experiment with the specified parameters.
    
    Args:
        method: IRL method to use
        env: Environment name
        q_pairs: Number of Q-pairs
        seed: Random seed
        clip_value: Clipping value (optional)
    """
    module = METHOD_MODULES[method]
    
    # Build command
    cmd = [
        'python', '-m', f'irl_methods.{module}',
        '--config', f'configs/samples/agents/{env}.yml',
        '--num_q_pairs', str(q_pairs),
        '--seed', str(seed)
    ]
    
    # Add clipping value if provided
    if clip_value is not None:
        cmd.extend(['--q_std_clip', str(clip_value)])
        
    # Setup log file name
    log_suffix = 'baseline' if q_pairs == 1 else f'clip{clip_value}'
    log_file = f'outputs/{env}/{method}_q{q_pairs}_seed{seed}_{log_suffix}.log'
    
    print(f"Starting {method} for {env} with q_pairs={q_pairs}, seed={seed}" + 
          (f", clip={clip_value}" if clip_value is not None else ""))
    
    # Run the experiment
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return process

def run_method_experiments(method: str, environments: List[str]) -> None:
    """
    Run all experiments for a specific method and environments.
    
    Args:
        method: IRL method to use
        environments: List of environments to run experiments for
    """
    processes = []
    
    for env in environments:
        clip_value = METHOD_CLIPS[method][env]
        
        # Run baseline (q=1) experiments
        for seed in SEEDS:
            process = run_experiment(method, env, q_pairs=1, seed=seed)
            processes.append(process)
            time.sleep(DELAY_SECONDS)
        
        # Run experiments with q=4 and optimal clipping
        for seed in SEEDS:
            process = run_experiment(method, env, q_pairs=4, seed=seed, clip_value=clip_value)
            processes.append(process)
            time.sleep(DELAY_SECONDS)
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()

def main():
    parser = argparse.ArgumentParser(description='Run IRL experiments')
    parser.add_argument('--method', choices=list(METHOD_MODULES.keys()),
                      help='IRL method to run')
    parser.add_argument('--env', choices=['ant', 'hopper', 'walker2d', 'humanoid', 'all'],
                      default='all', help='Environment to run experiments for')
    
    args = parser.parse_args()
    
    # Setup output directories
    setup_directories()
    
    # Determine environments to run
    environments = ['ant', 'hopper', 'walker2d', 'humanoid'] if args.env == 'all' else [args.env]
    
    # Run experiments
    run_method_experiments(args.method, environments)
    
    print("All experiments completed!")

if __name__ == "__main__":
    main()