# IL-SOAR : Imitation Learning with Soft Optimistic Actor cRitic

Note: This repository is based on [f-IRL](https://arxiv.org/abs/2011.04709) [[code](https://github.com/twni2016/f-IRL/tree/main)], which we forked and extended with our methods.

## Installation
- Mujoco environment from OpenAI [Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)
- `pip install -r requirements.txt` 

## File Structure

- Baselines ([AIRL](https://arxiv.org/abs/1710.11248), [GAIL+SAC](https://arxiv.org/abs/1606.03476), [SQIL](https://arxiv.org/abs/1905.11108),  BC): `baselines/`
- Environments: `envs/`
- Configurations: `configs/`
- Implented methods ([CISL](https://arxiv.org/abs/2305.16498), [ML-IRL](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf), [OPT-AIL](https://arxiv.org/abs/2411.00610)*, [f-IRL](https://arxiv.org/abs/2011.04709)) both basic and + SOAR implementation `irl_methods`
- Scripts for experiments: `scripts/`
- Code to make the plots: `plotting_code`
 

\* OPT-AIL is only used as a baseline, it is included in the same file as ML-IRL due to the similarity of the two. 

## Implemented methods

- We use yaml files in `configs/` for experimental configurations for each environment, please change `obj` value (in the first line) for each method, here is the list of `obj` values and corresponding file that needs to be run with it: 
    -  Our methods  
        * f-IRL: `rkl` `irl_methods/irl_samples_f_irl.py`
        * MaxEntIRL: `maxentirl` `irl_methods/irl_samples_ml_irl.py`
        * MaxEntIRL (State Actions): `maxentirl_sa` `irl_methods/irl_samples_ml_irl.py`
        * CISL `cisl` `irl_methods/irl_samples_cisl.py`
    -  Baselines:  
        * OPT-AIL `opt-AIL`  `irl_methods/irl_samples_ml_irl.py`
        * OPT-AIL (SA) `opt-AIL_sa`  `irl_methods/irl_samples_ml_irl.py`
        * SQIL `sqil` `baselines/sqil.py`
        * GAIL: (No config needed) `baselines/gail.py`
        * AIRL: (No config needed) `baselines/airl.py`
- Please keep all the other values in yaml files unchanged to reproduce the results in our paper.

## Command Line Arguments

All the files in the `irl_methods/` folder take the following command line arguments:

- `--config`: (Required) Path to the YAML configuration file containing model and training parameters.
- `--num_q_pairs`: Number of Q-network pairs to use in the ensemble. SOAR if more than 1 is used. 
- `--seed`: Random seed for reproducibility. If not specified, will use value from config file.
- `--uncertainty_coef`: Coefficient that scales the uncertainty-based exploration bonus. Default 1. 
- `--q_std_clip`: Maximum allowed value for Q-value standard deviations. Helps stabilize training by preventing extremely large uncertainty estimates.


## Experiments

All experiments are run with scripts from the `scripts` folder and  be run from the root folder:

### Grid Search Scripts for SOAR
- `grid_search_clip_f_irl.sh`: Grid search over clipping values for f-IRL (1 Q-network) and f-IRL+SOAR (4 Q-networks)
- `grid_search_clip_cisl.sh`: Grid search over clipping values for CISL (1 Q-network) and CISL+SOAR (4 Q-networks) 
- `grid_search_clip_ml_irl.sh`: Grid search over clipping values for MaxEntIRL (1 Q-network) and MaxEntIRL+SOAR (4 Q-networks)
- `grid_search_nn_clip_ml_irl.sh`: Grid search over clipping values for MaxEntIRL with state-action features (1 Q-network) and with SOAR (4 Q-networks)

The SOAR version of each method uses an ensemble of 4 Q-networks, while the base version uses a single Q-network.
The results are saved in the `logs` folder. 

## Baselines
The baselines can be run using the following scripts:

- `run_opt_ail.sh`: Run OPT-AIL baseline experiments  
- `run_gail.sh`: Run GAIL baseline experiments
- `run_sqil.sh`: Run SQIL baseline experiments


## Plots

The plotting process involves two steps:

1. Data Processing:
  ```bash
  python plotting_code/data_processor.py
  ```
  This generates preprocessed data files in the `processed_data` folder. Each file corresponds to a specific environment-method-expert trajectory combination. The preprocessing takes as input the logs file, so you need to run the experiments first. 

2. Generate Plots:
  Plots can be generated using the scripts in the `plotting_code` folder after processing the data.

  ```bash
  # Plot average returns across environments
  python plotting_code/plot_average_envs.py

  # Plot grid search results for appendix
  python plotting_code/plot_returns.py 
  
  # Plot best clipping values by environment and baselines
  python plotting_code/plot_q_bestclip_baselines.py
  ```

Various hyperparameters for plot appearance can be configured directly in the plotting scripts.


## Citation and References

TODO: insert citation

Parts of the codes are used from the references mentioned below:

- [SAC](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac) in part of `common/sac`
