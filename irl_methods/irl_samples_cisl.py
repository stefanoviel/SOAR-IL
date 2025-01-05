'''
f-IRL: Extract policy/reward from specified expert samples
'''
import sys, os, time
import numpy as np
import torch
import gymnasium as gym 
from ruamel.yaml import YAML
import argparse
import random

from irl_methods.divs.f_div_disc import f_div_disc_loss
from irl_methods.divs.f_div import maxentirl_loss
from irl_methods.divs.ipm import ipm_loss
from irl_methods.models.reward import MLPReward
from irl_methods.models.discrim import SMMIRLDisc as Disc
from irl_methods.models.discrim import SMMIRLCritic as Critic
# from common.sac import ReplayBuffer, SAC
from common.sac_irl_methods import ReplayBuffer, SAC

import envs
from utils import system, collect, logger, eval
from utils.plots.train_plot_high_dim import plot_disc
from utils.plots.train_plot import plot_disc as visual_disc

import datetime
import dateutil.tz
import json, copy
from torch.utils.tensorboard import SummaryWriter

from irl_methods.models.cisl_reward import CoherentReward

def try_evaluate(itr: int, policy_type: str, sac_info, writer, global_step, seed=None):
    """Add seed parameter and pass it through to evaluation functions"""
    assert policy_type in ["Running"]
    update_time = itr * v['reward']['gradient_step']
    env_steps = itr * v['sac']['epochs'] * v['env']['T']
    agent_emp_states = samples[0].copy()
    assert agent_emp_states.shape[0] == v['irl']['training_trajs']

    # Generate evaluation seed
    eval_seed = np.random.randint(0, 2**32-1) if seed is not None else None
    
    metrics = eval.KL_summary(expert_samples, agent_emp_states.reshape(-1, agent_emp_states.shape[2]), 
                         env_steps, policy_type, seed=eval_seed)
                         
    # Pass seed to evaluation functions
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], True, seed=eval_seed)
    print(f"real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], False, seed=eval_seed)
    print(f"real sto return avg: {real_return_sto:.2f}")
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))
    
    # Log to tensorboard
    writer.add_scalar('Returns/Deterministic', real_return_det, global_step)
    writer.add_scalar('Returns/Stochastic', real_return_sto, global_step)

    return real_return_det, real_return_sto

def log_metrics(itr: int, sac_agent, uncertainty_coef: float,  writer: SummaryWriter, v: dict):
    """
    Log training metrics to tensorboard
    
    Args:
        itr: Current iteration number
        sac_agent: SAC agent instance
        uncertainty_coef: Uncertainty coefficient for exploration
        loss: Current reward loss value
        writer: Tensorboard SummaryWriter instance
        v: Config dictionary
    """
    # Calculate global step
    global_step = itr * v['sac']['epochs'] * v['env']['T']
    
    # Log average Q-values and their std
    q_values, q_stds = sac_agent.get_q_stats()
    writer.add_scalar('SAC/Average_Q', q_values, global_step)
    writer.add_scalar('SAC/Q_Std', q_stds, global_step)
    
    
    return global_step

def setup_experiment_seed(seed):
    """Centralized seed setup for the entire experiment"""
    # Set basic Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Return a random number generator for generating other seeds
    return np.random.RandomState(seed)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='f-IRL training script')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    parser.add_argument('--num_q_pairs', type=int, default=1,
                      help='Number of Q-network pairs (default: 1)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (default: from config)')
    parser.add_argument('--uncertainty_coef', type=float, default=1.0,
                      help='Uncertainty coefficient for exploration (default: 1.0)')
    parser.add_argument('--q_std_clip', type=float, default=1.0,
                      help='Maximum value to clip Q-value standard deviations (default: 1.0)')
    

    args = parser.parse_args()

    # Load config
    yaml = YAML()
    v = yaml.load(open(args.config))

    # assumptions
    assert v['obj'] in ['cisl']
    assert v['IS'] == False
    
    # Use parsed arguments
    num_q_pairs = args.num_q_pairs
    seed = args.seed if args.seed is not None else v['seed']
    uncertainty_coef = args.uncertainty_coef
    q_std_clip = args.q_std_clip

    print("num_q_pairs:", num_q_pairs)
    print("seed:", seed)
    print("uncertainty_coef:", uncertainty_coef)
    print("q_std_clip:", q_std_clip)

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    
    # Setup main experiment seed
    seed = args.seed if args.seed is not None else v['seed']
    rng = setup_experiment_seed(seed)
    
    # Generate separate seeds for different components
    env_seed = rng.randint(0, 2**32-1)
    buffer_seed = rng.randint(0, 2**32-1)
    network_seed = rng.randint(0, 2**32-1)
    
    system.reproduce(seed)
    pid=os.getpid()
    
    # logs
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S') + f'_q{num_q_pairs}_seed{seed}' + f'_qstd{q_std_clip}'
    logger.configure(dir=log_folder)            
    writer = SummaryWriter(log_folder)
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp firl/irl_samples.py {log_folder}')
    os.system(f'cp {args.config} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    gym_env.reset(seed=env_seed)  # Seed the main environment
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    expert_trajs = torch.load(f'expert_data/states/{env_name}.pt').numpy()[:, :, state_indices]
    expert_trajs = expert_trajs[:num_expert_trajs, :, :] # select first expert_episodes
    expert_samples = expert_trajs.copy().reshape(-1, len(state_indices))
    print(expert_trajs.shape, expert_samples.shape) # ignored starting state

    # load expert actions
    expert_a = torch.load(f'expert_data/actions/{env_name}.pt').numpy()[:, :, :]
    expert_a = expert_a[:num_expert_trajs, :, :] # select first expert_episodes
    expert_a_samples = expert_a.copy().reshape(-1, action_size)
    expert_samples_sa=np.concatenate([expert_samples,expert_a_samples],1)
    print(expert_trajs.shape, expert_samples_sa.shape) # ignored starting state

    # Initialize reward function
    reward_func = CoherentReward(
        state_dim=state_size,
        action_dim=action_size,
        alpha=v['reward']['alpha'],
        hidden_sizes=v['reward']['hidden_sizes'],
        device=device,
        action_low=gym_env.action_space.low[0],  # Assume symmetric action space (true in Mujoco envs)
        action_high=gym_env.action_space.high[0],
        learning_rate=v['reward']['lr'],
        weight_decay=v['reward']['weight_decay']
    )


    # Train the reward function using expert demonstrations
    reward_func.train_policy(expert_samples, expert_a_samples, num_epochs=10, batch_size=v['reward']['batch_size'])

    max_real_return_det, max_real_return_sto = -np.inf, -np.inf

    for itr in range(v['irl']['n_itrs']):

        if v['sac']['reinitialize'] or itr == 0:
            # Reset SAC agent with old policy, new environment, and new replay buffer
            print("Reinitializing sac")
            replay_buffer = ReplayBuffer(
                state_size, 
                action_size,
                device=device,
                size=v['sac']['buffer_size'])
                
            sac_agent = SAC(env_fn, replay_buffer,
                steps_per_epoch=v['env']['T'],
                update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
                max_ep_len=v['env']['T'],
                seed=network_seed,
                start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                reward_state_indices=state_indices,
                device=device,
                num_q_pairs=int(num_q_pairs),
                uncertainty_coef=uncertainty_coef,
                q_std_clip=q_std_clip,
                use_actions_for_reward=True,
                **v['sac']
            )
        
            sac_agent.reward_function = reward_func.get_reward

        sac_info = sac_agent.learn_mujoco(print_out=True) 

        samples = collect.collect_trajectories_policy_single(gym_env, sac_agent, 
                        n = v['irl']['training_trajs'], state_indices=state_indices)

        # Log metrics and get global step
        global_step = log_metrics(itr, sac_agent, uncertainty_coef, writer, v)
        
        # evaluating the learned reward
        real_return_det, real_return_sto = try_evaluate(itr, "Running", sac_info, writer, global_step, seed=seed)
        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto

            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), 
                    f"model/reward_model_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}.pkl"))

        logger.record_tabular("Itration", itr)
        if v['sac']['automatic_alpha_tuning']:
            logger.record_tabular("alpha", sac_agent.alpha.item())

        logger.dump_tabular()

    writer.close()


# python -m irl_methods.irl_samples_cisl --config configs/samples/agents/halfcheetah.yml --num_q_pairs 1 --seed 0 --uncertainty_coef 1.0 --q_std_clip 1.0 