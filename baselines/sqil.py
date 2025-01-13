"""
SQIL: Instead of learning a reward function (f-IRL), we:
  - Initialize the SAC replay buffer with (s, a, s') from expert data, each having a reward = 1.
  - For the agent’s own transitions, store (s, a, s') with reward = 0.
"""

import sys, os, time
import numpy as np
import torch
import gymnasium as gym
from ruamel.yaml import YAML
import argparse
import random

# We only need SAC and the ReplayBuffer for SQIL:
from common.sac_sqil import ReplayBuffer, SAC

import envs  # Might be your custom envs init
from utils import system, collect, logger, eval
# from utils.plots.train_plot_high_dim import plot_disc
# from utils.plots.train_plot import plot_disc as visual_disc

import datetime
import dateutil.tz
import json, copy
from torch.utils.tensorboard import SummaryWriter


def setup_experiment_seed(seed):
    """Centralized seed setup for the entire experiment."""
    # Set basic Python random seed
    random.seed(seed)
    # Set NumPy seed
    np.random.seed(seed)
    # Set PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensures deterministic behavior (reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Return a random number generator for generating other seeds
    return np.random.RandomState(seed)


def try_evaluate(
    itr: int, 
    sac_agent, 
    env_fn, 
    writer, 
    global_step, 
    v: dict, 
    seed=None
):
    """
    Evaluate the agent by running it in the real environment 
    (with the true environment dynamics, ignoring the SQIL reward trick).
    """
    env_steps = itr * v['sac']['epochs'] * v['env']['T']

    eval_seed = np.random.randint(0, 2**32 - 1) if seed is not None else None

    # Evaluate deterministic policy
    real_return_det = eval.evaluate_real_return(
        sac_agent.get_action, 
        env_fn(), 
        v['irl']['eval_episodes'], 
        v['env']['T'], 
        deterministic=True, 
        seed=eval_seed
    )
    print(f"Real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    # Evaluate stochastic policy
    real_return_sto = eval.evaluate_real_return(
        sac_agent.get_action, 
        env_fn(), 
        v['irl']['eval_episodes'], 
        v['env']['T'], 
        deterministic=False, 
        seed=eval_seed
    )
    print(f"Real sto return avg: {real_return_sto:.2f}")
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))

    # Log to tensorboard
    writer.add_scalar('Returns/Deterministic', real_return_det, global_step)
    writer.add_scalar('Returns/Stochastic', real_return_sto, global_step)

    logger.record_tabular(f"Env Steps", env_steps)

    return real_return_det, real_return_sto


def log_metrics(itr: int, sac_agent, writer: SummaryWriter, v: dict, loss_val=None):
    """
    Log training metrics to tensorboard.
    """
    # Calculate global step
    global_step = itr * v['sac']['epochs'] * v['env']['T']
    

    
    # Optionally log some "loss" or surrogate if needed
    if loss_val is not None:
        writer.add_scalar('Training/Something_Loss', loss_val, global_step)

    return global_step


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 1. Parse arguments and load config
    # ----------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='SQIL training script')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (default: from config)')
    args = parser.parse_args()

    # Load config
    yaml = YAML()
    v = yaml.load(open(args.config))

    # Overwrite some config with command line arguments
    seed = args.seed if args.seed is not None else v['seed']

    print("seed:", seed)

    # ----------------------------------------------------------------------
    # 2. Setup environment, seeds, logging
    # ----------------------------------------------------------------------
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    num_expert_trajs = v['irl']['expert_episodes']

    device = torch.device(
        f"cuda:{v['cuda']}" 
        if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu"
    )
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)

    # Setup main experiment seed
    rng = setup_experiment_seed(seed)

    # Generate separate seeds for different components
    env_seed = rng.randint(0, 2**32 - 1)
    buffer_seed = rng.randint(0, 2**32 - 1)
    network_seed = rng.randint(0, 2**32 - 1)

    system.reproduce(seed)
    pid = os.getpid()

    # Logging
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/sqil"  # directory for logs
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = (
        exp_id
        + '/'
        + now.strftime('%Y_%m_%d_%H_%M_%S')
        + '_seed' + str(seed)
    )
    logger.configure(dir=log_folder)
    writer = SummaryWriter(log_folder)
    print(f"Logging to directory: {log_folder}")

    os.system(f'cp {args.config} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)

    os.makedirs(os.path.join(log_folder, 'plt'), exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'model'), exist_ok=True)

    # ----------------------------------------------------------------------
    # 3. Create Environment
    # ----------------------------------------------------------------------
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    gym_env.reset(seed=env_seed)  # Seed the main environment
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # ----------------------------------------------------------------------
    # 4. Load Expert Trajectories and Put Them in Buffer with R=1
    # ----------------------------------------------------------------------
    expert_states = torch.load(f'expert_data/states/{env_name}.pt').numpy()  
    # This might be shaped (N, T, state_dim). We'll slice `num_expert_trajs`
    expert_states = expert_states[:num_expert_trajs, :, :]  # shape: (E, T, state_dim)


    expert_actions = torch.load(f'expert_data/actions/{env_name}.pt').numpy()
    expert_actions = expert_actions[:num_expert_trajs, :, :]

    # Example flattening:
    E, T, _ = expert_states.shape
    all_expert_s = []
    all_expert_a = []
    all_expert_snext = []
    for eidx in range(E):
        for tidx in range(T - 1):
            s = expert_states[eidx, tidx, :]
            s_next = expert_states[eidx, tidx + 1, :]
            a = expert_actions[eidx, tidx, :]
            all_expert_s.append(s)
            all_expert_a.append(a)
            all_expert_snext.append(s_next)

    all_expert_s = np.array(all_expert_s)
    all_expert_a = np.array(all_expert_a)
    all_expert_snext = np.array(all_expert_snext)
    expert_rewards = np.ones(len(all_expert_s), dtype=np.float32)  # reward = 1
    expert_dones = np.zeros(len(all_expert_s), dtype=bool)         # done=False by default

    print("Expert transitions shape:", all_expert_s.shape, all_expert_a.shape, all_expert_snext.shape)

    # ----------------------------------------------------------------------
    # 5. Initialize Replay Buffer and SAC
    # ----------------------------------------------------------------------
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=state_size,
        act_dim=action_size,
        device=device,
        size=v['sac']['buffer_size']
    )

    # Add expert transitions with reward=1
    for i in range(len(all_expert_s)):
        replay_buffer.store(
            obs=all_expert_s[i],
            act=all_expert_a[i],
            rew=expert_rewards[i],
            next_obs=all_expert_snext[i],
            done=expert_dones[i]
        )
    print(f"Filled replay buffer with {len(all_expert_s)} expert transitions (reward=1).")

    v['sac']['reinitialize'] = True
    # Create SAC agent
    sac_agent = SAC(
        env_fn,
        replay_buffer,
        steps_per_epoch=v['env']['T'],
        update_after=v['env']['T'] * v['sac']['random_explore_episodes'],
        max_ep_len=v['env']['T'],
        seed=network_seed,
        start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
        reward_state_indices=state_indices,  # Not used for a learned reward now, but kept for structure
        device=device,
        **v['sac']
    )

    sac_agent.reward_function = lambda x : np.zeros(x.shape[0])  # Dummy reward function for SQIL 

    # ----------------------------------------------------------------------
    # 6. Main Training Loop for SQIL
    # ----------------------------------------------------------------------
    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    num_iterations = v['irl']['n_itrs']  # Or rename in your config to something else

    for itr in range(num_iterations):
        # -------------------------------------------------------------
        # 6A. Collect on-policy data (with reward=0) and add to buffer
        # -------------------------------------------------------------
        # We'll collect v['irl']['training_trajs'] trajectories 
        # or you might want to collect just 1 epoch worth of steps, etc.
        # The function below collects transitions using the current policy.
        # (s, a, r, s') from the environment, but we override r=0 for SQIL.
        # Note: done is still whatever the environment says for termination.
        #
        # If your environment naturally produces reward, you'll want to
        # override it to 0. We'll do that in code after we collect them.

        states, actions, _  = collect.collect_trajectories_policy_single(
            env=gym_env,
            sac_agent=sac_agent,
            n=v['irl']['training_trajs'],
            state_indices=state_indices
        )


        # Flatten them out:
        n_, T_, _ = states.shape
        for ep_i in range(n_):
            for t_i in range(T_):
                s = states[ep_i, t_i, :]
                a = actions[ep_i, t_i, :]
                s_next = states[ep_i, t_i + 1, :] if t_i < T_ - 1 else states[ep_i, t_i, :]
                d = t_i == T_ - 1

                # For SQIL, override the environment reward to 0
                # (If the environment ends, done might be True, that’s fine.)
                r_sqil = 0.0

                replay_buffer.store(
                    obs=s,
                    act=a,
                    rew=r_sqil,
                    next_obs=s_next,
                    done=d
                )

        print(f"Collected and stored {n_ * T_} on-policy transitions with reward=0.")

        # -------------------------------------------------------------
        # 6B. Update/Train the SAC agent
        # -------------------------------------------------------------
        # This internally samples from the replay buffer,
        # which has both expert transitions (reward=1) and
        # agent transitions (reward=0).
        sac_info = sac_agent.learn_mujoco(print_out=True)

        # -------------------------------------------------------------
        # 6C. Evaluate the policy in the real environment
        # -------------------------------------------------------------
        global_step = log_metrics(itr, sac_agent, writer, v)
        real_return_det, real_return_sto = try_evaluate(
            itr=itr,
            sac_agent=sac_agent,
            env_fn=env_fn,
            writer=writer,
            global_step=global_step,
            v=v,
            seed=seed
        )

        # Save best model so far (according to returns)
        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            torch.save(
                sac_agent.ac.state_dict(),
                os.path.join(
                    logger.get_dir(),
                    f"model/sqil_sac_model_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}.pth"
                )
            )

        # For logging
        logger.record_tabular("Iteration", itr)
        if v['sac']['automatic_alpha_tuning']:
            logger.record_tabular("alpha", sac_agent.alpha.item())
        logger.dump_tabular()

    writer.close()
    print("SQIL Training finished!")
