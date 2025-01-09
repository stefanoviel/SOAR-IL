import numpy as np
import torch
import time
from copy import deepcopy

# We assume you already have:
from common.sac_original import SAC, ReplayBuffer
# from your_code import MLPActorCritic  # or whatever you use for actor_critic
import gymnasium as gym 

import argparse
import os
import datetime
import dateutil.tz
import random
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter
import csv



class SQIL(SAC):
    def __init__(
        self,
        env_fn,
        replay_buffer,
        demonstrations,
        log_folder,
        **kwargs
    ):
        """
        SQIL Implementation on top of SAC.
        
        Args:
            demonstrations: list or array of expert transitions (s, a, r, s', d).
                            We will overwrite r=1 for these transitions.
            All other arguments are the same as in SAC.
        """

        # Step 1: Initialize the SAC agent
        super().__init__(
            env_fn=env_fn,
            replay_buffer=replay_buffer,
            **kwargs
        )

        # Step 2: Pre-fill the replay buffer with demonstration data (reward=1)
        self.log_folder = log_folder
        self._store_demonstrations(demonstrations)

    def _store_demonstrations(self, demonstrations):
        """
        Store demonstration transitions in the replay buffer with reward=1.
        demonstrations is expected to be an iterable (s, a, r, s_next, done).
        For SQIL, we override r=1. 
        """
        for (s, a, _, s_next, done) in demonstrations:
            # Force reward=1 for expert transitions
            rew = 1.0
            self.replay_buffer.store(s, a, rew, s_next, done)

        print(f"Loaded {len(demonstrations)} demonstration transitions into buffer.")

    def test_agent(self, n_eval_episodes=5, max_ep_len=None):
        """
        Evaluate the current policy with the real reward.
        
        Args:
            n_eval_episodes (int): How many episodes to run
            max_ep_len (int): Max steps per episode (optional)

        Returns:
            average_return (float): The average cumulative reward 
                                    over the evaluation episodes.
        """
        returns = []
        for _ in range(n_eval_episodes):
            o, info = self.test_env.reset()
            
            ep_ret = 0
            ep_len = 0
            done = False
            while not done:
                # Use the current policy
                a = self.get_action(o)
                o, r, done, truncated, _ = self.test_env.step(a)
                
                ep_ret += r
                ep_len += 1
                if max_ep_len and (ep_len >= max_ep_len):
                    # if we have a hard limit for steps
                    break
            
            returns.append(ep_ret)
        average_return = float(np.mean(returns))
        return average_return
    
    def learn_sqil(self, total_steps,  print_out=False):

        start_time = time.time()

        # Create (or overwrite) the CSV in the current log folder.
        # We assume your agent has a reference to `self.log_folder`, 
        # or you can pass it in as an argument. If not, 
        # pass it in from outside or store it somewhere accessible.
        csv_path = os.path.join(self.log_folder, 'progress.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(["episode", "Real Det Return"])
        
        o, info = self.env.reset(seed=self.seed)
        current_seed = self.seed
        ep_len = 0

        print(f"Training SQIL agent with total steps {total_steps:d}.")
        test_rets = []
        alphas = []
        log_pis = []
        test_time_steps = []

        for t in range(total_steps):
            # (1) Collect actions
            if self.replay_buffer.size > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # (2) Step in the env with reward=0 for on-policy transitions
            o2, r_env, done, _, _ = self.env.step(a)
            rew = 0.0
            ep_len += 1
            self.replay_buffer.store(o, a, rew, o2, done)
            o = o2

            # (3) Handle terminal
            time_out = (ep_len == self.max_ep_len)
            terminal = done or time_out
            if terminal:
                current_seed += 1
                o, info = self.env.reset(seed=current_seed)
                ep_len = 0

            # (4) SAC updates
            if t >= self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    loss_q, loss_pi, log_pi = self.update(batch)

            # (5) Logging & evaluation
            if (t + 1) % self.log_step_interval == 0:
                n_eval_episodes = 5
                eval_return = self.test_agent(n_eval_episodes=n_eval_episodes,
                                            max_ep_len=self.max_ep_len)
                print(f"[{t+1}/{total_steps}] Evaluation over {n_eval_episodes} episodes: "
                    f"Avg return = {eval_return:.2f}")
                
                # Append result to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([t+1, eval_return])
                
                test_rets.append(eval_return)
                test_time_steps.append(t + 1)

        print(f"Done SQIL training. Elapsed: {time.time() - start_time:.2f}s.")
        return test_rets, alphas, log_pis, test_time_steps


def setup_experiment_seed(seed):
    """Centralized seed setup for the entire experiment."""
    # Python built-in
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For full reproducibility (slower):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.RandomState(seed)



if __name__ == "__main__":
    # ------------------------------------------------------
    # 1) Parse arguments
    # ------------------------------------------------------
    parser = argparse.ArgumentParser(description='SQIL Training Script')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--expert_episodes', type=int, default=1,
                        help='Number of expert trajectories to load')
    parser.add_argument("--train_steps", type=int, default=1_500_000) 
    parser.add_argument('--env_name', type=str, default='Hopper-v5',
                        help='Gym environment name')
    parser.add_argument('--cuda', type=int, default=-1,
                        help='CUDA device index (set < 0 for CPU)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    args = parser.parse_args()

    # ------------------------------------------------------
    # 2) Set up seeds, device, logging
    # ------------------------------------------------------
    rng = setup_experiment_seed(args.seed)

    # Device
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )
    print("Using device:", device)

    # Logging
    env_name = args.env_name
    num_expert_trajs = args.expert_episodes

    exp_id = f"{args.log_dir}/{env_name}/exp-{num_expert_trajs}/sqil"
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = os.path.join(
        exp_id, now.strftime('%Y_%m_%d_%H_%M_%S') + f"_seed{args.seed}"
    )
    writer = SummaryWriter(log_folder)
    print(f"Logging to directory: {log_folder}")

    # ------------------------------------------------------
    # 3) Create environment
    # ------------------------------------------------------
    env_fn = lambda: gym.make(env_name)
    env = env_fn()
    env.reset(seed=args.seed)  # or rng.randint(2**32-1)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # For demonstration, we do not filter any state indices.
    # If you still need to subset states, you can parse them from CLI too.
    state_indices = list(range(state_size))

    # ------------------------------------------------------
    # 4) Load expert data (states + actions)
    # ------------------------------------------------------
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

    demonstration_data = []
    N, T, _ = expert_trajs.shape
    for traj_idx in range(N):
        for t in range(T - 1):
            s  = expert_trajs[traj_idx, t]
            a  = expert_a[traj_idx, t]
            s2 = expert_trajs[traj_idx, t + 1]
            done = 0.0  
            demonstration_data.append((s, a, 0.0, s2, done))  # reward=0 it will get overriden later

    # ------------------------------------------------------
    # 5) Create SQIL agent
    # ------------------------------------------------------
    replay_buffer = ReplayBuffer(
        obs_dim=state_size,
        act_dim=action_size,
        device=device,
        size=int(1e6),
    )

    sqil_agent = SQIL(
        env_fn=env_fn,
        replay_buffer=replay_buffer,
        demonstrations=demonstration_data,
        log_folder=log_folder,
    )

    # ------------------------------------------------------
    # 6) Train the SQIL agent
    # ------------------------------------------------------
    t_start = time.time()
    returns, alphas, log_pis, timesteps = sqil_agent.learn_sqil(args.train_steps, print_out=True)
    print(f"Done SQIL training in {time.time() - t_start:.1f}s.")


    writer.close()


# python -m baselines.sqil_my_sac --env_name Hopper-v5  --expert_episodes 16