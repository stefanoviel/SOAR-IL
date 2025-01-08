import os
import argparse
import datetime
import dateutil.tz

import numpy as np
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer

# Imitation library function to load pre-saved demonstrations
from baselines.util import load_expert_trajectories_imitation

def parse_args():
    parser = argparse.ArgumentParser(description="Train SQIL on a MuJoCo env from raw .pt expert data.")
    parser.add_argument("--env_name", type=str, default="Ant-v5",
                        help="MuJoCo Gym environment ID, e.g. Ant-v5.")
    parser.add_argument("--num_expert_trajs", type=int, default=5,
                        help="Number of expert episodes to use.")
    parser.add_argument("--train_steps", type=int, default=1_000_000,
                        help="Number of training timesteps.")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes.")
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Frequency (in timesteps) of policy evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

class ZeroRewardEnv(gym.RewardWrapper):
    """
    Wraps a Gym environment to override rewards with 0.
    This ensures that the agent receives *no real environment reward*,
    so it must rely on demonstration-based rewards for SQIL.
    """
    def reward(self, reward):
        return 0.0

def main():
    args = parse_args()
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    # Logging directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_dir = f"logs/{args.env_name}/exp-{args.num_expert_trajs}/sqil/{args.num_expert_trajs}/" + now.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Create a VecEnv but with zeroed-out rewards
    def make_env():
        env = gym.make(args.env_name)
        env = ZeroRewardEnv(env)
        return env
    venv = DummyVecEnv([make_env])  # vectorized env with a single subprocess

    # ---------------------------
    # 1) Load demonstrations
    # ---------------------------
    demonstrations = load_expert_trajectories_imitation(
        args.env_name, 
        args.num_expert_trajs
    )
    # demonstrations is a list of `imitation.data.types.Trajectory` objects

    # ---------------------------
    # 2) Create an off-policy RL algorithm
    #    We'll use SAC for continuous control
    # ---------------------------
    model = SAC(
        policy="MlpPolicy",
        env=venv,
        seed=args.seed,
        device=device,
        verbose=0,
        # You could tweak various hyperparams here
    )

    # ---------------------------
    # 3) Create a single ReplayBuffer
    #    3a) Fill it with demonstration transitions (reward=1)
    #    3b) We'll keep adding env transitions (reward=0) in training
    # ---------------------------
    # Determine buffer capacity. Typically, you want a large buffer,
    # but for demonstration, let's do something modest:
    buffer_size = 1_000_000
    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        device=device,
        optimize_memory_usage=False,
    )

    # 3a) Fill with demonstration transitions => reward=1
    for traj in demonstrations:
        for i in range(len(traj.obs) - 1):
            obs = traj.obs[i]
            act = traj.acts[i]
            next_obs = traj.obs[i+1]
            done = False
            # SQIL: demonstration transitions = reward=1
            reward = 1.0
            replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=act,
                reward=reward,
                done=done,
                infos=[{}]  # <-- list of length 1 (or some other info dict)
            )

    print(f"Added {replay_buffer.size()} demonstration transitions to the replay buffer.")

    # ---------------------------
    # 4) SQIL training loop
    #    We'll do manual training, so we can interleave:
    #      - env rollout (with reward=0)
    #      - off-policy updates on the combined buffer
    # ---------------------------
    total_timesteps = 0
    eval_count = 0
    next_eval = args.eval_freq

    # Keep track of the last observation for each env in the vectorized environment
    obs_array = venv.reset()

    # Simple helper function for evaluation
    def evaluate_policy(n_episodes=5):
        returns = []
        for _ in range(n_episodes):
            done = False
            obs = venv.reset()
            ep_ret = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, dones, infos = venv.step(action)
                ep_ret += rew[0]
                done = dones[0]
            returns.append(ep_ret)
        return np.mean(returns), returns

    # Evaluate untrained
    mean_rew, _ = evaluate_policy(args.eval_episodes)
    print(f"[Before Training] Mean Return: {mean_rew:.2f}")
    writer.add_scalar("eval/untrained_mean_return", mean_rew, total_timesteps)

    # We’ll do the training in a loop:
    #   collect env transitions -> set reward=0 -> add to buffer -> train
    max_episode_steps = 1000  # You can adjust or read from env.spec
    with tqdm(total=args.train_steps, desc="Training SQIL") as pbar:
        while total_timesteps < args.train_steps:
            # 1) Roll out a single step in the environment
            action, _ = model.predict(obs_array, deterministic=False)
            next_obs_array, rew_array, dones_array, infos_array = venv.step(action)

            # *SQIL step*: environment transitions => reward=0
            for i in range(len(obs_array)):
                obs = obs_array[i]
                act = action[i]
                next_obs = next_obs_array[i]
                done = dones_array[i]
                reward = 0.0  # Zero for environment transitions
                replay_buffer.add(obs, next_obs, act, reward, float(done), infos=[{}])

            obs_array = next_obs_array

            # If done, reset
            for i, done in enumerate(dones_array):
                if done:
                    obs_array[i] = venv.reset()[i]

            total_timesteps += 1
            pbar.update(1)

            # 2) Off-policy training step
            # By default, SB3’s SAC calls `train()` automatically when `.learn()` is called,
            # but we want more control, so we do it manually via private methods.
            # A simpler approach is to just call `.learn(total_timesteps, ...)`,
            # but that will handle exploration and training in the usual SB3 loop.
            # Instead, we do:
            if replay_buffer.size() > model.batch_size:
                model.agent.train_frequency = 1  # ensures training each step
                model.gradient_steps = 1  # train for 1 gradient step
                # Prepare a batch from the replay buffer
                replay_data = replay_buffer.sample(model.batch_size)
                train_logs = model.agent.train(replay_data)
                for k, val in train_logs.items():
                    if isinstance(val, float) or isinstance(val, int):
                        writer.add_scalar(f"train/{k}", val, total_timesteps)

            # 3) Evaluate periodically
            if total_timesteps >= next_eval:
                mean_rew, returns = evaluate_policy(args.eval_episodes)
                writer.add_scalar("eval/mean_return", mean_rew, total_timesteps)
                print(f"[Evaluation @ {total_timesteps} steps] Mean Return: {mean_rew:.2f}")
                eval_count += 1
                next_eval += args.eval_freq

    # Final evaluation
    mean_rew, _ = evaluate_policy(args.eval_episodes)
    print(f"[After Training] Mean Return: {mean_rew:.2f}")
    writer.add_scalar("eval/trained_mean_return", mean_rew, total_timesteps)

    writer.close()

if __name__ == "__main__":
    main()


# python -m baselines.sqil --env_name Ant-v5 --num_expert_trajs 16 