import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from baselines.util import load_expert_trajectories_imitation

# Replaced GAIL with AIRL
from imitation.algorithms.adversarial.airl import AIRL

from imitation.data.types import Trajectory
# For AIRL, we typically use BasicShapedRewardNet
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.vec_env import DummyVecEnv

import datetime
import dateutil.tz

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments for environment name, number of expert trajectories, etc.
    """
    parser = argparse.ArgumentParser(description="Train AIRL on a MuJoCo environment from raw .pt expert data.")
    parser.add_argument("--env_name", type=str, default="Ant-v5",
                        help="MuJoCo Gym environment ID, e.g. Ant-v5.")
    parser.add_argument("--num_expert_trajs", type=int, default=5,
                        help="Number of expert episodes to use.")
    parser.add_argument("--train_steps", type=int, default=1_500_000,
                        help="Number of AIRL training timesteps (generator steps).")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes (before/after training).")
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Frequency (in timesteps) of policy evaluation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Force CPU usage (no CUDA)
    device = "cuda:2"

    #  Logging directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_dir = f"logs/{args.env_name}/exp-{args.num_expert_trajs}/airl/" + now.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_dir, exist_ok=True)
    
    #  Instantiate TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    #  Create vectorized environment
    env_fn = lambda: gym.make(args.env_name)
    venv = DummyVecEnv([env_fn])

    batch_size = 4096  # how many timesteps we collect each generator round

    # Load expert trajectories
    trajectories = load_expert_trajectories_imitation(args.env_name, args.num_expert_trajs)

    # Create the PPO learner on CPU
    learner = PPO(
        policy="MlpPolicy",
        env=venv,
        seed=args.seed,
        device=device  # <--- Force CPU device here
    )

    # Create the reward network for AIRL
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
    )

    # Create AIRL trainer
    airl_trainer = AIRL(
        demonstrations=trajectories,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        demo_batch_size=batch_size,
        allow_variable_horizon=True,
    )

    # Evaluate untrained policy
    venv.reset()
    rewards_before, _ = evaluate_policy(
        learner,
        venv,
        n_eval_episodes=args.eval_episodes,
        return_episode_rewards=True,
    )
    mean_before = np.mean(rewards_before)
    print(f"[Before Training] Mean Return: {mean_before:.2f}")
    writer.add_scalar("eval/untrained_mean_return", mean_before, 0)

    # We want to do an evaluation at these times:
    next_eval = args.eval_freq

    # Create CSV file and write column headers
    csv_path = os.path.join(log_dir, "progress.csv")
    with open(csv_path, "w") as f:
        f.write("episode,Real Det Return\n")

    # Train AIRL manually so we can log inside the loop
    total_timesteps = 0

    with tqdm(total=args.train_steps, desc="Training AIRL") as pbar:
        while total_timesteps < args.train_steps:

            # Generator (policy) training
            airl_trainer.train_gen()

            # Discriminator (reward) training
            disc_losses = []
            for _ in range(airl_trainer.n_disc_updates_per_round):
                disc_stats = airl_trainer.train_disc()
                disc_losses.append(disc_stats["disc_loss"])

                # Log each of the returned discriminator metrics
                for k, v in disc_stats.items():
                    writer.add_scalar(f"disc/{k}", v, total_timesteps)

            #  Log the average discriminator loss
            mean_disc_loss = np.mean(disc_losses)
            writer.add_scalar("disc/mean_disc_loss", mean_disc_loss, total_timesteps)

            #  Update counters
            total_timesteps += batch_size
            pbar.update(batch_size)

            # Evaluate *once we pass* the next_eval threshold
            if total_timesteps >= next_eval:
                venv.reset()
                eval_rewards, _ = evaluate_policy(
                    learner,
                    venv,
                    n_eval_episodes=args.eval_episodes,
                    return_episode_rewards=True,
                )
                mean_eval = np.mean(eval_rewards)
                writer.add_scalar("eval/mean_return", mean_eval, total_timesteps)
                print(f"[Evaluation @ {total_timesteps} steps] Mean Return: {mean_eval:.2f}")

                # Save evaluation results to CSV
                with open(csv_path, "a") as f:
                    f.write(f"{next_eval},{mean_eval}\n")

                next_eval += args.eval_freq

    # Final evaluation
    venv.reset()
    rewards_after, _ = evaluate_policy(
        learner,
        venv,
        n_eval_episodes=args.eval_episodes,
        return_episode_rewards=True,
    )
    mean_after = np.mean(rewards_after)
    print(f"[After Training] Mean Return: {mean_after:.2f}")
    writer.add_scalar("eval/trained_mean_return", mean_after, total_timesteps)

    # Close writer
    writer.close()

    # Simple bar plot comparing returns
    plt.figure(figsize=(6, 4))
    plt.bar(["Before", "After"], [mean_before, mean_after],
            color=["red", "blue"], alpha=0.6)
    plt.ylabel("Mean Return")
    plt.title(f"AIRL on {args.env_name}\n({args.num_expert_trajs} expert episodes)")
    plot_path = os.path.join(log_dir, "airl_return_comparison.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved reward comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()
