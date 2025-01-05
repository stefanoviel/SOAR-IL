import gymnasium as gym
import numpy as np
from typing import List
import torch
from imitation.data.types import Trajectory


def load_expert_trajectories(env_name: str, num_trajs: int):
    """
    Loads expert states and actions from .pt files:
      - states: expert_data/states/{env_name}.pt
      - actions: expert_data/actions/{env_name}.pt
    Each shape is [N, T+1, state_dim] or [N, T, action_dim], etc.

    Returns a list of `Trajectory` objects, each a full episode.
    """
    states_path = f"expert_data/states/{env_name}.pt"
    actions_path = f"expert_data/actions/{env_name}.pt"

    # Load raw expert data
    expert_states_all = torch.load(states_path).numpy()  # shape: (N, T+1, state_dim)
    expert_actions_all = torch.load(actions_path).numpy() # shape: (N, T, act_dim)

    # Keep only the first `num_trajs` trajectories
    expert_states = expert_states_all[:num_trajs]
    expert_actions = expert_actions_all[:num_trajs]

    # Build a list of Trajectory objects
    trajectories = []
    for i in range(expert_states.shape[0]):
        states_i = expert_states[i]    # shape (T, state_dim)
        actions_i = expert_actions[i]  # shape (T, act_dim)

        # We want len(obs) = len(acts)+1, but we currently have T == T.
        # So we’ll keep all T states, and drop the last action:
        #   - obs = shape (T, state_dim)
        #   - acts = shape (T-1, act_dim)
        obs = states_i
        acts = actions_i[:-1]          # discard last action

        # Now len(obs)=T, len(acts)=T-1 => T == (T-1) +1 => OK
        infos = [{} for _ in range(len(acts))]  # length T-1

        # Mark trajectory as terminal
        traj = Trajectory(obs=obs, acts=acts, infos=infos, terminal=True)
        trajectories.append(traj)

    return trajectories


def measure_expert_returns(env_name: str, expert_trajectories: List[Trajectory]):
    """
    Steps through the environment using the stored actions from each demonstration trajectory.
    Returns the list of returns (sum of environment rewards) for each trajectory.
    
    WARNING: This only makes sense if your environment is at least somewhat Markovian
    and you don't require an exact environment state match. For MuJoCo, you'd ideally
    set the simulator state to each demonstration state for perfect replay, but here's
    a simplified approach.
    """
    env = gym.make(env_name)

    returns = []
    for traj_idx, traj in enumerate(expert_trajectories):
        obs = env.reset()  # environment's state won't match exactly
        done = False
        total_reward = 0.0

        # Step through each action in the trajectory
        for t in range(len(traj.acts)):
            # If done was encountered early, break to avoid stepping further
            if done:
                break
            action = traj.acts[t]
            o, reward, done, _, _ = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return returns, mean_return, std_return


# if __name__ == "__main__":
#     expert_trajectories = load_expert_trajectories("Ant-v5", num_trajs=10)
#     returns, mean_return, std_return = measure_expert_returns("Ant-v5", expert_trajectories)
#     print(f"Expert returns: {returns}")
#     print(f"Mean return: {mean_return}, std: {std_return}")


def measure_expert_returns_from_raw(
    env_name: str,
    states_path: str,
    actions_path: str,
    num_trajs: int,
):
    """
    Measures the returns (sum of rewards) for each expert trajectory by replaying
    the raw actions in the given environment. This does NOT rely on the `Trajectory`
    class or the `load_expert_trajectories` function.

    :param env_name: Gym environment ID, e.g. "Ant-v5".
    :param states_path: Path to the `.pt` file containing expert states,
        shape (N, T+1, state_dim).
    :param actions_path: Path to the `.pt` file containing expert actions,
        shape (N, T, act_dim).
    :param num_trajs: Number of trajectories to evaluate from these files.
    :return: (returns, mean_return, std_return)
        - returns: list of per-trajectory returns (floats)
        - mean_return: mean of those returns
        - std_return: standard deviation of those returns
    """
    # 1. Create the environment
    env = gym.make(env_name)

    # 2. Load raw expert data from .pt
    #    These are typically torch tensors, so we call .numpy() for easy handling.
    all_states = torch.load(states_path).numpy()   # shape: (N, T+1, state_dim)
    all_actions = torch.load(actions_path).numpy() # shape: (N, T, act_dim)

    # 3. Keep only the first `num_trajs` trajectories (if you have more than needed)
    all_states = all_states[:num_trajs]    # shape: (num_trajs, T+1, state_dim)
    all_actions = all_actions[:num_trajs]  # shape: (num_trajs, T, act_dim)

    returns = []

    # 4. Replay each trajectory's actions in the environment
    for i in range(num_trajs):
        # Reset the environment to a (random) initial state
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # all_actions[i] has shape (T, act_dim)
        # We'll step the environment with each action in sequence
        for t in range(all_actions[i].shape[0]):
            if done:
                break
            action = all_actions[i][t]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        returns.append(total_reward)

    # 5. Calculate mean and std of all returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    return returns, mean_return, std_return


if __name__ == "__main__":
    # Example usage:
    env_name = "Ant-v5"
    states_path = f"expert_data/states/{env_name}.pt"
    actions_path = f"expert_data/actions/{env_name}.pt"
    num_trajs = 10

    returns, mean_ret, std_ret = measure_expert_returns_from_raw(
        env_name,
        states_path,
        actions_path,
        num_trajs,
    )

    print(f"Expert returns (raw data): {returns}")
    print(f"Mean return: {mean_ret:.2f}, Std: {std_ret:.2f}")