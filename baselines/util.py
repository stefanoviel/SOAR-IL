
import torch
from imitation.data.types import Trajectory

def load_expert_trajectories_imitation(env_name: str, num_trajs: int):
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
        obs = states_i
        acts = actions_i[:-1]          # discard last action

        infos = [{} for _ in range(len(acts))]  # length T-1
        traj = Trajectory(obs=obs, acts=acts, infos=infos, terminal=False)
        trajectories.append(traj)

    return trajectories