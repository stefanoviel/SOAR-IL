# '''
# Code from spinningup repo.
# Refer[Original Code]: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
# '''

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
# import pdb
import gymnasium as gym
import time
import sys
import common.sac_agent as core
import random

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, device=torch.device('cpu'), size=int(1e6)):
        self.state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.next_state = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.action = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        # print(device)
        self.rng = np.random.RandomState()

    def set_seed(self, seed):
        """Improved seeding for replay buffer"""
        self.rng = np.random.RandomState(seed)
        # If using PyTorch for sampling, also set its seed
        torch.manual_seed(seed)

    def store_batch(self, obs, act, rew, next_obs, done):
        num = len(obs)
        full =  self.ptr + num > self.max_size
        if not full:
            self.state[self.ptr: self.ptr + num] = obs
            self.next_state[self.ptr: self.ptr + num] = next_obs
            self.action[self.ptr: self.ptr + num] = act
            self.reward[self.ptr: self.ptr + num] = rew
            self.done[self.ptr: self.ptr + num] = done
            self.ptr = self.ptr + num
        else:
            idx = np.arange(self.ptr,self.ptr+num)%self.max_size
            self.state[idx] = obs
            self.next_state[idx]=next_obs
            self.action[idx]=act
            self.reward[idx]=rew
            self.done[idx]=done
            self.ptr= (self.ptr+num)%self.max_size            

        self.size = min(self.size + num, self.max_size)

    def store(self, obs, act, rew, next_obs, done):
        self.state[self.ptr] = obs
        self.next_state[self.ptr] = next_obs
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        """Update sampling to use seeded RNG"""
        idxs = self.rng.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.state[idxs],
                     obs2=self.next_state[idxs],
                     act=self.action[idxs],
                     rew=self.reward[idxs],
                     done=self.done[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}


class SAC:

    def __init__(self, env_fn, replay_buffer, k=1, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, add_time=False,
            polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, update_num=20,
            update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
            log_step_interval=None, reward_state_indices=None,
            save_freq=1, device=torch.device("cpu"), automatic_alpha_tuning=True, reinitialize=True,
            num_q_pairs=3, use_actions_for_reward=False, writer=None, ema_decay=0.995, **kwargs):

        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\\text{targ}} + (1-\\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

            q_std_clip (float): Maximum value to clip Q-value standard deviations. Default: 1.0
        """

        self.ema_decay = ema_decay  # Decay factor for EMA (close to 1 for more smoothing)
        self.clip_value_ema = None  # Initialize EMA value as None, first iteration doesn't have clipping
        
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.max_ep_len=max_ep_len
        self.start_steps=start_steps
        self.batch_size=batch_size
        self.gamma=gamma    
        self.use_actions_for_reward = use_actions_for_reward
        
        self.polyak=polyak
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.udpate_num = update_num
        self.num_test_episodes = num_test_episodes
        self.epochs = epochs
        # Create actor-critic module and set its seed
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, k, 
                             add_time=add_time, device=device, num_q_pairs=num_q_pairs, **ac_kwargs)
        self.ac.set_seed(seed)
        self.num_q_pairs = num_q_pairs
        
        # Create target networks and set their seeds
        self.ac_targ = deepcopy(self.ac)
        self.ac_targ.set_seed(seed + 1)  # Use different seed for target network

        # Freeze target networks with respect to optimizers
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # pdb.set_trace()
            
        # Create separate optimizers for each Q-network pair
        self.q_optimizers = [Adam(pair_params, lr=lr) 
                           for pair_params in self.ac.q_params_list]

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = replay_buffer
        self.replay_buffer.set_seed(seed)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)

        self.device = device

        self.automatic_alpha_tuning = automatic_alpha_tuning
        if self.automatic_alpha_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        self.true_state_dim = self.env.observation_space.shape[0]

        if log_step_interval is None:
            log_step_interval = steps_per_epoch
        self.log_step_interval = log_step_interval
        self.reinitialize = reinitialize

        self.reward_function = None
        self.reward_state_indices = reward_state_indices

        self.test_fn = self.test_agent

        self.writer = writer  # Store the TensorBoard writer

        # Add comprehensive seeding at initialization
        self.seed = seed
        self._setup_seeds(seed)

    def _setup_seeds(self, seed):
        """Centralized seed setup for reproducibility"""
        
        # Set seeds for PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Set seeds for NumPy
        np.random.seed(seed)
        
        # Set seeds for Python's random
        random.seed(seed)
        
        # Set seeds for environments
        self.env.reset(seed=seed)
        self.test_env.reset(seed=seed+10000)
        
        # Set seeds for replay buffer and networks
        self.replay_buffer.set_seed(seed)
        self.ac.set_seed(seed)
        self.ac_targ.set_seed(seed+1)

        self.action_rng = np.random.RandomState(seed)


    def update_clip_value_ema(self, new_value):
        """Update the exponential moving average of the clipping value"""
        if self.clip_value_ema is None:
            self.clip_value_ema = new_value
        else:
            self.clip_value_ema = self.ema_decay * self.clip_value_ema + (1 - self.ema_decay) * new_value


    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data, q_idx):
        """
        Compute Q-loss for a specific pair of Q-networks, and also
        store some logging information about target Q-values.
        """
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1_list[q_idx](o, a)
        q2 = self.ac.q2_list[q_idx](o, a)

        # Bellman backup
        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(o2[:, :self.true_state_dim])
            q1_pi_targ = self.ac_targ.q1_list[q_idx](o2, a2)
            q2_pi_targ = self.ac_targ.q2_list[q_idx](o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

            # For logging: let's store the mean and std of the target values
            self._current_target_q_mean = backup.mean().item()
            self._current_target_q_std = backup.std().item()

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        return loss_q1 + loss_q2


    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, clip_value):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o[:, :self.true_state_dim])
        
        # Get Q-values from all networks
        q1_vals = [q1(o, pi) for q1 in self.ac.q1_list]
        q2_vals = [q2(o, pi) for q2 in self.ac.q2_list]
        
        # Compute mean and std of minimum Q-values  
        q_mins = [torch.min(q1, q2) for q1, q2 in zip(q1_vals, q2_vals)]
        q_mean = torch.mean(torch.stack(q_mins, dim=0), dim=0)
        
        if self.num_q_pairs > 1:
            exploration_bonus = torch.clamp(torch.std(torch.stack(q_mins, dim=0), dim=0), 0, clip_value)
        else:
            exploration_bonus = 0

        # Use mean + exploration bonus in policy loss
        # By minimizing the log probability of all the actions, we are maximizing the entropy of the policy.
        loss_pi = (self.alpha * logp_pi - (q_mean + exploration_bonus)).mean()
        return loss_pi, logp_pi


    def update(self, data, clip_value, global_step_logging):
        # Update each Q-network pair with different batches
        losses_q = []
        for i in range(len(self.ac.q1_list)):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            self.q_optimizers[i].zero_grad()
            loss_q = self.compute_loss_q(batch, i)
            loss_q.backward()
            self.q_optimizers[i].step()
            losses_q.append(loss_q.item())

        # Freeze Q-networks
        for params in self.ac.q_params_list:
            for p in params:
                p.requires_grad = False

        # Update policy
        self.pi_optimizer.zero_grad()
        loss_pi, log_pi = self.compute_loss_pi(data, clip_value)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Automatic alpha tuning
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # Unfreeze Q-networks
        for params in self.ac.q_params_list:
            for p in params:
                p.requires_grad = True

        # Update target networks (Polyak averaging)
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # ----------------------------------------------------------------------
        #  Gather metrics instead of logging here
        # ----------------------------------------------------------------------
        mean_q_loss = np.mean(losses_q)
        target_q_mean = getattr(self, '_current_target_q_mean', 0.)
        target_q_std = getattr(self, '_current_target_q_std', 0.)
        policy_entropy = -log_pi.mean().item()
        ensemble_q_mean, ensemble_q_std = self.get_q_stats()

        # Return them so we can log outside
        return (
            np.array([mean_q_loss, loss_pi.item(), log_pi.detach().cpu().mean().item()]),
            {
                'Loss_Q': mean_q_loss,
                'QTarget_mean': target_q_mean,
                'QTarget_std': target_q_std,
                'Policy_entropy': policy_entropy,
                'QEnsemble_mean': ensemble_q_mean,
                'QEnsemble_std': ensemble_q_std
            }
        )
    


    def get_action(self, o, deterministic=False, get_logprob=False):
        if len(o.shape) < 2:
            o = o[None, :]
        return self.ac.act(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), 
                    deterministic, get_logprob=get_logprob)

    def get_action_batch(self, o, deterministic=False):
        if len(o.shape) < 2:
            o = o[None, :]
        return self.ac.act_batch(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), 
                    deterministic)


    def reset(self):
        """Reset random number generators"""
        if hasattr(self, 'seed'):
            self.ac.set_seed(self.seed)
            self.ac_targ.set_seed(self.seed + 1)
            self.replay_buffer.set_seed(self.seed)

    def test_agent(self):
        avg_ep_return = 0.
        test_seed_offset = 10000  # Use different seed range for testing
        
        for j in range(self.num_test_episodes):
            test_seed = self.seed + test_seed_offset + j
            o, info = self.test_env.reset(seed=test_seed)
            obs = np.zeros((self.max_ep_len, o.shape[0]))
            acts = np.zeros((self.max_ep_len, self.act_dim))
            for t in range(self.max_ep_len):
                # Take deterministic actions at test time?
                o, a, _, _, _ = self.test_env.step(self.get_action(o, True))
                obs[t] = o.copy()
                acts[t] = a.copy()
            obs = torch.FloatTensor(obs).to(self.device)[:, self.reward_state_indices]
            acts = torch.FloatTensor(acts).to(self.device)
            # Concatenate states and actions before passing to reward function
            if self.use_actions_for_reward:
                combined_input = torch.cat([obs, acts], dim=1)
                avg_ep_return += self.reward_function(combined_input).sum()
            else:
                avg_ep_return += self.reward_function(obs).sum()
            
        return avg_ep_return/self.num_test_episodes

    def test_agent_ori_env(self, deterministic=True):
        # for expert evaluation
        if hasattr(self.test_env, 'eval'):
            self.test_env.eval()

        rets = []
        for j in range(self.num_test_episodes):
            ret = 0
            o, info = self.test_env.reset(seed=self.seed+j)
            for t in range(self.max_ep_len):
                a = self.get_action(o, deterministic)
                o, r, done, _, _ = self.test_env.step(a)
                ret += r
                if done:
                    break
            rets.append(ret)      
        return np.mean(rets)
    
    def test_agent_batch(self):
        if hasattr(self.test_env, 'eval'):
            self.test_env.eval()

        # Use different seeds for each test episode
        test_seeds = np.arange(self.seed, self.seed + self.num_test_episodes)
        o, info = self.test_env.reset(test_seeds[0])  # Adjust based on your env implementation
        ep_ret = np.zeros((self.num_test_episodes))
        log_pi = np.zeros(self.num_test_episodes)
        for t in range(self.max_ep_len-1):
            # Take stochastic action!
            a, log_pi_ = self.get_action_batch(o)
            o, r, _, _, _ = self.test_env.step(a)
            # print(t, o, r, a)
            ep_ret += r
            log_pi += log_pi_

        return ep_ret.mean(), log_pi.mean()

   

    def sample_action(self):
        """Sample a random action using seeded RNG"""
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            return self.action_rng.uniform(
                low=self.env.action_space.low,
                high=self.env.action_space.high,
                size=self.env.action_space.shape
            )
        else:
            raise NotImplementedError("Only continuous action spaces supported")


    # Learns from single trajectories rather than batch
    def learn_mujoco(self, global_step_logging, print_out=False, save_path=None):
        # Reset all seeds at the start of training
        self._setup_seeds(self.seed)
        
        # Use separate seed sequences for different aspects
        train_seed_sequence = np.random.RandomState(self.seed)
        eval_seed_sequence = np.random.RandomState(self.seed + 5000)
        
        # Initialize environment with base seed
        o, info = self.env.reset(seed=self.seed)
        
        current_seed = self.seed
        ep_len = 0

        print(f"Training SAC for IRL agent: Total steps {self.steps_per_epoch * self.epochs:d}")
        # Main loop: collect experience in env and update/log each epoch
        test_rets = []
        alphas = []
        log_pis = []
        test_time_steps = []
        local_time = time.time()
        start_time = time.time()

        best_eval = -np.inf

        # Add tracking variables for discounted rewards
        current_discounted_reward = 0
        gamma_power = 1
        trajectory_states = []  # Store states for computing rewards later
        trajectory_actions = []  # Store actions if needed
        trajectory_returns = []
        
        for t in range(self.steps_per_epoch * self.epochs):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if self.replay_buffer.size > self.start_steps:
                a = self.get_action(o)

            else:
                a = self.env.action_space.sample()


            # Step the env
            o2, r, d, _, _ = self.env.step(a)
            
            # Store state and action for reward computation
            trajectory_states.append(o.copy())
            if self.use_actions_for_reward:
                trajectory_actions.append(a.copy())
            
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # important, assume all trajecotires are synchronized.
            # HACK:
            # For expert, done = True is episode terminates early
            # done = False if episode terminates at end of time horizon
            d = False if ep_len==self.max_ep_len else d

            # print(r,d)
            # Store experience to replay buffer
            # self.replay_buffer.store_batch(o, a, r, o2, d)
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or ep_len == self.max_ep_len:
                # Convert lists to numpy arrays first
                traj_states_np = np.array(trajectory_states)
                traj_states = torch.FloatTensor(traj_states_np).to(self.device)
                if self.reward_state_indices is not None:
                    traj_states = traj_states[:, self.reward_state_indices]
                
                if self.use_actions_for_reward:
                    traj_actions_np = np.array(trajectory_actions)
                    traj_actions = torch.FloatTensor(traj_actions_np).to(self.device)
                    combined_input = torch.cat([traj_states, traj_actions], dim=1)
                    rewards = self.reward_function(combined_input)
                    learned_rewards = rewards.detach().cpu().numpy() if torch.is_tensor(rewards) else rewards
                else:
                    rewards = self.reward_function(traj_states)
                    learned_rewards = rewards.detach().cpu().numpy() if torch.is_tensor(rewards) else rewards
                
                # Compute discounted sum of rewards
                discounted_sum = 0
                for r in reversed(learned_rewards):
                    discounted_sum = r + self.gamma * discounted_sum
                
                # Store the discounted return for this trajectory
                trajectory_returns.append(discounted_sum)
                
                # Reset trajectory tracking variables
                trajectory_states = []
                trajectory_actions = []
                
                # Reset environment
                current_seed = train_seed_sequence.randint(0, 2**32-1)
                o, info = self.env.reset(seed=current_seed)
                ep_len = 0

            # Update handling
            log_pi = 0
            if self.reinitialize:
                if t >= self.update_after and t % self.update_every == 0:
                    # We do multiple updates here
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        results, logs_dict = self.update(
                            data=batch, 
                            clip_value=self.clip_value_ema, 
                            global_step_logging=global_step_logging
                        )
                        _, _, log_pi = results
                        
                    test_epret = self.test_fn()
                    if print_out:
                        print(f"SAC Training | Evaluation: {test_epret:.3f} Timestep: {t+1:d} Elapsed {time.time() - local_time:.0f}s")        
                    print(f"Update: {t+1:d} Loss: {log_pi:.3f}")

            else:
                if self.replay_buffer.size >= self.update_after and t % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        obs = batch['obs'][:, self.reward_state_indices]
                        
                        if self.use_actions_for_reward:
                            acts = batch['act']
                            combined_input = torch.cat([obs, acts], dim=1)
                            batch['rew'] = torch.FloatTensor(self.reward_function(combined_input)).to(self.device)
                        else:
                            batch['rew'] = torch.FloatTensor(self.reward_function(obs)).to(self.device)
                        
                        results, logs_dict = self.update(
                            data=batch, 
                            clip_value=self.clip_value_ema, 
                            global_step_logging=global_step_logging
                        )
                        _, _, log_pi = results


            # End of epoch handling
            if t % self.log_step_interval == 0:
                # Test the performance of the deterministic version of the agent.
                test_epret = self.test_fn()
                if print_out:
                    print(f"SAC Training | Evaluation: {test_epret:.3f} Timestep: {t+1:d} Elapsed {time.time() - local_time:.0f}s")
                if save_path is not None:
                    if test_epret>best_eval:
                        best_eval=test_epret
                        torch.save(self.ac.state_dict(),save_path)
                alphas.append(self.alpha.item() if self.automatic_alpha_tuning else self.alpha)
                test_rets.append(test_epret)
                log_pis.append(log_pi)
                test_time_steps.append(t+1)
                local_time = time.time()


        # After we finish all steps, we log the final clip value:
        print("len(trajectory_returns)", len(trajectory_returns), "self.writer", self.writer)
        if len(trajectory_returns) > 0 and self.writer is not None:
            print("global_step_logging", global_step_logging)

            new_clip_value = np.abs(np.median(trajectory_returns))
            self.update_clip_value_ema(new_clip_value)
            self.writer.add_scalar('Training/reward_clip_value', self.clip_value_ema, global_step_logging)
            self.writer.add_scalar('Training/trajectory_return', new_clip_value, global_step_logging)

            self.writer.add_scalar('SAC/Loss_Q', logs_dict['Loss_Q'], global_step_logging)
            self.writer.add_scalar('SAC/QTarget_mean', logs_dict['QTarget_mean'], global_step_logging)
            self.writer.add_scalar('SAC/QTarget_std', logs_dict['QTarget_std'], global_step_logging)
            self.writer.add_scalar('SAC/Policy_entropy', logs_dict['Policy_entropy'], global_step_logging)
            self.writer.add_scalar('SAC/QEnsemble_mean', logs_dict['QEnsemble_mean'], global_step_logging)
            self.writer.add_scalar('SAC/QEnsemble_std', logs_dict['QEnsemble_std'], global_step_logging)
        print(f"SAC Training End: time {time.time() - start_time:.0f}s")
        return [test_rets, alphas, log_pis, test_time_steps]


    @property
    def networks(self):
        return [self.ac.pi, self.ac.q1, self.ac.q2]

    def get_q_stats(self):
        """Get average Q-values and their standard deviations across Q-networks for each state."""
        if not hasattr(self, '_q_stats_batch'):
            # Cache a batch of states for consistent monitoring
            self._q_stats_batch = self.replay_buffer.sample_batch(100)
        
        batch = self._q_stats_batch
        o, a = batch['obs'], batch['act']
        
        with torch.no_grad():
            # Get Q-values from all networks for current state-action pairs
            q1_vals = torch.stack([q1(o, a) for q1 in self.ac.q1_list])
            q2_vals = torch.stack([q2(o, a) for q2 in self.ac.q2_list])
            
            # Get minimum Q-values from each pair
            q_mins = torch.minimum(q1_vals, q2_vals)
            
            # Compute mean and std for each state across Q-networks
            q_mean = q_mins.mean(dim=0).mean().item()  # Average across networks then states
            q_std = q_mins.std(dim=0).mean().item()    # Std across networks, averaged over states
            
            return q_mean, q_std