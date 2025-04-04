'''
Code from spinningup repo.
Refer[Original Code]: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.rng = torch.Generator()
        self.np_rng = np.random.RandomState()
        
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def set_seed(self, seed):
        """Set seeds for all random number generators"""
        self.rng.manual_seed(seed)
        self.np_rng.seed(seed)
        torch.manual_seed(seed)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            # reparameterization trick
            # 1. Sample base noise from standard normal

            epsilon = torch.randn(mu.shape, device=mu.device)  # ε ~ N(0,1)

            # 2. Transform the noise using μ and σ
            pi_action = mu + std * epsilon  # x = μ + σ * ε

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1) # compute log probability of action
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)  # idk 
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)  # project action to [-1,1], but why? 
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def log_prob(self, obs, act):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        # 20210306: fix this bug for reversal operation on action. 
        # this may improve AIRL results, leaving for future work.
        act = act / self.act_limit
        act = torch.atanh(act) # arctanh to project [-1,1] to real

        logp_pi = pi_distribution.log_prob(act).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - act - F.softplus(-2*act))).sum(axis=1)

        return logp_pi

class SquashedGmmMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, k):
        super().__init__()
        self.rng = torch.Generator()
        self.np_rng = np.random.RandomState()
        
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], k*act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], k*act_dim)
        self.act_limit = act_limit
        self.k = k 

    def set_seed(self, seed):
        """Set seeds for all random number generators"""
        self.rng.manual_seed(seed)
        self.np_rng.seed(seed)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # n = batch size
        n, _ = mu.shape
        mixture_components = torch.from_numpy(self.np_rng.randint(0, self.k, (n)))

        # change shape to k x batch_size x act_dim
        mu = mu.view(n, self.k, -1).permute(1, 0, 2)
        std = std.view(n, self.k, -1).permute(1, 0, 2)

        mu_sampled = mu[mixture_components, torch.arange(0,n).long(), :]
        std_sampled = std[mixture_components, torch.arange(0,n).long(), :]

        if deterministic:
            pi_action = mu_sampled
        else:
            pi_action = Normal(mu_sampled, std_sampled).rsample() # (n, act_dim)

        if with_logprob:
            # logp_pi[i,j] contains probability of ith action under jth mixture component
            logp_pi = torch.zeros((n, self.k)).to(pi_action)

            for j in range(self.k):
                pi_distribution = Normal(mu[j,:,:], std[j,:,:]) # (n, act_dim)

                logp_pi_mixture = pi_distribution.log_prob(pi_action).sum(axis=-1)
                logp_pi_mixture -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
                logp_pi[:,j] = logp_pi_mixture

            # logp_pi = (sum of p_pi over mixture components)/k
            logp_pi = torch.logsumexp(logp_pi, dim=1) - torch.FloatTensor([np.log(self.k)]).to(logp_pi) # numerical stable
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        # print(obs, act)
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, k, hidden_sizes=(256,256), add_time=False,
                 activation=nn.ReLU, device=torch.device("cpu"), num_q_pairs=1):
        super().__init__()
        self.rng = torch.Generator()
        self.np_rng = np.random.RandomState()
        
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.device = device

        # Policy network
        if k == 1:
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(self.device)
        else:
            # not used
            self.pi = SquashedGmmMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit, k).to(self.device)

        # Create multiple pairs of Q-networks
        self.q1_list = nn.ModuleList([
            MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
            for _ in range(num_q_pairs)
        ])
        self.q2_list = nn.ModuleList([
            MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(self.device)
            for _ in range(num_q_pairs)
        ])

        # Keep first pair accessible as q1/q2 for backward compatibility
        self.q1 = self.q1_list[0]
        self.q2 = self.q2_list[0]

        # Create list of parameters for each Q-network pair
        self.q_params_list = [
            list(self.q1_list[i].parameters()) + list(self.q2_list[i].parameters())
            for i in range(num_q_pairs)
        ]

    def set_seed(self, seed):
        """Set seeds for all random number generators"""
        self.rng.manual_seed(seed)
        self.np_rng.seed(seed)
        if hasattr(self.pi, 'set_seed'):
            self.pi.set_seed(seed)

    def act(self, obs, deterministic=False, get_logprob=False):
        with torch.no_grad():
            a, logpi = self.pi(obs, deterministic, True)
            if get_logprob:
                return a.cpu().data.numpy().flatten(), logpi.cpu().data.numpy()
            else:
                return a.cpu().data.numpy().flatten()

    def act_batch(self, obs, deterministic=False):
        with torch.no_grad():
            a, logpi = self.pi(obs, deterministic, True)
            return a.cpu().data.numpy(), logpi.cpu().data.numpy()

    def log_prob(self, obs, act):
        return self.pi.log_prob(obs, act)
