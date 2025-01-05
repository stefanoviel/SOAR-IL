import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader


class CoherentReward:
    def __init__(self, state_dim, action_dim, alpha=1.0, device="cpu", action_low=-1.0, action_high=1.0, hidden_sizes=[256, 256], learning_rate=3e-4, weight_decay=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Policy network for q(a|s) - build MLP with specified hidden sizes
        layers = []
        prev_size = state_dim
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.ReLU()
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, 2 * action_dim))  # Output mean and log_std
        
        self.policy = nn.Sequential(*layers).to(device)
        
        # Define uniform prior bounds
        self.action_low = action_low
        self.action_high = action_high

    def train_policy(self, expert_states, expert_actions, num_epochs=100, batch_size=256):
        """Train policy using behavioral cloning with L2 regularization"""
        optimizer = torch.optim.Adam(self.policy.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay)
        
        dataset = TensorDataset(
            torch.FloatTensor(expert_states).to(self.device),
            torch.FloatTensor(expert_actions).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for states, actions in dataloader:
                means, log_stds = self.policy(states).chunk(2, dim=-1)
                log_stds = torch.clamp(log_stds, -20, 2)
                
                # Gaussian policy distribution
                dist = Normal(means, log_stds.exp())
                log_prob = dist.log_prob(actions).sum(-1)
                
                loss = -log_prob.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    def get_reward(self, states, actions):
        """Compute the coherent reward r(s,a) = α(log q(a|s) - log p(a|s))"""
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            
            # Compute log q(a|s)
            means, log_stds = self.policy(states).chunk(2, dim=-1)
            log_stds = torch.clamp(log_stds, -20, 2)
            dist = Normal(means, log_stds.exp())
            log_q = dist.log_prob(actions).sum(-1)
            
            # Compute log p(a|s) for uniform prior
            # For uniform distribution over [low, high], log_prob = -log(high - low)
            log_p = -np.log(self.action_high - self.action_low) * self.action_dim
            
            # Compute reward
            reward = self.alpha * (log_q - log_p)
            
            return reward.cpu().numpy()
        

    def state_dict(self):
        return self.policy.state_dict()