import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from models import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent():
    
    def __init__(self, params):
        
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']
        self.weight_decay = params['weight_decay']

        self.seed = params['random_seed']

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, self.hidden_size_1, self.hidden_size_2, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.hidden_size_1, self.hidden_size_2, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        self.actor_target.load_state_dict(self.actor_local.state_dict())

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, self.hidden_size_1, self.hidden_size_2, self.seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.hidden_size_1, self.hidden_size_2, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
 
        self.actor_local.eval()
        with torch.no_grad(): 
            action = self.actor_local(state).cpu().data.numpy().squeeze(0)
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def learn(self, experiences):
        pass


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size) # long-term mean
        self.theta = theta # rate of mean reversion (how fast the process returns to the mean)
        self.sigma = sigma # volatility (randomness)
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        # This implicitly assumes dt = 1
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state