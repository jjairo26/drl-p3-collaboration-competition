from collections import deque
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=64, random_seed=None):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.random_seed = random_seed
        random.seed(random_seed)

    def add(self, states, actions, rewards, next_states, dones):

        concat_states = np.concatenate(states)
        concat_actions = np.concatenate(actions)
        rewards_array = np.array(rewards)
        concat_next_states = np.concatenate(next_states)
        dones_array = np.array(dones).astype(np.uint8)  # Convert True/False to 1/0

        #assert concat_states.shape[0] % 2 == 0
        #assert concat_actions.shape[0] % 2 == 0
        #assert rewards_array.shape == (2,)
        #assert dones_array.shape == (2,)

        self.buffer.append((concat_states, concat_actions, rewards_array, concat_next_states, dones_array))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)

        states = torch.from_numpy(np.vstack([b[0] for b in batch])).float().to(device)
        actions = torch.from_numpy(np.vstack([b[1] for b in batch])).float().to(device)
        rewards = torch.from_numpy(np.vstack([b[2] for b in batch])).float().to(device)
        next_states = torch.from_numpy(np.vstack([b[3] for b in batch])).float().to(device)
        dones = torch.from_numpy(np.vstack([b[4] for b in batch])).float().to(device) # True/False to 1/0, then to float
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)