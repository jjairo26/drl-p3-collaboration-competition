from ddpg_agent import DDPGAgent
from ReplayBuffer import ReplayBuffer
import random
import torch

class MADDPG():
    def __init__(self, num_agents, maddpg_params, ddpg_agent_params):
        self.num_agents = num_agents
        self.agents = [DDPGAgent(ddpg_agent_params) for _ in range(num_agents)]

        self.learn_every = maddpg_params['learn_every']
        self.num_training = maddpg_params['num_training']
        self.batch_size = maddpg_params['batch_size']
        self.buffer_size = int(maddpg_params['buffer_size'])

        self.seed = maddpg_params['random_seed']

        self.memory = ReplayBuffer(buffer_size=self.buffer_size,
                                    batch_size=self.batch_size,
                                    random_seed=self.seed)

        self.t_step = 0      # Initialize time step (for updating every UPDATE_EVERY steps)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        actions = []
        for i in range(0, self.num_agents):
            actions.append(self.agents[i].act(states[i], add_noise=add_noise))
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn every self.learn_every steps
        self.t_step = (self.t_step + 1) % self.learn_every

        if self.t_step == 0:
            # Update the model num_training times
            for i in range(0, self.num_training):
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    for i in range(0, self.num_agents):
                        self.agents[i].learn(experiences)

    def save_models(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_{i}.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_{i}.pth')

