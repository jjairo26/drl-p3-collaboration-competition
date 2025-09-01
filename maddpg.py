from ddpg_agent import DDPGAgent
from ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
class MADDPG():
    def __init__(self, num_agents, maddpg_params, ddpg_agent_params):
        self.num_agents = num_agents
        self.agents = [DDPGAgent({**ddpg_agent_params, "random_seed": ddpg_agent_params["random_seed"] + i},
                         num_agents)
               for i in range(num_agents)]
        self.learn_every = maddpg_params['learn_every']
        self.num_training = maddpg_params['num_training']
        self.batch_size = maddpg_params['batch_size']
        self.buffer_size = int(maddpg_params['buffer_size'])
        self.tau = ddpg_agent_params['tau']

        self.seed = maddpg_params['random_seed']

        self.memory = ReplayBuffer(buffer_size=self.buffer_size,
                                    batch_size=self.batch_size,
                                    random_seed=self.seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0                 
        # Variance for OU noise process                
        self.current_sigma = ddpg_agent_params['sigma'] 

    def reset(self):
        '''Reset noise for all agents'''
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        '''Make all agents act according to their current policy'''
        actions = []
        for i in range(0, self.num_agents):
            actions.append(self.agents[i].act(states[i], add_noise=add_noise))
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        '''Add experience to common buffer and make agents learn if possible'''
        
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn every self.learn_every steps
        self.t_step = (self.t_step + 1) % self.learn_every

        if self.t_step == 0:
            # Update the model num_training times
            for i in range(0, self.num_training):
                # If there are enough experience tuples
                if len(self.memory) > self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)
                    
    def learn(self, experiences):
        '''Learning procedure for the agents'''

        states, actions_buf, rewards, next_states, dones = experiences

        state_size = self.agents[0].state_size
        action_size = self.agents[0].action_size

        # Determine target
        with torch.no_grad():
            next_actions_list = []
            for i in range(0, self.num_agents):
                # Slice state for agent i
                agent_next_states = next_states[:, i*state_size:(i+1)*state_size]
                next_action = self.agents[i].actor_target(agent_next_states)
                next_actions_list.append(next_action)

            next_actions_target = torch.cat(next_actions_list, dim=1)

        # Accumulate critic losses for all agents
        critic_loss_total = 0.0
        for i in range(self.num_agents):
            reward_i = rewards[:, i:i+1]
            done_i = dones[:, i:i+1]

            with torch.no_grad():
                Q_target_next = self.agents[i].critic_target(next_states, next_actions_target)
                # Compute Q targets for current states (y_i)
                Q_target = reward_i + (self.agents[i].gamma * Q_target_next * (1 - done_i))
            
            # Here we need the actions from the experience buffer and all states
            Q_expected = self.agents[i].critic_local(states, actions_buf)
            critic_loss = F.mse_loss(Q_expected, Q_target)
            critic_loss_total += critic_loss

        # Backpropagate once for all agents
        for i in range(self.num_agents):
            self.agents[i].critic_optimizer.zero_grad()
        critic_loss_total.backward()
        for i in range(self.num_agents):
            # Gradient clipping for more stability
            torch.nn.utils.clip_grad_norm_(self.agents[i].critic_local.parameters(), 1.0)
            self.agents[i].critic_optimizer.step()

        # --- Update the actor using the sampled policy gradient ---
        for i in range(self.num_agents):
            actions_pred_list = []
            for j in range(self.num_agents):
                states_j = states[:, j*state_size:(j+1)*state_size]
                action_j = self.agents[j].actor_local(states_j)
                if j != i:
                    # We detach in order to avoid computing gradients through the actions of 
                    # other players j which are not currently learning
                    action_j = action_j.detach()
                actions_pred_list.append(action_j)
                
            actions_pred = torch.cat(actions_pred_list, dim=1)

            # We slice again to get the predicted actions for the current agent i
            action_i = actions_pred[:, i*action_size:(i+1)*action_size]
            # The critic takes into account all actions (with j != i detached)
            actor_q = -self.agents[i].critic_local(states, actions_pred).mean()
            # Small penalty for big actions
            act_l2_penalty = (action_i**2).mean()
            # Overall loss to optimize
            actor_loss = actor_q + 1e-3*act_l2_penalty

            # Learning step
            self.agents[i].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.agents[i].actor_optimizer.step() 

            # Soft updates of target networks
            self.soft_update(self.agents[i].critic_local, self.agents[i].critic_target, self.tau)
            self.soft_update(self.agents[i].actor_local, self.agents[i].actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau=1e-3):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def set_sigma_for_all_agents(self):
        '''Reduce sigma with increasing episodes'''
        self.current_sigma = np.max([self.current_sigma * 0.9995, 0.05])
        for agent in self.agents:
            agent.noise.set_sigma(self.current_sigma)

    def save_models(self):
        '''Saves network parameters'''
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'checkpoint_actor_{i}.pth')
            torch.save(agent.critic_local.state_dict(), f'checkpoint_critic_{i}.pth')

