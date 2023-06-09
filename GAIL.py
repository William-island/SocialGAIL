import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ExpertTraj
import numpy as np
from PPO import PPO
import random
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x
    
    def score(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = -F.logsigmoid(-self.l3(x))
        return x
    
class GAIL(PPO):
    def __init__(self, args, state_dim, action_dim):
        super().__init__(state_dim=state_dim, action_dim=action_dim, lr_actor=0.0003, \
                       lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2, has_continuous_action_space=True, action_std_init=0.6, device=device)
                       # lr_critic=0.001
        
        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
        self.expert = ExpertTraj(args)
        
        self.loss_fn = nn.BCELoss()
            
    # 在线更新
    def update_online(self, n_iter, batch_size):
        #######################
        # update discriminator
        #######################
        for i in range(n_iter):
            ## sample expert transitions
            exp_state, exp_action = self.expert.sample(batch_size)
            exp_state = torch.FloatTensor(exp_state).to(device)
            exp_action = torch.FloatTensor(exp_action).to(device)
            
            ## sample expert states for actor
            state, action = self.memory_sample(self.buffer.states, self.buffer.actions, batch_size)
            state = torch.squeeze(torch.stack(state, dim=0)).detach().to(self.device)
            action = torch.squeeze(torch.stack(action, dim=0)).detach().to(self.device)
            
            self.optim_discriminator.zero_grad()
            
            # label tensors
            exp_label= torch.full((batch_size,1), 1, dtype=torch.float, device=device)
            policy_label = torch.full((batch_size,1), 0, dtype=torch.float, device=device)
            
            # with expert transitions
            prob_exp = self.discriminator(exp_state, exp_action)
            loss = self.loss_fn(prob_exp, exp_label)
            
            # with policy transitions
            prob_policy = self.discriminator(state, action)
            loss += self.loss_fn(prob_policy, policy_label)
            
            # take gradient step
            loss.backward()
            self.optim_discriminator.step()
            
        ################
        # update policy
        ## ##############
        reward_state = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        reward_action = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        # rewards = torch.log(self.discriminator(reward_state, reward_action))
        rewards = self.discriminator.score(reward_state, reward_action)
        self.buffer.rewards = rewards.detach().cpu().numpy().flatten().tolist()

        self.update_ppo()


    def memory_sample(self, states, actions, batch_size):
        indexes = np.random.randint(0, len(states), size=batch_size)
        state, action = [], []
        for i in indexes:
            s = states[i]
            a = actions[i]
            state.append(s)
            action.append(a)
        return state, action
    
    def memory_sample_all(self, memory, batch_size):
        state, action = [], []
        for i in range(len(memory)):
            s = memory[i][0]
            a = memory[i][1]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
        return np.array(state), np.array(action)
            
    def save(self, directory='./preTrained', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory,name))
        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory,name))
        
    def load(self, directory='./preTrained', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory,name)))
        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory,name)))
