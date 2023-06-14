#!/usr/bin/env python
# coding: utf-8

# In[12]:


import random
import os
from time import time
from collections import deque, namedtuple
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob

from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp


# In[2]:


env = gym.make('LunarLander-v2')

seed = 0
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# In[3]:


device = torch.device('cpu')


# In[4]:


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)  
        
    def forward(self, x):
        #Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x        


# In[5]:


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        # Replay memory allow agent to record experiences and learn from them

        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        # Add experience
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
                
    def sample(self):
        # Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors 

        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([experience.state for experience in experiences if experience is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)        
        rewards = torch.from_numpy(np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)        
        next_states = torch.from_numpy(np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(device)  

        dones = torch.from_numpy(np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(device)        
        
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        return len(self.memory)


# In[6]:


class DQNAgent:
    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Initialize Q and Fixed Q networks
        self.q_network = QNetwork(state_size, action_size, seed).to(device)
        self.fixed_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory 
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0
        
    
    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)
        
    def learn(self, experiences):
        # Learn from experience by training the q_network 
        states, actions, rewards, next_states, dones = experiences
        
        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)
        
        # If done just use reward, else update Q_target with discounted action values
        Q_target = rewards + (GAMMA * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        
        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()
        
        # Log the loss
        writer.add_scalar('Loss', loss.item(), self.timestep)
        
        # Update fixed weights
        self.update_fixed_network(self.q_network, self.fixed_network)
        
    def update_fixed_network(self, q_network, fixed_network):
        # Update fixed network by copying weights from Q network using TAU param

        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data)
        
        
    def act(self, state, eps=0.0):
        # Choose the action

        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # set the network into evaluation mode 
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action    
        
    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)


# In[7]:


# Default Hyperparameters
BUFFER_SIZE = int(1e5) # Replay memory size
BATCH_SIZE = 64         # Number of experiences to sample from memory
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Soft update parameter for updating fixed q network
LR = 1e-4               # Q Network learning rate
UPDATE_EVERY = 4        # How often to update Q network

MAX_EPISODES = 2000  # Max number of episodes to play
MAX_STEPS = 1000     # Max steps allowed in a single episode/play
ENV_SOLVED = 100     # MAX score at which we consider environment to be solved
PRINT_EVERY = 10    # How often to print the progress
SAVE_EVERY = 500

# Epsilon schedule
EPS_START = 1.0      # Default/starting value of eps
EPS_DECAY = 0.999    # Epsilon decay rate
EPS_MIN = 0.01       # Minimum epsilon 


# In[8]:


def train_agent(agent, run_dir):
    # Train a DQN
    start = time()
    scores = []                        # list containing scores from each episode
    eps = EPS_START                    # initialize epsilon
    for i_episode in range(1, MAX_EPISODES+1):
        state = env.reset()
        score = 0
        for t in range(MAX_STEPS):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 

        scores.append(score)              # save most recent score
        mean_score = np.mean(scores[-100:]) #last 100 episodes only
        
        eps = max(eps * EPS_DECAY, EPS_MIN)    # decrease epsilon
        
        # Logging
        writer.add_scalar('Reward', reward, i_episode)
        writer.add_scalar('Training score', score, i_episode)
        writer.add_scalar('Training score mean', mean_score, i_episode)
        
        if mean_score >= ENV_SOLVED:
            agent.checkpoint(f'{run_dir}/save_{i_episode}_solved.pth')
            print(f'\n\tSolved! Ep - {i_episode}')
            break
        
        if i_episode % PRINT_EVERY == 0:
            print(f'\r\tEpisode {i_episode}/{MAX_EPISODES} \tScore {score:.1f} \tAvg Score {mean_score:.1f}/{ENV_SOLVED}', end=' ')
                            
        if i_episode % SAVE_EVERY == 0:
            agent.checkpoint(f'{run_dir}/save_{i_episode}.pth')
            
        elif i_episode == MAX_EPISODES:
            agent.checkpoint(f'{run_dir}/save_{i_episode}_max.pth')
            
    end = time()
    print(f'\n\tIn {(end - start:.1f)/60} min')
    
    return scores


# In[9]:


# Hyperparameter dictionary
HPARAMS = {
    'GAMMA': [0.95, 0.99],
    'TAU': [1e-3, 1e-2],
    'LR': [1e-4, 1e-3, 1e-2],
}

METRICS = {
    'average_score': hp.Metric('average_score', display_name='Mean Score'),
}

permuts = 1
for values in HPARAMS.values():
    permuts *= len(values)
    
def run_training(hparams, run_dir):
    GAMMA, TAU, LR = hparams['GAMMA'], hparams['TAU'], hparams['LR']

    # Initialize a new DQNAgent with given hyperparameters
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, seed=0)
    scores = train_agent(agent, run_dir)

    return np.mean(scores[-100:])

# Iterate over each hyperparameter configuration
j = 1
for GAMMA in HPARAMS['GAMMA']:
    for TAU in HPARAMS['TAU']:
        for LR in HPARAMS['LR']:
            hparams = {
                'GAMMA': GAMMA,
                'TAU': TAU,
                'LR': LR,}
            
            print(f"\n{j}/{permuts}\tGamma {GAMMA} \ttau {TAU} \tlr {LR}")

            run_id = f"{j}_G{str(GAMMA)[1:]}_T{str(TAU)[1:]}_LR{str(LR)[1:]}"
            abs_dir = f"runs/{run_id}/"
            path = os.path.dirname(os.getcwd() + os.sep + abs_dir)  
            

            if os.path.isdir(path): # Skip if already done
                if glob.glob(os.path.join(path, f'*{MAX_EPISODES}_max*')) or glob.glob(os.path.join(path, f'*{MAX_EPISODES}_solved*')):
                    print('^ \tSkipping')
                    j += 1
                    continue
                else:
                    print('^ \tCleaning')
                    for f in glob.glob(path+'/*'):
                        os.remove(f)
                    os.rmdir(path)
            
            writer = SummaryWriter(path, purge_step=0)
            writer.add_hparams(hparams, {metric: run_training(hparams, path) for metric in METRICS}, run_name=path)
            
            j += 1
            env.reset()


# In[10]:


#record video

def Video(agent):
    agent.q_network.load_state_dict(torch.load(r'C:\Users\luken\Desktop\GOOD\runs\{}\save_{}.pth'.format(folder, MAX_EPISODES)))

    env = gym.wrappers.RecordVideo(env, r'C:\Users\luken\Desktop\GOOD\runs\{}'.format(folder))

    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state

    # Close the environment
    env.close()
