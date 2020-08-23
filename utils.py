# -*- coding: utf-8 -*-

import torch
import numpy as np


# * * * * * * * UTILITY CLASSES * * * * * * *


class ExperienceBuffer:
    
    def __init__(self, state_dim, act_dim=1, max_len=int(1e6)):
        """
        Initialize a replay memory for storing experiences.
        
        For continuous action agents, provide act_dim for storing actions correctly.
        For discrete action agents, do not provide act_dim.
        
        Parameters
        ----------
        state_dim : number
            Dimension of states.
        act_dim : number
            Dimension of actions.
        max_len : number
            Capacity of memory.
            
        """
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_len = max_len
        self.last_idx = -1
        
        # memory is a dictionary that stores various elements as numpy arrays.
        # each array has dimensions max_len x <element_dimension>
        self.memory = dict(states = np.empty((self.max_len,) + self.state_dim),
                           actions = np.empty((self.max_len, self.act_dim)),
                           rewards = np.empty((self.max_len, 1)),
                           next_states = np.empty((self.max_len,) + self.state_dim),
                           dones = np.empty((self.max_len,1)))
        
        
    def add(self, state, action, reward, next_state, done):
        """
        Add experiences to replay memory.

        Parameters
        ----------
        state : numpy array
            State of the environment.
        action : numpy array
            Actions.
        reward : numpy array
            Rewards.
        next_state : numpy array
            Next states.
        done : numpy array
            Termination flag.

        """
        
        # increment last index in circular fashion
        self.last_idx += 1
        if self.last_idx >= self.max_len:
            self.last_idx = 0
        
        # append experiences
        self.memory['states'][self.last_idx] = state
        self.memory['actions'][self.last_idx] = action
        self.memory['rewards'][self.last_idx] = reward
        self.memory['next_states'][self.last_idx] = next_state
        self.memory['dones'][self.last_idx] = done
        
        
    def sample(self, batch_size, device='cpu'):
        """
        Get randomly sampled experiences.

        Parameters
        ----------
        batch_size : number
            Batch size.
        device : char, optional
            cpu or gpu. The default is 'cpu'.

        Returns
        -------
        data : dictionary
            Dictionary with keys 'states', 'actions', 'rewards', 'next_states'
            and 'dones'

        """
        
        # random indices
        batch_idxs = np.random.choice(self.last_idx+1, batch_size)
        
        # convert to tensors
        states_batch = torch.from_numpy(self.memory['states'][batch_idxs]).float().to(device)
        actions_batch = torch.from_numpy(self.memory['actions'][batch_idxs]).float().to(device)
        rewards_batch = torch.from_numpy(self.memory['rewards'][batch_idxs]).float().to(device)
        next_states_batch = torch.from_numpy(self.memory['next_states'][batch_idxs]).float().to(device)
        dones_batch = torch.from_numpy(self.memory['dones'][batch_idxs]).float().to(device)
        
        return dict(states = states_batch, 
                    actions = actions_batch, 
                    rewards = rewards_batch,
                    next_states = next_states_batch,
                    dones = dones_batch)


    def len(self):
        """
        Return the current size of internal memory.
        
        """
        
        return self.last_idx + 1
    
    
class Logger:
    
    def __init__(self):
        """
        Initialize a Logger for storing data.

        """
        
        self.data = dict(actor_loss = [0], 
                         critic_loss = [0],
                         epsilon = [],
                         episode_reward = [],
                         average_reward = [])
        
        
    def get(self, name, start=0, end=-1):
        """
        Get data in logger at specified range of indices.

        Parameters
        ----------
        name : char
            DESCRIPTION.
        start : number, optional
            Start index. The default is 0.
        end : number, optional
            End index. The default is -1.

        Returns
        -------
        dictionary
            Data at specified range of indices.

        """
        
        assert(start>end), "start index must be less than or equal to end."
        
        if end==-1:
            end = len(self.data[name])
        return self.data[name][start:end]
    
    
    def last(self, name):
        """
        Get data in logger at specified range of indices.

        Parameters
        ----------
        name : char
            DESCRIPTION.
        start : number, optional
            Start index. The default is 0.
        end : number, optional
            End index. The default is -1.

        Returns
        -------
        dictionary
            Data at specified range of indices.

        """
        
        if not self.data[name]:
            elem = []
        else:
            idx = len(self.data[name]) - 1
            elem = self.data[name][idx]
        return elem
    
    
    def store(self, name, entry):
        """
        Store entry in log.

        Parameters
        ----------
        name : char
            DESCRIPTION.
        entry : numpy array or torch tensor
            Data to log.

        Returns
        -------
        None.

        """
        
        self.data[name].append(entry)
    
    
######################################
#         UTILITY FUNCTIONS
######################################
        
def soft_update(target_model, model, tau):
    """
    Soft update target networks.
    
    """
    with torch.no_grad():
        for target_params, params in zip(target_model.parameters(), model.parameters()):
            target_params.data.copy_(tau*params + (1-tau)*target_params.data)
    