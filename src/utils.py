# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import namedtuple


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




class PrioritizedExperienceReplay:

    def __init__(self, max_len=int(1e6)):

        self.max_len = max_len
        self.memory = {}


    def len(self):

        return len(self.memory)


    def get_max_prio(self):

        return max(self.memory.keys())


    def add(self, state, action, reward, next_state, done):

        experience = namedtuple('experience', ['state','action','reward','next_state','done'])
        e = experience(state=state, action=action, reward=reward, next_state=next_state, done=done)

        priority = self.get_max_prio()
        self.memory[priority] = e


    def sort(self):

        self.memory = sorted(self.memory.items(), key=lambda x: x[1])


    def sample(self, batch_size, device='cpu'):

        # To sample, the buffer is divided into K=batch_size equal segments.
        # One experience is then uniformly sampled from each segment to create
        # a batch of K experiences.

        num_segments = np.floor(self.len() / batch_size)

        # init batches
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []

        for idx in range(num_segments):

            # choose a segment
            seg_choice = np.random.choice(num_segments)

            # choose a transition in the segment
            exp_choice = seg_choice * batch_size + np.random.choice(batch_size)

            # get the experience for this segment
            key = self.memory.keys()[exp_choice]
            state, action, reward, next_state, done = self.memory[key]

            # append to batch
            states_batch.append(state)
            actions_batch.append(action)
            rewards_batch.append(reward)
            next_states_batch.append(next_state)
            dones_batch.append(done)

        # convert batches to torch
        return dict(states = torch.from_numpy(np.array(states_batch)).float().to(self.device),
                    actions = torch.from_numpy(np.array(actions_batch)).float().to(self.device),
                    rewards = torch.from_numpy(np.array(rewards_batch)).float().to(self.device),
                    next_states = torch.from_numpy(np.array(next_states_batch)).float().to(self.device),
                    dones = torch.from_numpy(np.array(dones_batch)).float().to(self.device))



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


class OUNoise:

    def __init__(self, size, mean=0, mac=0.15, var=0.1, varmin=0.01, decay=1e-6):
        """
        Initialize Ornstein-Uhlenbech action noise.

        """

        self.mean = mean * np.ones(size)
        self.mac = mac
        self.var = var
        self.varmin = varmin
        self.decay = decay
        self.x = np.zeros(size)
        self.xprev = self.x
        self.step_count = 0
        self.size = size

    def step(self):
        """
        Step the OU noise model by computing the noise and decaying variance.

        """
        self.x = self.xprev + self.mac * (self.mean - self.xprev) + self.var * np.random.randn(self.size)
        self.xprev = self.x
        dvar = self.var * (1-self.decay)
        self.var = np.maximum(dvar, self.varmin)
        self.step_count += 1
        return self.x



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


def epsilon_sim(steps,epsilon,decay,epsilon_min):
    for i in range(steps):
        epsilon = max(epsilon_min, epsilon * (1-decay))
    print('Final epsilon after {:d} steps = {:f}'.format(steps,epsilon))
