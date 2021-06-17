#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural networks for Deep RL algorithms.

@author: abiswas

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):

    def __init__(self, obs_dim, num_act, seed=0):
        """
        Initialize a Deep Q-network.

        Parameters
        ----------
        obs_dim : number, list, array
            Dimension of observations.
        num_act : number
            Number of possible actions.
        seed : number, optional
            Random seed. The default is 0.

        Returns
        -------
        None.

        """

        self.seed = torch.manual_seed(seed)
        self.image_obs = True if len(obs_dim) > 1 else False

        super(QNetwork,self).__init__()

        # initialize layers
        if self.image_obs:
            # Conv2d transformations:
            # Wout = (Win + 2*P - D*(K-1) -1)/S + 1
            # Wout: output size
            # Win: input size
            # P: padding
            # D: dilation
            # K: kernel size
            # S: stride
            self.conv1 = nn.Conv2d(obs_dim[2], 16, kernel_size=4, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            convw = (obs_dim[0] + 2*0 - 1*(4-1) - 1)/2 + 1
            convh = (obs_dim[1] + 2*0 - 1*(4-1) - 1)/2 + 1
            self.layer_size = [int(convw * convh * 16), 64, 32, num_act]
        else:
            self.layer_size = [obs_dim[0], 64, 32, num_act]

        # linear layers
        self.fc1 = nn.Linear(self.layer_size[0], self.layer_size[1])
        self.fc2 = nn.Linear(self.layer_size[1], self.layer_size[2])
        self.fc3 = nn.Linear(self.layer_size[2], self.layer_size[3])

    def forward(self, state):
        """
        Forward pass through the network.

        Parameters
        ----------
        state : numpy array or tensor
            State of the environment.

        Returns
        -------
        x : tensor
            State-action value Q(s,a).

        """

        # convert to torch
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        # make forward pass through the network
        if self.image_obs:

            # Inputs here are NxHxWxC (batch) or HxWxC (single)
            #
            # For conv2d, inputs must have dimension NxCxHxW
            # N = number of batches
            # C = number of channels
            # H = height of image
            # W = width of image
            N = x.shape[-4] if len(x.shape)==4 else 1
            H = x.shape[-3]
            W = x.shape[-2]
            C = x.shape[-1]
            x = x.reshape((N,C,H,W))

            x = F.relu(self.bn1(self.conv1(x)))
            x = x.view(-1,self.layer_size[0])   # flatten the tensor

        # continue with the linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DeterministicActor(nn.Module):

    def __init__(self, num_obs, num_act, seed=0):
        """
        Initialize a deterministic actor network.

        """

        super(DeterministicActor, self).__init__()

        torch.manual_seed(seed)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_obs = num_obs

        # layers
        self.fc1 = nn.Linear(num_obs,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,num_act)
        self.tanh = nn.Tanh()


    def forward(self, state):
        """
        Perform forward pass through the network.

        """

        # convert to torch
        if isinstance(state, numpy.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor) or isinstance(state, torch.cuda.FloatTensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


    def mu(self, state):
        """
        Get the deterministic policy.

        """

        return torch.clamp(self.forward(state), -1, 1)


"""
Critic class for approximating the state-action value function Q(s,a).

"""
class QCritic(nn.Module):

    def __init__(self, num_obs, num_act, seed=0):

        torch.manual_seed(seed)

        super(QCritic, self).__init__()

        self.num_obs = num_obs

        # ------ layers ------

        # state path
        self.sfc1 = nn.Linear(num_obs,128)

        # action path
        self.afc1 = nn.Linear(num_act,128)

        # common path
        self.cfc1 = nn.Linear(128*2,64)
        self.cfc2 = nn.Linear(64,1)


    def forward(self, state, action):
        """
        Perform forward pass through the network.

        """

        # convert to torch
        if isinstance(state, numpy.ndarray):
            s = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            s = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        if isinstance(action, numpy.ndarray):
            a = torch.from_numpy(action).float().to(self.device)
        elif isinstance(action, torch.Tensor) or isinstance(state, torch.cuda.FloatTensor):
            a = action
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        # state path
        xs = F.relu(self.sfc1(s))

        # action path
        xa = F.relu(self.afc1(a))

        # common path
        xc = torch.cat((xs,xa), dim=1)
        xc = F.relu(self.cfc1(xc))
        xc = self.cfc2(xc)

        return xc


    def Q(self, state, action):
        """
        Compute Q(s,a)

        """

        return self.forward(state, action)
