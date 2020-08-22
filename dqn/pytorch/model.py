#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-network representation.

@author: abiswas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, layer_size, seed=0):
        """
        Initialize a Q-network.

        Parameters
        ----------
        layer_size : list
            Size of layers.
        seed : number, optional
            Random seed. The default is 0.

        """

        self.seed = torch.manual_seed(seed)
        self.layer_size = layer_size

        super(QNetwork,self).__init__()

        # initialize layers
        for idx in range(len(layer_size)-1):
            self.fc[idx] = nn.Linear(layer_size[idx],layer_size[idx+1])


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
        if isinstance(state, numpy.ndarray):
            x = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        # make forward pass through the network
        for idx in range(len(self.layer_size)):
            x = F.relu(self.fc[idx](x))

        return x
