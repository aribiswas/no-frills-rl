# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import numpy as np
import models
import utils


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        Deep Q-Network (DQN)                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class DQNAgent:
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, obs_dim, actions, seed, params):
        """
        Initialize a Deep Q-Network agent.

        Parameters
        ----------
        obs_dim : tuple
            Dimension of observations.
        num_act : number
            Number of possible actions.
        seed : number
            Random seed.
        options : dictionary
            DQN hyperparameters.

        """
        
        # initialize agent parameters
        self.obs_dim = obs_dim
        self.actions = actions
        self.num_act = len(actions)
        self.buffer_size = params['BUFFER_SIZE']
        self.gamma = params['GAMMA']
        self.epsilon = params['EPSILON']
        self.epsilon_decay = params['EPSILON_DECAY']
        self.epsilon_min = params['EPSILON_MIN']
        self.batch_size = params['BATCH_SIZE']
        self.lr = params['LR']
        self.update_freq = params['UPDATE_FREQ']
        self.tau = params['TAU']
        self.step_count = 0
        
        # initialize logger for storing training data
        self.logger = utils.Logger()
        
        # set the random seed
        self.seed = torch.manual_seed(seed)
        
        # create local and target Q networks
        self.Qnet = models.QNetwork(self.obs_dim, self.num_act, seed).to(self.device)
        self.target_Qnet = models.QNetwork(self.obs_dim, self.num_act, seed).to(self.device)
        self.target_Qnet.load_state_dict(self.Qnet.state_dict())  # copy network weights to make identical
        
        # initialize optimizer
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=self.lr)
        
        # initialize experience buffer
        self.buffer = utils.ExperienceBuffer(self.obs_dim, max_len=self.buffer_size)
        
        
    def step(self, state, action, reward, next_state, done):
        """
        Step the agent by storing experiences and learning from data.

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

        Returns
        -------
        None.

        """
        
        # increase step count
        self.step_count += 1
        
        # convert action to action choice
        act_choice = self.actions.index(action)
        
        # add experience to buffer
        self.buffer.add(state, act_choice, reward, next_state, done)
        
        # learn from experiences
        if self.buffer.len() > self.batch_size:
            
            # create batch experiences for learning
            experiences = self.buffer.sample(self.batch_size, self.device)
            
            # train the agent
            self.learn(experiences)
        
        # decay epsilon
        ep_next = self.epsilon * (1-self.epsilon_decay)
        self.epsilon = max(self.epsilon_min, ep_next)
        self.logger.store('epsilon', self.epsilon)
        
        
    def get_action(self, state):
        """
        Get action from the policy, given the state.

        Parameters
        ----------
        state : numpy array
            State of the environment.

        Returns
        -------
        action : numpy array
            Action.

        """
        
        # obtain network output Q(s,a1) ... Q(s,an)
        with torch.no_grad():
            y = self.Qnet(state)
            
        # select action
        ep_choice = np.random.rand(1)
        if ep_choice > self.epsilon:
            # epsilon greedy action (exploitation)
            action_choice = np.argmax(y.numpy())
        else:
            # random action (exploration)
            action_choice = np.random.choice(np.arange(self.num_act))
        
        return self.actions[action_choice]
        
    
    def learn(self, experiences):
        """
        Learn from experiences.

        Parameters
        ----------
        experiences : dictionary
            Batch experiences.

        Returns
        -------
        None.

        """
        
        # unpack experience
        states = experiences['states']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_states = experiences['next_states']
        dones = experiences['dones']
        
        with torch.no_grad():
            
            # *** Double DQN ***
            
            # amax = argmax Q(s+1) 
            a_max = torch.argmax(self.Qnet(next_states), dim=1)
            a_max = a_max.reshape((self.batch_size,1))  
            
            # Q'(s+1|amax)  -> q value for argmax of actions
            tQnet_out = self.target_Qnet(next_states)
            targetQ = torch.stack([tQnet_out[i][a_max[i]] for i in range(self.batch_size)])
            
            # y = r + gamma * Q'(s+1|amax) 
            y = rewards + self.gamma * targetQ * (1-dones)   
            
        # Q(s|a) -> q value for action from local policy
        Qnet_out = self.Qnet(states)
        Q = torch.stack([Qnet_out[i][actions[i].numpy()] for i in range(self.batch_size)])
        
        # calculate mse loss
        loss = torch.mean((y-Q)**2)
        
        # update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qnet.parameters(), 1)  # gradient clipping
        self.optimizer.step()
        
        # soft update target network
        if self.step_count % self.update_freq == 0:
            utils.soft_update(self.target_Qnet, self.Qnet, self.tau)
            
        # log data
        self.logger.store('critic_loss', loss.detach().cpu().data.numpy())
        

