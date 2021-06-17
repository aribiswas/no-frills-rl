# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import numpy as np
import models
import utils


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        Deep Q-Network (DQN)                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def dqn_params(buffer_size=int(1e6),
               gamma=0.99,
               epsilon=0.9,
               epsilon_decay=1e-6,
               epsilon_min=0.1,
               batch_size=64,
               lr=0.01,
               update_freq=5,
               tau=0.01):
    """
    Parameters
    ----------
    buffer_size : number, optional
        Capacity of experience buffer. The default is int(1e6).
    gamma : number optional
        Discount factor. The default is 0.99.
    epsilon : number, optional
        Exploration parameter. The default is 0.05.
    epsilon_decay : number, optional
        Decay rate for epsilon. The default is 1e6.
    epsilon_min : number, optional
        Minimum value of epsilon. The default is 0.1.
    batch_size : number, optional
        Batch size for training. The default is 128.
    lr : number, optional
        Learn rate for Q-Network. The default is 0.01.
    update_freq : number, optional
        Update frequency for target Q-Network. The default is 0.01.
    tau : number, optional
        Smoothing factor for target Q-Network update. The default is 0.01.

    Returns
    -------
    params

    """
    params = {
        'buffer_size': buffer_size,
        'gamma': gamma,
        'epsilon': epsilon,
        'epsilon_decay': epsilon_decay,
        'epsilon_min': epsilon_min,
        'batch_size': batch_size,
        'lr': lr,
        'update_freq': update_freq,
        'tau': tau,
        'device': 'cpu'
    }
    return params


class DQNAgent:

    def __init__(self, obs_dim, actions_list, seed=0, params=None, logger=None):
        """
        Initialize a Deep Q-Network agent.

        Parameters
        ----------
        obs_dim: tuple
            Dimension of observations.
        actions_list: list
            List of possible actions.
        seed:
            Random seed.
        params: dict
            DQN hyperparameters.
        logger:
            Logger object.

        """

        if params is None:
            params = dqn_params()
        self.params = params

        if not torch.cuda.is_available() and self.params['device'] != 'cpu':
            print("GPU is not available. Selecting CPU...")
            self.params['device'] = 'cpu'

        # initialize agent parameters
        self.obs_dim = obs_dim
        self.actions = actions_list
        self.num_act = len(actions_list)
        self.step_count = 0

        # logger for storing training data
        self.logger = logger

        # set the random seed
        self.seed = torch.manual_seed(seed)

        # create local and target Q networks
        self.Qnet = models.QNetwork(self.obs_dim, self.num_act, seed).to(self.params['device'])
        self.target_Qnet = models.QNetwork(self.obs_dim, self.num_act, seed).to(self.params['device'])
        self.target_Qnet.load_state_dict(self.Qnet.state_dict())  # copy network weights to make identical

        # initialize optimizer
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=self.params['lr'])

        # initialize experience buffer
        self.buffer = utils.ExperienceBuffer(obs_dim, max_len=self.params['buffer_size'])


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
        if self.buffer.len() > self.params['batch_size']:

            # create batch experiences for learning
            experiences = self.buffer.sample(self.params['batch_size'], self.params['device'])

            # train the agent
            self.learn(experiences)

        # decay epsilon
        ep_next = self.epsilon * (1-self.params['epsilon_decay'])
        self.params['epsilon'] = max(self.params['epsilon_min'], ep_next)

        if self.logger is not None:
            self.logger.store('epsilon', self.params['epsilon'])


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
        if ep_choice > self.params['epsilon']:
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
            a_max = a_max.reshape((self.params['batch_size'],1))

            # Q'(s+1|amax)  -> q value for argmax of actions
            tQnet_out = self.target_Qnet(next_states)
            targetQ = torch.stack([tQnet_out[i][a_max[i]] for i in range(self.params['batch_size'])])

            # y = r + gamma * Q'(s+1|amax)
            y = rewards + self.params['gamma'] * targetQ * (1-dones)

        # Q(s|a) -> q value for action from local policy
        Qnet_out = self.Qnet(states)
        Q = torch.stack([Qnet_out[i][actions[i].numpy()] for i in range(self.params['batch_size'])])

        # calculate mse loss
        loss = torch.mean((y-Q)**2)

        # update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Qnet.parameters(), 1)  # gradient clipping
        self.optimizer.step()

        # soft update target network
        if self.step_count % self.params['update_freq'] == 0:
            utils.soft_update(self.target_Qnet, self.Qnet, self.params['tau'])

        # log data
        if self.logger is not None:
            self.logger.store('critic_loss', loss.detach().cpu().data.numpy())


    def attach_logger(self, logger):
        self.logger = logger
