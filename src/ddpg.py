# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import numpy as np
import models
import utils

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#              Deep Deterministic Policy Gradient (DDPG)              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def ddpg_params(buffer_size=int(1e6),
                gamma=0.99,
                batch_size=64,
                actor_lr=0.01,
                critic_lr=0.01,
                update_freq=5,
                tau=0.01,
                device='cpu',
                train_iters=50,
                grad_threshold=None,
                noise_mean=0,
                noise_mac=0.2,
                noise_var=0.05,
                noise_var_min=0.005,
                noise_var_decay=5e-6):
    """
    Parameters
    ----------
    buffer_size : number, optional
        Capacity of experience buffer. The default is int(1e6).
    gamma : number optional
        Discount factor. The default is 0.99.
    batch_size : number, optional
        Batch size for training. The default is 128.
    actor_lr : number, optional
        Learn rate for actor neural network. The default is 0.01.
    critic_lr : number, optional
        Learn rate for critic neural network. The default is 0.01.
    update_freq : number, optional
        Update frequency for target Q-Network. The default is 0.01.
    tau : number, optional
        Smoothing factor for target Q-Network update. The default is 0.01.
    device : str
        Device 'cpu' or 'cuda:0'
    train_iters : number, optional
        Number of training iterations.
    grad_threshold : number, optional
        Threshold for gradients.
    noise_mean : number, optional
        Mean of OU noise. The default is 0.
    noise_mac : number, optional
        Mean attraction coefficient of OU noise. The default is 0.2.
    noise_var : number, optional
        Variance of OU noise. The default is 0.05.
    noise_var_min : number, optional
        Minimum variance of OU noise. The default is 0.005.
    noise_var_decay : number, optional
        Variance decay rate of OU noise. The default is 5e-6.

    Returns
    -------
    params

    """
    params = {
        'buffer_size': buffer_size,
        'gamma': gamma,
        'batch_size': batch_size,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'update_freq': update_freq,
        'tau': tau,
        'train_iters': train_iters,
        'grad_threshold': grad_threshold,
        'noise_mean': noise_mean,
        'noise_mac': noise_mac,
        'noise_var': noise_var,
        'noise_var_min': noise_var_min,
        'noise_var_decay': noise_var_decay
    }
    return params


class DDPGAgent:

    def __init__(self, obs_size, act_size, seed=0, params=None, logger=None):
        """
        Initialize a Deep Deterministic Policy Gradient (DDPG) agent.

        Parameters
        ----------
        obs_size : number
            Number of observation elements.
        act_size : number
            Number of action elements.
        seed : number, optional
            Random seed. The default is 0.
        params :
            Hyperparameters data structure.

        """

        if params is None:
            params = ddpg_params()

        # logger for storing training data
        self.logger = logger

        # parameters
        self.params = params
        self.step_count = 0

        if not torch.cuda.is_available() and self.params['device'] != 'cpu':
            print("GPU is not available. Selecting CPU...")
            self.params['device'] = 'cpu'

        # initialize actor
        self.actor = models.DeterministicActor(obs_size, act_size, seed).to(self.params['device'])
        self.target_actor = models.DeterministicActor(obs_size, act_size, seed)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # initialize critic
        self.critic = models.QCritic(obs_size, act_size, seed).to(self.params['device'])
        self.target_critic = models.QCritic(obs_size, act_size, seed)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.params['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.params['critic_lr'])

        # Experience replay
        self.buffer = utils.ExperienceBuffer(obs_size, act_size, params['buffer_length'])

        # Noise model
        self.noise_model = utils.OUNoise(size=act_size,
                                         mean=self.params['noise_mean'],
                                         mac=self.params['noise_mac'],
                                         var=self.params['noise_var'],
                                         varmin=self.params['noise_var_min'],
                                         decay=self.params['noise_decay'])


    def get_action(self, state, clip=(-1, 1), train_mode=False):
        """
        Get the action by sampling from the policy. If train is set to True
        then the action contains added noise.

        Parameters
        ----------
        state : numpy array or tensor
            State of the environment.
        clip : tuple
            Upper and lower action bounds. The default is (-1,1).
        train_mode : boolean, optional
            Flag for train mode. The default is False.

        Returns
        -------
        action : numpy array
            Action with optional added noise.

        """

        with torch.no_grad():
            action = self.actor.mu(state).cpu().numpy()

        # If in train mode then add noise
        if train_mode:
            noise = self.noise_model.step()
            action += noise

        # clip the action, just in case
        action = np.clip(action, clip[0], clip[1])

        return action


    def step(self, state, action, reward, next_state, done):
        """
        Step the agent, store experiences and learn.

        Parameters
        ----------
        state : numpy array
            State of the environment.
        action : numpy array
            Actions, given the states.
        reward : numpy array
            Reward obtained from the environment.
        next_state : numpy array
            Next states of the environment.
        done : numpy array
            Termination criteria.

        """

        # add experience to replay
        self.buffer.add(state, action, reward, next_state, done)

        # increase step count
        self.step_count += 1

        # learn from experiences
        if self.buffer.__len__() > self.params['batch_size']:

            # train with multiple sgd steps
            for _ in range(self.params['train_iters']):

                # create mini batch for learning
                experiences = self.buffer.sample(self.params['batch_size'], self.params['device'])

                # train the agent
                self.learn(experiences)


    def learn(self, experiences):
        """
        Train the actor and critic.

        Parameters
        ----------
        experiences : list
            Experiences (s,a,r,s1,d).

        """

        # unpack experience
        states, actions, rewards, next_states, dones = experiences

        # normalize rewards
        #if self.normalize_rewards:
        #    rewards = (rewards - np.mean(self.buffer.rew_buf)) / (np.std(self.buffer.rew_buf) + 1e-5)

        # compute td targets
        with torch.no_grad():
            target_action = self.target_actor.mu(next_states)
            targetQ = self.target_critic.Q(next_states,target_action)
            y = rewards + self.params['gamma'] * targetQ * (1-dones)

        # compute local Q values
        Q = self.critic.Q(states, actions)

        # critic loss
        critic_loss = torch.mean((y-Q)**2)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.params['grad_threshold'])  # gradient clipping
        self.critic_optimizer.step()

        # freeze critic before policy loss computation
        for p in self.critic.parameters():
            p.requires_grad = False

        # actor loss
        actor_loss = -self.critic.Q(states, self.actor.mu(states)).mean()

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.params['grad_threshold'])  # gradient clipping
        self.actor_optimizer.step()

        # Unfreeze critic
        for p in self.critic.parameters():
            p.requires_grad = True

        # log the loss and noise
        self.logger.store('actor_loss', actor_loss.detach().cpu().data.numpy())
        self.logger.store('critic_loss', critic_loss.detach().cpu().data.numpy())
        self.logger.store('ou_noise', np.mean(self.noise_model.x))

        # soft update target actor and critic
        if self.step_count % self.params['update_freq'] == 0:
            utils.soft_update(self.target_actor, self.actor, self.params['tau'])
            utils.soft_update(self.target_critic, self.critic, self.params['tau'])


    def attach_logger(self, logger):
        self.logger = logger
