# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from agents import DQNAgent
import numpy as np
import collections
import torch
import gym

def train(env,
          obs_dim=(),
          actions=[],
          seed=0,
          buffer_size=int(1e6),
          gamma=0.99,
          epsilon=0.9,
          epsilon_decay=1e-6,
          epsilon_min=0.1,
          batch_size=64,
          lr=0.01,
          update_freq=5,
          tau=0.01,
          max_episodes=1000,
          max_steps_per_episode=1000,
          stop_value=100,
          avg_window=100,
          render=False,
          verbose=True,
          save_agent=True,
          save_file='checkpoint_dqn.pth'):
    
    """
    Train a DQN agent for an environment.

    Parameters
    ----------
    env : OpenAI Gym environment.
        Environment.
        
    obs_dim : list
        Dimension of observations.
        
    actions : list
        List of possible actions.
        
    seed : number
        Random seed.
        
    buffer_size : number, optional
        Capcity of experience buffer. The default is int(1e6).
        
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
        
    max_episodes : number, optional
        Maximum number of training episodes. The default is 1000.
        
    max_steps_per_episode : number, optional
        Maximum environment steps in a training episode. The default is 1000.
        
    stop_value : number, optional
        Average score over avg_window episodes. Training is terminated on
        reaching this value. The default is 100.
        
    avg_window : number, optional
        Window length for score averaging. The default is 100.
        
    render : boolean, optional
        Flag to set environment visualization. The default is False.
        
    verbose : boolean, optional
        Flag for verbose training mode. The default is True.
        
    save_agent : boolean, optional
        Flag to save agent after training. The default is True.
        
    save_file : char, optional
        File name for saving agent. The default is 'checkpoint_dqn.pth'.

    """
    
    # check if action space is discrete
    assert(isinstance(env.action_space,gym.spaces.discrete.Discrete)), "Specify an environment with discrete action space."
    
    # I/O specs
    if not obs_dim:
        obs_dim = env.observation_space.shape
    if not actions:
        actions = np.arange(env.action_space.n)
    
    params = dict(BUFFER_SIZE=buffer_size,
                   GAMMA=gamma,
                   EPSILON=epsilon,
                   EPSILON_DECAY=epsilon_decay,
                   EPSILON_MIN=epsilon_min,
                   BATCH_SIZE=batch_size,
                   LR=lr,
                   UPDATE_FREQ=update_freq,
                   TAU=tau)
    
    # create DQN agent
    agent = DQNAgent(obs_dim, actions, seed, params)
    
    # avg score window
    avg_window = collections.deque(maxlen=avg_window)
    
    # train agent
    for episode in range(max_episodes):
        
        # reset environment
        state = env.reset()
        
        ep_reward = 0
        
        for steps in range(max_steps_per_episode):
            
            # visualize
            if render:
                env.render()
            
            # sample action from the current policy
            action = agent.get_action(state)
            
            # step the environment
            next_state, reward, done, info = env.step(action)
            
            # step the agent
            agent.step(state, action, reward, next_state, done)
            
            # update state
            state = next_state
            
            # cumulative reward
            ep_reward += reward
            
            # terminate if done
            if done:
                break
        
        # avg scores
        avg_window.append(ep_reward)
        avg_reward = np.mean(avg_window)
            
        # log scores
        agent.logger.store('episode_reward', ep_reward)
        agent.logger.store('average_reward', avg_reward)
        
        # print training progress
        if verbose:
            loss = agent.logger.last('critic_loss')
            eps = agent.logger.last('epsilon')
            print('Episode: {:4d} \tAgent Steps: {:8d} \tAverage Reward: {:6.2f} \tLoss: {:8.4f} \tEpsilon: {:6.4f}'.format(episode+1, agent.step_count, avg_reward, loss, eps))
            
        # check termination
        if avg_reward >= stop_value:
            print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:6.2f}'.format(episode+1, avg_reward))
            break
    
    # save the policy
    if save_agent:
        torch.save(agent.Qnet.state_dict(), save_file)
    
    # Close environment
    env.close()
    
    # plot results
    plt.ion()
    
    # plot score history
    fig1, ax1 = plt.subplots(1,1, figsize=(6,4), dpi=200)
    ax1.set_title("Training Results")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Score")
    ax1.plot(agent.logger.get('average_reward'))
    
    # plot loss
    fig2, ax2 = plt.subplots(1,1, figsize=(6,4), dpi=200)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    ax2.plot(agent.logger.get('critic_loss'))
    
    plt.show()
    
    return agent








