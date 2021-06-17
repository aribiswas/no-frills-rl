# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from dqn import DQNAgent, dqn_params
import collections, torch, gym, utils
import numpy as np

def train_params(max_episodes=1000,
                 max_steps_per_episode=1000,
                 stop_value=100,
                 avg_window=100,
                 render=False,
                 verbose=True,
                 save_agent=True,
                 save_file='checkpoint_dqn.pth'):
    """

    Parameters
    ----------
    max_episodes : number, optional
        Maximum number of training episodes. The default is 1000.
    max_steps_per_episode : number, optional
        Maximum environment steps in a training episode. The default is 1000.
    stop_value : number, optional
        Average score over avg_window episodes. Training is terminated on reaching this value. The default is 100.
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

    Returns
    -------

    """

    params = {
        'max_episodes': max_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'stop_value': stop_value,
        'avg_window': avg_window,
        'render': render,
        'verbose': verbose,
        'save_agent': save_agent,
        'save_file': save_file
    }
    return params

def train(env, agent, training_params=None):
    """
    Parameters
    ----------
    env:
        OpenAI Gym environment.
    agent:
        Reinforcement learning agent.
    training_params: dict
        Training hyperparameters.

    Returns
    -------

    """

    if training_params is None:
        training_params = train_params()

    # check if action space is discrete
    assert(isinstance(env.action_space,gym.spaces.discrete.Discrete)), \
        "Specify an environment with discrete action space."

    # create DQN agent
    obs_dim = env.observation_space.shape
    actions = range(env.action_space.n)
    agent = DQNAgent(obs_dim, actions, seed, params)

    # create a logger
    logger = utils.logger()
    agent.attach_logger(logger)

    # avg score window
    avg_window = collections.deque(maxlen=training_params['avg_window'])

    # train agent
    for episode in range(training_params['max_episodes']):

        # reset environment
        state = env.reset()

        ep_reward = 0

        for steps in range(training_params['max_steps_per_episode']):

            # visualize
            if training_params['render']:
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
        logger.store('episode_reward', ep_reward)
        logger.store('average_reward', avg_reward)

        # print training progress
        if training_params['verbose']:
            loss = logger.last('critic_loss')
            eps = logger.last('epsilon')
            print('Episode: {:4d} \tAgent Steps: {:8d} \tAverage Reward: {:6.2f} \tLoss: {:8.4f} \tEpsilon: {:6.4f}'.format(episode+1, agent.step_count, avg_reward, loss, eps))

        # check termination
        if avg_reward >= training_params['stop_value']:
            print('\nEnvironment solved in {:d} episodes!\tAverage Reward: {:6.2f}'.format(episode+1, avg_reward))
            break

    # save the policy
    if training_params['save_agent']:
        torch.save(agent.Qnet.state_dict(), training_params['save_file'])

    # Close environment
    env.close()

    # plot results
    plt.ion()

    # plot score history
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
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

    info = {
        'plot': (fig1, fig2, ax1, ax2),
        'logger': logger,
        'training_params': training_params
    }
    return info
