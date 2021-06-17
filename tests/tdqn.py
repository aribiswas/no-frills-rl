# -*- coding: utf-8 -*-

import gym
from dqn import DQNAgent, dqn_params
from trainer import train, train_params

agent_params = dqn_params(buffer_size=int(1e6),
                          gamma=0.99,
                          epsilon=0.95,
                          epsilon_decay=5e-5,
                          epsilon_min=0.1,
                          batch_size=64,
                          lr=2e-4,
                          update_freq=5,
                          tau=0.01)
training_params = train_params(max_episodes=1200,
                               max_steps_per_episode=1000,
                               stop_value=475,
                               avg_window=100)

# create Gym environment
env = gym.make('CartPole-v1')

# create a DQN agent
obs_dim = env.observation_space.shape
actions = range(env.action_space.n)
agent = DQNAgent(obs_dim, actions, seed=0, params=agent_params)

# train the agent
train(env, agent, training_params)
