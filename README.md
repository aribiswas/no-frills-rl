# No-Frills RL: Simple implementations of popular RL algorithms

No-Frills RL is a repository for reinforcement learning algorithms with simple and easy-to-understand implementation in PyTorch. The current implmentation uses Python 3.6 and has been tested with PyTorch 1.5.0.

No-Frills RL is primarily intended to be used for personal or educational purposes.

## Getting Started

### Install dependencies
To use no-frills-rl you must install Python and PyTorch. One easy way to install Python is to install the Anaconda distribution, which also comes with useful IDEs like JupyterLab and Spyder. You can follow the instructions [here](https://docs.anaconda.com/anaconda/install/) to install Anaconda.

After installing Python, follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch. If you face issues with the latest version while running the code, feel free to roll back to version 1.5.0 (link to previous versions also in this page) and create an issue for me to investigate.

### Train agents

To train an agent, first create an environment using Gym. Then start the training using the train_[agent] method. 

```python
import gym
from dqn import train_dqn

env = gym.make('CartPole-v1')
train_dqn(env, max_episodes=500, stop_value=150, render=True)
```
The train_[agent] function needs only a Gym environment as argument, but you can specify additional hyperparameters. For a full list of hyperparameters, type help on the function.
```python
help(train_[agent])
```


