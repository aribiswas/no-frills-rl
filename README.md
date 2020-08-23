# No-Frills RL: Simple implementations of popular RL algorithms

No-Frills RL is a repository for reinforcement learning algorithms with simple and easy-to-understand implementation in PyTorch. The current implmentation uses Python 3.6 and has been tested with PyTorch 1.5.0.

No-Frills RL is primarily intended to be used for personal or educational purposes.

## Getting Started

### Install dependencies
To use no-frills-rl you must have Python, PyTorch and Gym installed. One easy way to install Python is to install the [Anaconda distribution](https://docs.anaconda.com/anaconda/install/), which also comes with useful IDEs like JupyterLab and Spyder.

After installing Python, follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch. If you face issues with the latest version while running the code, feel free to roll back to version 1.5.0 (link to previous versions also in the link) and create an issue for me to investigate.

Finally, install Gym following the instructions [here](https://gym.openai.com/docs/#installation).

### Train agents

To train an agent, first create a Gym environment. Then start the training using the **train** function. 

```python
import gym
from dqn import train

env = gym.make('CartPole-v1')
train(env, max_episodes=500, stop_value=150, render=True)
```
The train function needs only a Gym environment as argument, but you can specify additional hyperparameters. For a full list of hyperparameters, type help on the function after importing the RL module. For example, to see the help on DQN agent, type
```python
help(dqn.train)
```


