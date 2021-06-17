# No-Frills RL: Simple implementations of popular RL algorithms

No-Frills RL is a repository for reinforcement learning algorithms with simple and easy-to-understand implementation in PyTorch. The current implmentation uses Python 3.6 and has been tested with PyTorch 1.5.0. There is no support for scalability yet.

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
from dqn import DQNAgent
from trainer import train

env = gym.make('CartPole-v1')
agent = DQNAgent(obs_dim=[4,1], actions=[0,1])
train(env,agent)
```
The train function needs a Gym environment and a reinforcement learning agent as arguments. You can also specify hyperparameters through optional arguments. 
```python
from dqn import dqn_params
from trainer import train_params

agent_params = dqn_params(batch_size=256, lr=0.001)
training_params = train_params(max_episodes=100,stop_value=200)

agent = DQNAgent(obs_dim=[4,1], actions=[0,1], agent_params)
train(env, agent, training_params)
```


