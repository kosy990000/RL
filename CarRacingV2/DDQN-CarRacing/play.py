
import gymnasium as gym
from IPython.display import HTML

import torch

import src.DDQN as DDQN
from src.Preprocess import ImageEnv

def play():
    env = gym.make('CarRacing-v2',continuous=False, render_mode="human")
    env = ImageEnv(env)

    s, _ = env.reset()
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    agent = DDQN.DDQN(state_dim, action_dim, epsilon=0.1)
    agent.total_steps = 100000

    agent.network.load_state_dict(torch.load('data/DDQNEi_dqn.pt',  weights_only=True))
    agent.target_network.load_state_dict(agent.network.state_dict())



    s, _ = env.reset()

    done = False
    while not done:
        a = agent.act(s, training=False)
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

play()