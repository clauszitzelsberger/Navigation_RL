import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from collections import deque
#import pdb

from agent import Agent

def initialize_env(unity_file):
    # Initialize the environment
    env = UnityEnvironment(file_name=unity_file)

    # Get default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Get state and action spaces
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    print('State size:', state_size)
    print('Action size:', action_size)

    return env, brain_name, state_size, action_size




def dqn(env, brain_name, n_episodes=2000,
        max_steps=1000, epsilon_start=1.0,
        epsilon_end=.01, epsilon_decay=.99):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_stepsx_t (int): maximum number of timesteps per episode
        epsilon_start (float): starting value of epsilon, for epsilon-greedy action selection
        epsilon_end (float): minimum value of epsilon
        epsilon_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []
    scores_window = deque(maxlen=100)
    epsilon = epsilon_start
    for e in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break

        # Relative score
        scores_window.append(score)
        scores.append(score)

        # Update epsilon
        epsilon = max(epsilon_end, epsilon_decay*epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)), end="")
        if e % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e-100, np.mean(scores_window)))
            torch.save(agent.qnet_local.state_dict(), 'checkpoint.pth')
            break
    env.close()
    return scores

def apply():
    pass

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scores_smoothed = gaussian_filter1d(scores, sigma=10)
    plt.plot(np.arange(len(scores)), scores_smoothed)
    #plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('smoothed Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    env, brain_name, state_size, action_size = \
        initialize_env('Banana_Linux/Banana.x86')
    # Initialize agent
    agent = Agent(state_size, action_size)
    scores = dqn(env, brain_name, n_episodes=5000)
    plot_scores(scores)
