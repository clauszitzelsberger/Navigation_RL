import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from collections import deque
#import pdb

from agent import Agent
from model import QNetwork

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
    
        



def dqn(env, brain_name, 
        agent, n_episodes=2000,
        epsilon_start=1.0, epsilon_end=.1, 
        epsilon_decay=.99):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
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
            checkpoint = {'state_size': state_size,
                          'action_size': action_size,
                          'hidden_layers': [each.out_features for each in agent.qnet_local.hidden_layers],
                          'state_dict': agent.qnet_local.state_dict()}
            torch.save(checkpoint, 'checkpoint.pth')
            break
    return scores

def apply(env, brain_name, filepath):
    model = load_checkpoints(filepath)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        state = torch.from_numpy(state).float().unsqueeze(0).to('cpu')
        model.eval()
        with torch.no_grad():
            action_values = model(state)
        action = np.argmax(action_values.cpu().data.numpy())
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
    print('Score: {}'.format(score))

def plot_scores(scores_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for key, scores in scores_dict.items():
        scores_smoothed = gaussian_filter1d(scores, sigma=5)
        plt.plot(np.arange(len(scores)), scores_smoothed, label=key)
    plt.ylabel('smoothed Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()

def load_checkpoints(filepath):
    checkpoint = torch.load(filepath)
    model = QNetwork(checkpoint['state_size'],
                     checkpoint['action_size'],
                     checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    env, brain_name, state_size, action_size = \
        initialize_env('Banana_Linux/Banana.x86')
        
    # Initialize agents
    hidden_layers = [256, 128, 64]
    tau_soft = 1e-3
    update_target_soft = 4
    
    agent1 = Agent(state_size, action_size, 
                   hidden_layers)
    agent2 = Agent(state_size, action_size, 
                   hidden_layers, tau=tau_soft, 
                   update_target=update_target_soft)
    agent3 = Agent(state_size, action_size, 
                   hidden_layers, ddqn=True)
    agent4 = Agent(state_size, action_size, 
                   hidden_layers, tau=tau_soft, 
                   update_target=update_target_soft, ddqn=True)
    
    # Train agent
    n = 2000
    
    scores_dqn = dqn(env, brain_name, agent1, n_episodes=n)
    scores_dqn_soft_update = dqn(env, brain_name, agent2, n_episodes=n)
    scores_ddqn = dqn(env, brain_name, agent3, n_episodes=n)
    scores_ddqn_soft_update = dqn(env, brain_name, agent4, n_episodes=n)
    
    
    plot_scores({'DQN': scores_dqn,
                 'DQN, soft update': scores_dqn_soft_update,
                 'DDQN': scores_ddqn,
                 'DDQN, soft update': scores_ddqn_soft_update})
    
    # Show trained agents how it is acting in the environment
    apply(env, brain_name, 'checkpoint.pth')
    
    env.close()
