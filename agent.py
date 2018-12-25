
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
                            
device = 'cpu'

class Agent():

    def __init__(self, state_size,
                 action_size, hidden_layers,
                 buffer_size=int(1e6), batch_size=32,
                 gamma=.99, tau=1,
                 lr=2.5e-4, update_local=4,
                 update_target=10000, ddqn=False, 
                 seed=1):
        """Initialize Agent object

        Params
        ======
            state_size (int): Dimension of states
            action_size (int): Dimension of actions
            hidden_layers (list of ints): number of nodes in the hidden layers
            buffer_size (int): size of replay buffer
            batch_size (int): size of sample
            gamma (float): discount factor
            tau (float): (soft) update of target parameters
            lr (float): learning rate
            update_local (int): update local after every x steps
            update_target (int): update target after every x steps
            ddqn (boolean): Double Deep Q-Learning
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Hyperparameters
        self.buffer_size = buffer_size      # replay buffer
        self.batch_size = batch_size        # minibatch size
        self.gamma = gamma                  # discount factor
        self.tau = tau                      # (soft) update of target parameters
        self.lr = lr                        # learning rate
        self.update_local = update_local    # update local network after every x steps
        self.update_target = update_target  # update target network with local network weights

        # Q Network
        self.qnet_local = \
            QNetwork(state_size, action_size, hidden_layers, seed).to(device)
        self.qnet_target = \
            QNetwork(state_size, action_size, hidden_layers, seed).to(device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step
        self.t_step = 0
        
        # Double Deep Q-Learning flag
        self.ddqn = ddqn
        
        
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE LOCAL time steps
        self.t_step += 1
        if self.t_step % self.update_local == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                sample = self.memory.sample()
                if self.t_step % self.update_target == 0:
                    do_target_update = True
                else:
                    do_target_update = False
                self.__learn(sample, self.gamma, do_target_update)

    def act(self, state, epsilon=0):
        """Returns action given a state according to local Q Network (current policy)

        Params
        ======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        # Epsilon greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def __learn(self, sample, gamma, do_target_update):
        """Update value parameters using given batch of sampled experiences tuples

        Params
        ======
            sample (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = sample
        
        if not self.ddqn:

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = \
                self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
            
        else:
            # Get actions (for next states) with max Q values from local net
            next_actions = \
                self.qnet_local(next_states).detach().max(1)[1].unsqueeze(1)
                
            # Get predicted Q values from target model
            Q_targets_next = \
                self.qnet_target(next_states).gather(1, next_actions)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnet_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if do_target_update:
            self.__target_net_update(self.qnet_local, self.qnet_target, self.tau)

    def __target_net_update(self, local_net, target_net, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param \
            in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.\
                copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
    def get_info(self):
        output = """
            Replay Buffer size: {} \n
            Batch size: {} \n
            Discout factor: {} \n
            tau: {} \n
            Learning Rate: {} \n
            Update local network after every {} steps \n
            Update target network with local network parameters after every {} steps \n
            DDQN: {}
        """
        print(output.format(self.buffer_size, self.batch_size, 
                            self.gamma, self.tau, 
                            self.lr, self.update_local,
                            self.update_target, self.ddqn))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples in"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batchparamteres
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = \
            namedtuple('Experience',
                       field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.memory.\
            append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.\
            from_numpy(np.vstack([e.state for e in experiences if e is not None])).\
            float().to(device)
        actions = torch.\
            from_numpy(np.vstack([e.action for e in experiences if e is not None])).\
            long().to(device)
        rewards = torch.\
            from_numpy(np.vstack([e.reward for e in experiences if e is not None])).\
            float().to(device)
        next_states = torch.\
            from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).\
            float().to(device)
        dones = torch.\
            from_numpy(np.vstack([e.done for e in experiences if e is not None]).\
            astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
