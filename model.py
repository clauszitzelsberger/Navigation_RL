import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QNetwork(nn.Module):
    """Deep Q Network """

    def __init__(self, state_size,
                 action_size, fc1=256,
                 fc2=128, fc3=64,
                 seed=1):
        """Initialize paramteres and build model
        Params
        ======
            state_size (int): Dimension of state
            action_size (int): Dimension of action
            fc1, fc2, fc3 (int): Number of nodes for each hidden layer
            seed (int): random seed
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(OrderedDict([
                  ('fc1', nn.Linear(state_size, fc1)),
                  ('relu1', nn.ReLU()),
                  ('fc2', nn.Linear(fc1, fc2)),
                  ('relu2', nn.ReLU()),
                  ('fc3', nn.Linear(fc2, fc3)),
                  ('relu3', nn.ReLU()),
                  ('output', nn.Linear(fc3, action_size))]))

    def forward(self, state):
        """Build a network that maps state to action values"""
        return self.model.forward(state)
