import torch
import torch.nn as nn
import torch.nn.functional as F
#from collections import OrderedDict

class QNetwork(nn.Module):
    """Deep Q Network """

    def __init__(self, state_size,
                 action_size, hidden_layers=[256, 128, 64],
                 seed=1):
        """Initialize paramteres and build model
        Params
        ======
            state_size (int): Dimension of state
            action_size (int): Dimension of action
            hidden_layers: list of integers, the sizes of the hidden layers
            seed (int): random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # Input layer to hidden layer
        self.hidden_layers = \
            nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])

        # Add a variable number of hidden hidden_layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state to action values"""
        for each in self.hidden_layers:
            state = F.relu(each(state))
        return self.output(state)
