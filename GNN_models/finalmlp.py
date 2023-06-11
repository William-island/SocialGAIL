from torch_geometric.data import Data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np



class FinalPredMLP(nn.Module):
    """Predict one feature speeed vector, in offset format"""

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(FinalPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)