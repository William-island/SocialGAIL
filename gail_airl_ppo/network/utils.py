import math
import torch
from torch import nn
from torch.distributions import Normal
import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
import pickle



def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)



def reparameterize_old(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)



def reparameterize(in_means, log_stds):
    mean = torch.tanh(in_means)
    log_std = log_stds.expand_as(mean)
    std = torch.exp(log_std)
    dist = Normal(mean, std)
    actions = dist.sample()
    actions = torch.clamp(actions, -1, 1)
    return actions, dist.log_prob(actions).sum()


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi_old(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def evaluate_lop_pi(in_means, log_stds, actions):
    mean = torch.tanh(in_means)
    log_std = log_stds.expand_as(mean)
    std = torch.exp(log_std)
    dist = Normal(mean, std)
    log_pis = dist.log_prob(actions).sum(1,keepdim=True)
    return log_pis


