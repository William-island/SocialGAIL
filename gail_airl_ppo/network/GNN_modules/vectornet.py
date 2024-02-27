from .finalmlp import FinalPredMLP
from .selfatten import SelfAttentionLayer
from .subgraph import SubGraph
import os
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing, max_pool
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.data import Data, DataLoader
import sys
sys.path.append('..')


class HGNN(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, goal_shape=2, num_subgraph_layers=2, num_global_graph_layer=1, subgraph_width=32, global_graph_width=32, final_mlp_hidden_width=64):
        super(HGNN, self).__init__()
        self.goal_shape=goal_shape
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)
        self.traj_pred_mlp = FinalPredMLP(global_graph_width+self.goal_shape, out_channels, final_mlp_hidden_width)

    def forward(self, data):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len,goal,last_speed]
        """
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len
        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new=torch.cat((out[:, [0]].squeeze(1),data.goal.view(-1,2)),dim=1)
        pred = self.traj_pred_mlp(out_new)
        return pred
    


class HGNN_Disc(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, action_dim, out_channels=1, goal_shape=2, num_subgraph_layers=2, num_global_graph_layer=1, subgraph_width=32, global_graph_width=32, final_mlp_hidden_width=64):
        super(HGNN_Disc, self).__init__()
        self.goal_shape = goal_shape
        self.action_dim = action_dim
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)
        self.traj_pred_mlp = FinalPredMLP(global_graph_width+self.goal_shape+self.action_dim, out_channels, final_mlp_hidden_width)

    def forward(self, state, action):
        """
        args: 
            state (Data): [x, cluster, edge_index, valid_len,goal,last_speed]
            action
        """
        time_step_len = int(state.time_step_len[0])
        valid_lens = state.valid_len
        sub_graph_out = self.subgraph(state)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new=torch.cat((out[:, [0]].squeeze(1), state.goal.view(-1,2), action.view(-1,2)), dim=1)
        pred = self.traj_pred_mlp(out_new)
        return pred