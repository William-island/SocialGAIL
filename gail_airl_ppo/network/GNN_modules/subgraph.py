import copy
from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, max_pool
import numpy as np


class SubGraph(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', GraphLayerProp(in_channels, hidden_unit))
            in_channels *= 2

    def forward(self, data):
        """
        polyline vector set in torch_geometric.data.Data format
        args:
            sub_data (Data): [x, y, cluster, edge_index, valid_len]
        """
        sub_data = copy.deepcopy(data)
        x, edge_index = sub_data.x, sub_data.edge_index
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        # try:
        assert out_data.x.shape[0] % int(sub_data.time_step_len[0]) == 0
        # except:
            # from pdb import set_trace; set_trace()
        # out_data.x = out_data.x / out_data.x.norm(dim=1,keepdim=True)
        return out_data



class GraphLayerProp(MessagePassing):
    """
    Message Passing mechanism for infomation aggregation
    """

    def __init__(self, in_channels, hidden_unit=64, verbose=False):
        super(GraphLayerProp, self).__init__(
            aggr='max')  # MaxPooling aggragation
        self.verbose = verbose
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, in_channels)
        )

    def forward(self, x, edge_index):
        if self.verbose:
            print(f'x before mlp: {x}')
        x = self.mlp(x)
        if self.verbose:
            print(f"x after mlp: {x}")
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        if self.verbose:
            print(f"x after mlp: {x}")
            print(f"aggr_out: {aggr_out}")
        return torch.cat([x, aggr_out], dim=1)