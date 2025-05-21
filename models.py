import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
import random
from torch_geometric.utils import dropout_adj
from datetime import datetime
import numpy as np 

class GCNII(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, alpha=0.1, theta=0.5):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_c, hid_c))
        self.lins.append(torch.nn.Linear(hid_c, out_c))
        self.convs = torch.nn.ModuleList([
            GCN2Conv(hid_c, alpha=alpha, theta=theta, layer=i + 1) for i in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        x0 = self.lins[0](x)  
        x = x0
        for conv in self.convs:
            x = conv(x, x0, edge_index)
            x = F.relu(x)
        x = self.lins[1](x)
        return x
class GCN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, dropout=0.5):
        super().__init__()
        self.proj = torch.nn.Linear(in_c, hid_c)
        self.convs = torch.nn.ModuleList([
            GCNConv(hid_c, hid_c) for _ in range(num_layers)
        ])
        self.out = torch.nn.Linear(hid_c, out_c)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.proj(x)
        for i, conv in enumerate(self.convs):
            h = F.relu(conv(h, edge_index))
            # apply dropout after each layer except before final output
            if i != len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h)


# --- ARGC_GCN: ARGC variant using standard GCNConv ---
class ARGC(torch.nn.Module):
    """
    Adaptive Residual Gate Convolution using standard GCNConv.
    """
    def __init__(self, in_c, hid_c, out_c, num_layers,
                 dropout=0.3, tau_init=0.5, alpha_skip=0.3):
        super().__init__()
        # Hyperparameters
        self.base_drop = dropout
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.alpha_skip = alpha_skip

        # Input projection
        self.proj = nn.Linear(in_c, hid_c)

        # GCNConv stack
        self.convs = nn.ModuleList([
            GCNConv(hid_c, hid_c) for _ in range(num_layers)
        ])

        # Shared adaptive scalar gate
        self.gate = nn.Sequential(
            nn.Linear(hid_c * 2, hid_c // 2),
            nn.ReLU(),
            nn.Linear(hid_c // 2, 1)
        )
        nn.init.constant_(self.gate[-1].bias, 5.0)

        # Normalization layers
        self.norm_z = nn.LayerNorm(hid_c * 2)
        self.norm_post = nn.LayerNorm(hid_c)

        # Output layer
        self.out = nn.Linear(hid_c, out_c)

    def forward(self, x, edge_index):
        h0 = self.proj(x)
        h = h0
        L = len(self.convs)

        for layer_idx, conv in enumerate(self.convs):
            # raw computation with GCNConv
            raw = conv(h, edge_index)
            # Inject initial features to combat deep-layer smoothing
            raw = raw + self.alpha_skip * h0

            # Gate input normalization
            z = torch.cat([raw, h], dim=-1)
            z = self.norm_z(z)

            # Scalar gate
            gate = torch.sigmoid(self.gate(z) / self.tau)  # [N,1]

            # Adaptive residual mix with input skip
            h = gate * raw + (1 - gate) * (h + self.alpha_skip * h0)

            # Post-norm + dropout floor + activation
            h = self.norm_post(h)
            drop_p = max(0.1, self.base_drop * (1 - layer_idx / L))
            h = F.dropout(h, p=drop_p, training=self.training)
            h = F.relu(h)

        return self.out(h)


