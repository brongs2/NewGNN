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
        x0 = self.lins[0](x)  # 🟢 x0도 변환된 hidden dim이어야 함
        x = x0
        for conv in self.convs:
            x = conv(x, x0, edge_index)
            x = F.relu(x)
        x = self.lins[1](x)
        return x
class GCN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers):
        super().__init__()
        self.proj = torch.nn.Linear(in_c, hid_c)
        self.convs = torch.nn.ModuleList([
            GCNConv(hid_c, hid_c) for _ in range(num_layers)
        ])
        self.out = torch.nn.Linear(hid_c, out_c)

    def forward(self, x, edge_index):
        h = self.proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return self.out(h)


# --- ARGC_GCN: ARGC variant using standard GCNConv ---
class ARGC_GCN(torch.nn.Module):
    """
    Adaptive Residual Gate Convolution using standard GCNConv.
    """
    def __init__(self, in_c, hid_c, out_c, num_layers,
                 dropout=0.3, tau_init=1.0, alpha_skip=0.2):
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


class GS(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.proj = torch.nn.Linear(in_c, hid_c)
        self.convs = torch.nn.ModuleList()
        layer_idx = 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
        layer_idx += 1
        for _ in range(num_layers-2):
            self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
            layer_idx += 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
    def forward(self, x, edge_index):
        x0 = self.proj(x)
        h = x0
        for i, conv in enumerate(self.convs):
            h = conv(h, x0, edge_index)
            if i != len(self.convs)-1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class GDN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, deltas=None, dropout=0.5):
        super().__init__()
        if deltas is None:
            self.deltas = [1.0 / (2 ** k) for k in range(num_layers)]
        else:
            assert len(deltas) == num_layers
            self.deltas = deltas
        self.dropout = dropout
        self.proj = torch.nn.Linear(in_c, hid_c)
        layer_idx = 1
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
            layer_idx += 1
        self.out_head = torch.nn.Linear(hid_c * 2, out_c)

    def quantize(self, h, delta):
        b = (torch.rand_like(h) - 0.5) * delta
        return torch.floor((h + b) / delta) * delta - b

    def forward(self, x, edge_index):
        h0 = self.proj(x)
        h = h0
        for k, conv in enumerate(self.convs):
            Δ = self.deltas[k]
            h_q = self.quantize(h, Δ)
            eps = torch.randn_like(h_q) * 0.01
            h = conv(h_q, h0, edge_index) + eps
            h = F.relu(h)
            if k != len(self.convs) -1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out_head(torch.cat([h0, h], dim=-1))
class GLF(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(in_c, hid_c))
        for _ in range(num_layers - 2):
            self.linears.append(torch.nn.Linear(hid_c, hid_c))
        self.linears.append(torch.nn.Linear(hid_c, out_c))

    def forward(self, x, edge_index):
        row, col = edge_index
        N = x.size(0)
        deg = torch.bincount(row, minlength=N).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = x
        for i in range(self.num_layers):
            sign = 1 if i % 2 == 0 else -1  # 홀수: +L / 짝수: -L
            message = sign * norm.unsqueeze(1) * h[row]
            agg = scatter_add(message, col, dim=0, dim_size=N)
            h = self.linears[i](agg)
            if i != self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class SRP(torch.nn.Module):
    """
    SRP introduces anchor–pointer pairs and a reversal loss to break oversmoothing.
    During forward it stores a Δ‑step‑old feature snapshot; call `.reversal_loss`
    after the forward pass to add the auxiliary objective.
    """
    def __init__(self, in_c, hid_c, out_c, num_layers, delta: int = 4, margin: float = 1.0, dropout=0.5):
        super().__init__()
        self.delta = delta
        self.margin = margin
        self.dropout = dropout
        self.gamma = 0.1  # scaling factor for high-frequency residual
        self.proj = torch.nn.Linear(in_c, hid_c)
        self.convs = torch.nn.ModuleList()
        layer_idx = 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
        layer_idx += 1
        for _ in range(num_layers - 2):
            self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
            layer_idx += 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
        # buffer to keep a snapshot that is delta layers old
        self.register_buffer("_snapshot", torch.empty(0))

    def forward(self, x, edge_index):
        x0 = self.proj(x)
        h = x0
        feats = []
        for i, conv in enumerate(self.convs):
            raw = conv(h, x0, edge_index)
            h = raw + h   # identity mapping
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            feats.append(h)
        # store snapshot that is `delta` layers behind the final output
        if self.training and len(feats) > self.delta:
            self._snapshot = feats[-1 - self.delta].detach()
        return h  # logits

    def reversal_loss(self, anchors: torch.Tensor, pointers: torch.Tensor) -> torch.Tensor:
        """Compute the reversal loss given index tensors `anchors` and `pointers`."""
        if self._snapshot.numel() == 0:
            return torch.tensor(0.0, device=anchors.device)
        dist = (self._snapshot[anchors] - self._snapshot[pointers]).pow(2).sum(-1).sqrt()
        loss = F.relu(self.margin - dist).mean()
        return loss

# ✅ Adversarial Spectral Augmentation (ASA) 모델
class ASA(torch.nn.Module):
    """
    ASA jointly trains an encoder and a lightweight adversary that perturbs edge
    weights (via dropout) to minimise the Laplacian spectral gap.  The encoder
    must remain robust to such perturbations, discouraging oversmoothing.
    """
    def __init__(self, in_c, hid_c, out_c, num_layers, drop_prob: float = 0.2, dropout=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.dropout = dropout
        self.proj = torch.nn.Linear(in_c, hid_c)
        self.convs = torch.nn.ModuleList()
        layer_idx = 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
        layer_idx += 1
        for _ in range(num_layers - 2):
            self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
            layer_idx += 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))

    def forward(self, x, edge_index):
        # Adversary: randomly drop/scale edges at every call (training only)
        if self.training:
            edge_index, _ = dropout_adj(edge_index, p=self.drop_prob, force_undirected=True)
        x0 = self.proj(x)
        h = x0

        for i, conv in enumerate(self.convs):
            raw = conv(h, x0, edge_index)
            h = raw + h   # identity mapping
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h

class SRR(torch.nn.Module):
    """
    SRR maintains a small reservoir of subgraph feature snapshots and injects
    them stochastically to re‑introduce high‑frequency signals.
    """
    def __init__(self, in_c, hid_c, out_c, num_layers, reservoir_size: int = 32, mix_alpha: float = 0.5):
        super().__init__()
        self.mix_alpha = mix_alpha
        self.reservoir_size = reservoir_size
        self.reservoir = []  # list of (node_idx tensor, feature tensor) tuples
        self.proj = torch.nn.Linear(in_c, hid_c)
        self.convs = torch.nn.ModuleList()
        layer_idx = 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
        layer_idx += 1
        for _ in range(num_layers - 2):
            self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))
            layer_idx += 1
        self.convs.append(GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=layer_idx))

    def _store_snapshot(self, node_idx, feat):
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append((node_idx.clone(), feat.detach().cpu()))
        else:
            # reservoir sampling replacement
            j = random.randint(0, len(self.reservoir) - 1)
            self.reservoir[j] = (node_idx.clone(), feat.detach().cpu())

    def forward(self, x, edge_index):
        x0 = self.proj(x)
        h = x0
        for i, conv in enumerate(self.convs):
            raw = conv(h, x0, edge_index)
            h = raw + h   # identity mapping
            if i != len(self.convs) - 1:
                h = F.relu(h)
            # Mix only after final convolution to match feature dimensions
            if i == len(self.convs) - 1 and self.training and self.reservoir and random.random() < 0.2:
                node_idx, feat_snap = random.choice(self.reservoir)
                feat_snap = feat_snap.to(h.device)
                h[node_idx] = self.mix_alpha * h[node_idx] + (1 - self.mix_alpha) * feat_snap

        # After forward pass, randomly store a new snapshot
        if self.training:
            num_nodes = h.size(0)
            chosen = torch.randperm(num_nodes, device=h.device)[: min(64, num_nodes)]
            self._store_snapshot(chosen.cpu(), h[chosen].detach())

        return h


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
        x0 = self.lins[0](x)  # 🟢 x0도 변환된 hidden dim이어야 함
        x = x0
        for conv in self.convs:
            x = conv(x, x0, edge_index)
            x = F.relu(x)
        x = self.lins[1](x)
        return x




# --- Original ARGC implementation using GCN2Conv ---
class ARGC(torch.nn.Module):
    """
    Adaptive Residual Gate Convolution (ARGC)

    ▸ 각 레이어에서 raw 메시지 ↔ 이전 표시를 게이트로 혼합  
    ▸ τ(temperature)로 gate 안정화  
    ▸ α-skip으로 입력 h0 재주입, oversmoothing 완화  
    ▸ LayerNorm, dropout floor로 학습 안정성 추가
    """
    def __init__(self, in_c, hid_c, out_c, num_layers,
                 dropout=0.3, tau_init=1.0, alpha_skip=0.2):
        super().__init__()

        # 하이퍼파라미터
        self.base_drop = dropout
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.alpha_skip = alpha_skip

        # 입력 투영
        self.proj = nn.Linear(in_c, hid_c)

        # GCN2Conv stack
        self.convs = nn.ModuleList([
            GCN2Conv(hid_c, alpha=0.1, theta=0.5, layer=i + 1)
            for i in range(num_layers)
        ])

        # Adaptive scalar gate (shared)
        self.gate = nn.Sequential(
            nn.Linear(hid_c * 2, hid_c // 2),
            nn.ReLU(),
            nn.Linear(hid_c // 2, 1)
        )
        nn.init.constant_(self.gate[-1].bias, 5.0)

        # 정규화
        self.norm_z = nn.LayerNorm(hid_c * 2)
        self.norm_post = nn.LayerNorm(hid_c)

        # 출력
        self.out = nn.Linear(hid_c, out_c)

    def forward(self, x, edge_index):
        h0 = self.proj(x)
        h = h0
        L = len(self.convs)

        for layer_idx, conv in enumerate(self.convs):
            # raw computation with GCN2Conv
            raw = conv(h, h0, edge_index)

            # gate input normalization
            z = torch.cat([raw, h], dim=-1)
            z = self.norm_z(z)

            # scalar gate
            gate = torch.sigmoid(self.gate(z) / self.tau)  # shape [N,1]

            # adaptive residual mix with input skip
            h = gate * raw + (1 - gate) * (h + self.alpha_skip * h0)

            # post-norm + dropout floor + activation
            h = self.norm_post(h)
            drop_p = max(0.1, self.base_drop * (1 - layer_idx / L))
            h = F.dropout(h, p=drop_p, training=self.training)
            h = F.relu(h)

        return self.out(h)
