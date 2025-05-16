import random
import numpy as np
from models import GLF, GDN, SRP, ASA, SRR, GCNII, GS, GCN, ARGC, ARGC_GCN
from torch_geometric.nn import GraphNorm
random.seed(42)
np.random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
import matplotlib.pyplot as plt
import os
os.makedirs("logs", exist_ok=True)
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from datetime import datetime

# --- Jacobian spectral norm function ---
import torch.autograd.functional as AF

def compute_jacobian_spectral_norm(model, x, edge_index):
    x = x.clone().detach().requires_grad_(True)
    def model_output(inp):
        return model(inp, edge_index)[0]  # First node's output
    jac = AF.jacobian(model_output, x)
    jac = jac.view(jac.size(0), -1)
    spectral_norm = torch.linalg.norm(jac, ord=2).item()
    return spectral_norm


# --- Adaptive Jacobian Range Regularization (AJR) loss ---
# Planetoid Cora dataset
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]

# device 설정 (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader (전체 그래프 1개이므로 이렇게 설정)
from torch_geometric.loader import DataLoader
train_loader = DataLoader([data], batch_size=1)
val_loader = DataLoader([data], batch_size=1)
# 학습 및 평가 함수 정의
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_mean_cosine_similarity(embeddings, mask=None):
    """
    주어진 임베딩 간 평균 cosine similarity 계산
    - embeddings: [N, D]
    - mask: 계산에 포함할 노드 인덱스 (예: val_mask)
    """
    if mask is not None:
        embeddings = embeddings[mask]
    sim_matrix = cosine_similarity(embeddings.cpu().numpy())
    upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    return upper_triangle.mean()

def train_and_evaluate(model, device, train_loader, val_loader,
                       epochs=100, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    all_embeddings = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            if isinstance(model, GDN):
                h0 = model.proj(batch.x)
                h = h0
                for k, conv in enumerate(model.convs):
                    Δ = model.deltas[k]
                    h_q = model.quantize(h, Δ)
                    eps = torch.randn_like(h_q) * 0.01
                    h = conv(h_q, h0, batch.edge_index) + eps
                    h = F.relu(h)
                    h = F.dropout(h, p=model.dropout, training=model.training)
                embeddings = h
                out = model.out_head(torch.cat([h0, h], dim=-1))
            else:
                out = model(batch.x, batch.edge_index)
                embeddings = out if not isinstance(model, GS) else out
            preds = out.argmax(dim=1)
            correct += (preds[batch.val_mask] == batch.y[batch.val_mask]).sum().item()
            total += batch.val_mask.sum().item()
    acc = correct / total
    sim = compute_mean_cosine_similarity(embeddings, batch.val_mask)
    return acc, sim

models = [
    ("GCNII", GCNII),
    ("GCN", GCN),
    # ("GLF", GLF),
    # ("GLF_GCN", GLF_GCN),
    # ("SRP", SRP),
    # ("SRP_GCN", SRP_GCN),
    # ("ASA", ASA),
    # ("ARGC", ARGC),
    ("ARGC_GCN", ARGC_GCN),
    # ("ASA_GCN", ASA_GCN),
    # ("SRR", lambda in_c, hid_c, out_c, num_layers: SRR(in_c, hid_c, out_c, num_layers, reservoir_size=32, mix_alpha=0.5)),
    # ("SRR_GCN", lambda in_c, hid_c, out_c, num_layers: SRR_GCN(in_c, hid_c, out_c, num_layers, reservoir_size=32, mix_alpha=0.5)),
]
layer_list = [2, 8, 16,32,64]
results = {name: {"acc": [], "sim": []} for name, _ in models}

for name, ModelClass in models:
    print(f"{name} Evaluation Across Depths")
    for num_layers in layer_list:
        # Handle lambda (for SRR and SRR_GCN) vs. standard class
        if callable(ModelClass) and not isinstance(ModelClass, type):
            # Lambda: expects (in_c, hid_c, out_c, num_layers)
            model = ModelClass(dataset.num_node_features, 64, dataset.num_classes, num_layers)
        else:
            # Most models accept dropout=0.5 except GCNII
            if ModelClass in [GLF, GDN, SRP, ASA, GS]:
                model = ModelClass(dataset.num_node_features, 64, dataset.num_classes, num_layers, dropout=0.5)
            else:
                model = ModelClass(dataset.num_node_features, 64, dataset.num_classes, num_layers)
        acc, sim = train_and_evaluate(model, device, train_loader, val_loader)
        results[name]["acc"].append(acc)
        results[name]["sim"].append(sim)
        print(f"Layers: {num_layers} | Acc: {acc:.4f} | CosSim: {sim:.4f}")
        jacobian_norm = compute_jacobian_spectral_norm(model, data.x.to(device), data.edge_index.to(device))
        print(f"Jacobian Spectral Norm: {jacobian_norm:.4f}")

# Plot Accuracy Across Depths
plt.figure(figsize=(10, 5))
for name in results:
    plt.plot(layer_list, results[name]["acc"], marker='o', label=name)
plt.xlabel("Number of Layers")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Across Depths")
plt.grid(True)
plt.xticks(layer_list)
plt.legend()
plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"logs/accuracy_across_depths_{timestamp}.png")

# Plot Cosine Similarity Across Depths
plt.figure(figsize=(10, 5))
for name in results:
    plt.plot(layer_list, results[name]["sim"], marker='s', label=name)
plt.xlabel("Number of Layers")
plt.ylabel("Cosine Similarity")
plt.title("Model Cosine Similarity Across Depths")
plt.grid(True)
plt.xticks(layer_list)
plt.legend()
plt.tight_layout()

plt.savefig(f"logs/cosine_similarity_across_depths_{timestamp}.png")

print(f"Plots saved: accuracy_across_depths_{timestamp}.png, cosine_similarity_across_depths_{timestamp}.png")
