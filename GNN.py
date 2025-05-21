import random
import numpy as np
from models import  GCNII, GCN, ARGC
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

# Specify dataset: choose 'Cora', 'CiteSeer', or 'PubMed'
DATASET_NAME = 'Cora'  # Change this to 'Cora' or 'PubMed' as needed

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
# Planetoid dataset (selectable)
dataset = Planetoid(root=f'./data/{DATASET_NAME}', name=DATASET_NAME)
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

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            # Special handling for GDN

            out = model(batch.x, batch.edge_index)
            embeddings = out  # GS도 out 반환물이 embeddings 역할

            preds = out.argmax(dim=1)
            correct += (preds[batch.val_mask] == batch.y[batch.val_mask]).sum().item()
            total += batch.val_mask.sum().item()

    acc = correct / total
    sim = compute_mean_cosine_similarity(embeddings, batch.val_mask)

    # Dirichlet energy: 1/2 Σ_{(i,j)∈E} ||h_i - h_j||^2
    row, col = batch.edge_index
    diff = embeddings[row] - embeddings[col]
    sq = diff.pow(2).sum(dim=1)
    energy = float(sq.sum().item() / 2)

    return acc, sim, energy

models = [
    ("GCN", GCN),
    ("GCNII", GCNII),
    # ("GLF", GLF),
    # ("GLF_GCN", GLF_GCN),
    # ("SRP", SRP),
    # ("SRP_GCN", SRR_GCN),
    # ("ASA", ASA),
    ("ARGC", ARGC),
    # ("ASA_GCN", ASA_GCN),
    # ("SRR", lambda in_c, hid_c, out_c, num_layers: SRR(in_c, hid_c, out_c, num_layers, reservoir_size=32, mix_alpha=0.5)),
    # ("SRR_GCN", lambda in_c, hid_c, out_c, num_layers: SRR_GCN(in_c, hid_c, out_c, num_layers, reservoir_size=32, mix_alpha=0.5)),
]

layer_list = [2, 8, 16, 32, 64]
results = {name: {"acc": [], "sim": [], "energy": []} for name, _ in models}

for name, ModelClass in models:
    print(f"{name} Evaluation Across Depths")
    for num_layers in layer_list:
        # Instantiate model
        if callable(ModelClass) and not isinstance(ModelClass, type):
            model = ModelClass(dataset.num_node_features, 64, dataset.num_classes, num_layers)
        else:
            model = ModelClass(dataset.num_node_features, 64, dataset.num_classes, num_layers)
        # Train and evaluate
        acc, sim, energy = train_and_evaluate(model, device, train_loader, val_loader)
        # Save metrics
        results[name]["acc"].append(acc)
        results[name]["sim"].append(sim)
        results[name]["energy"].append(energy)
        # Print metrics
        print(f"Layers: {num_layers} | Acc: {acc:.4f} | CosSim: {sim:.4f}")
        jacobian_norm = compute_jacobian_spectral_norm(model, data.x.to(device), data.edge_index.to(device))
        print(f"Jacobian Spectral Norm: {jacobian_norm:.4f}")
        print(f"Dirichlet Energy: {energy:.4f}")

# ---------------- ARGC 파라미터 그리드 실험 ----------------
# 실험할 파라미터 값들
# tau_grid     = [0.5, 1.0, 2.0]   # gate temperature
# alpha_grid   = [0.2]   # skip 비율
# # dropout_grid = [0.1, 0.3, 0.5]   # base dropout
# dropout_grid = [0.3]

# layer_list = [2, 8, 16, 32, 64]

# 결과 저장 dict  (key: config 이름)
# results = {}

# for tau in tau_grid:
#     for alpha in alpha_grid:
#         for drop in dropout_grid:
#             cfg_name = f"ARGC_tau{tau}_alpha{alpha}_drop{drop}"
#             results[cfg_name] = {"acc": [], "sim": [], "energy": []}

#             print(f"\n{cfg_name} Evaluation Across Depths")
#             for num_layers in layer_list:
#                 # ARGC 인스턴스 생성
#                 model = ARGC(
#                     dataset.num_node_features, 64, dataset.num_classes,
#                     num_layers,
#                     dropout=drop,
#                     tau_init=tau,
#                     alpha_skip=alpha
#                 )

#                 # 학습 & 평가
#                 acc, sim, energy = train_and_evaluate(
#                     model, device, train_loader, val_loader
#                 )

#                 # 결과 저장
#                 results[cfg_name]["acc"].append(acc)
#                 results[cfg_name]["sim"].append(sim)
#                 results[cfg_name]["energy"].append(energy)

#                 # 콘솔 출력
#                 print(f"Layers: {num_layers:>2} | "
#                       f"Acc: {acc:.4f} | CosSim: {sim:.4f} | Energy: {energy:.1f}")

#                 jac_norm = compute_jacobian_spectral_norm(
#                     model, data.x.to(device), data.edge_index.to(device)
#                 )
#                 print(f"  Jacobian Spectral Norm: {jac_norm:.4f}")
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

# Plot Dirichlet Energy Across Depths
plt.figure(figsize=(10, 5))
for name in results:
    plt.plot(layer_list, results[name]["energy"], marker='^', label=name)
plt.xlabel("Number of Layers")
plt.ylabel("Dirichlet Energy")
plt.title("Model Dirichlet Energy Across Depths")
plt.grid(True)
plt.xticks(layer_list)
plt.legend()
plt.tight_layout()
plt.savefig(f"logs/dirichlet_energy_across_depths_{timestamp}.png")

print(f"Plots saved: accuracy_across_depths_{timestamp}.png, cosine_similarity_across_depths_{timestamp}.png, dirichlet_energy_across_depths_{timestamp}.png")
