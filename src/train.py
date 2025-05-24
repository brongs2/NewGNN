import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from models import GCN, GCNII, ARGC

def train_and_eval(model, data, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # evaluation
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum().item()
    acc = correct / int(data.val_mask.sum())
    print(f"Validation Accuracy: {acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="GNN 실험")
    parser.add_argument("--dataset", type=str, default="Cora",
                        choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--model", type=str, default="ARGC",
                        choices=["GCN", "GCNII", "ARGC"])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dataset = Planetoid(root=f'./data/{args.dataset}', name=args.dataset)
    data = dataset[0]
    ModelClass = {"GCN": GCN, "GCNII": GCNII, "ARGC": ARGC}[args.model]
    model = ModelClass(dataset.num_node_features, 64, dataset.num_classes, args.layers)

    train_and_eval(model, data, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()