#!/usr/bin/env python3
"""
Demo script showing the complete ST-GNN pipeline.
This creates synthetic data to demonstrate the workflow when UNSW-NB15 is not available.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models import GCNOnly, STGNN, create_model
from src.data_loader import UNSWNB15Loader


def create_synthetic_data(num_nodes=100, num_edges=500, num_features=30, num_temporal=10):
    """Create synthetic graph data for demonstration."""
    print(f"Creating synthetic data: {num_nodes} nodes, {num_edges} edges, {num_temporal} temporal graphs")

    # Node features
    x = torch.randn(num_nodes, num_features)

    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    # Labels (some nodes are "attack")
    y = torch.zeros(num_nodes, dtype=torch.long)
    attack_nodes = torch.randperm(num_nodes)[:num_nodes // 10]  # 10% attack
    y[attack_nodes] = 1

    # Static graph
    static_graph = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

    # Temporal graphs (slightly varying)
    temporal_graphs = []
    for t in range(num_temporal):
        # Add some noise to features over time
        x_t = x + 0.1 * torch.randn_like(x)
        # Slightly different edges
        n_edges_t = num_edges + torch.randint(-50, 50, (1,)).item()
        n_edges_t = max(100, n_edges_t)
        edge_index_t = torch.randint(0, num_nodes, (2, n_edges_t))
        edge_index_t = torch.cat([edge_index_t, edge_index_t.flip(0)], dim=1)
        mask_t = edge_index_t[0] != edge_index_t[1]
        edge_index_t = edge_index_t[:, mask_t]

        graph_t = Data(x=x_t, edge_index=edge_index_t, y=y, num_nodes=num_nodes)
        temporal_graphs.append(graph_t)

    return static_graph, temporal_graphs


def train_demo():
    """Run a quick training demo with synthetic data."""
    print("="*60)
    print("ST-GNN Demo with Synthetic Data")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create synthetic data
    train_static, train_temporal = create_synthetic_data(num_nodes=200, num_edges=1000, num_temporal=20)
    test_static, test_temporal = create_synthetic_data(num_nodes=150, num_edges=700, num_temporal=10)

    in_channels = train_static.x.shape[1]
    print(f"Input features: {in_channels}")

    config = {
        'hidden_channels': 64,
        'gcn_layers': 2,
        'temporal_layers': 1,
        'num_heads': 2,
        'dropout': 0.3,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'epochs': 30,
        'patience': 10,
        'sequence_length': 3,
        'max_train_sequences': 50
    }

    # Train GCN-only
    print("\n--- Training GCN-only ---")
    gcn_model = create_model('gcn', in_channels, **config)
    gcn_model.to(device)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    train_graph = train_static.to(device)
    test_graph = test_static.to(device)

    for epoch in range(config['epochs']):
        gcn_model.train()
        optimizer.zero_grad()

        output = gcn_model([train_graph], return_node_scores=True)
        loss = criterion(output['graph_logits'], train_graph.y.unsqueeze(0).to(device))
        if 'node_logits' in output:
            loss += 0.5 * criterion(output['node_logits'], train_graph.y.to(device))

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            gcn_model.eval()
            with torch.no_grad():
                test_output = gcn_model([test_graph], return_node_scores=True)
                test_loss = criterion(test_output['graph_logits'], test_graph.y.unsqueeze(0).to(device))
                probs = test_output['graph_probs'].cpu().numpy()
                preds = probs.argmax(axis=1)
                true = test_graph.y.unsqueeze(0).cpu().numpy()
                from sklearn.metrics import f1_score, roc_auc_score
                f1 = f1_score(true, preds)
                auc = roc_auc_score(true, probs[:, 1]) if len(np.unique(true)) > 1 else 0
                print(f"  Epoch {epoch}: Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Train ST-GNN
    print("\n--- Training ST-GNN ---")
    stgnn_model = create_model('stgnn', in_channels, **config)
    stgnn_model.to(device)

    optimizer = torch.optim.Adam(stgnn_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Prepare temporal sequences
    seq_len = config['sequence_length']
    train_seqs = []
    train_labels = []
    for i in range(len(train_temporal) - seq_len + 1):
        train_seqs.append(train_temporal[i:i+seq_len])
        train_labels.append(train_temporal[i+seq_len-1].y)

    test_seqs = []
    test_labels = []
    for i in range(len(test_temporal) - seq_len + 1):
        test_seqs.append(test_temporal[i:i+seq_len])
        test_labels.append(test_temporal[i+seq_len-1].y)

    for epoch in range(config['epochs']):
        stgnn_model.train()
        epoch_losses = []

        for seq, label in zip(train_seqs, train_labels):
            seq = [g.to(device) for g in seq]
            label = label.to(device)

            optimizer.zero_grad()
            output = stgnn_model(seq, return_node_scores=True)
            loss = criterion(output['graph_logits'], label.unsqueeze(0))
            if 'node_logits' in output:
                loss += 0.5 * criterion(output['node_logits'], label)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % 10 == 0:
            stgnn_model.eval()
            with torch.no_grad():
                test_seq = [g.to(device) for g in test_seqs[0]]
                test_label = test_labels[0].to(device)

                test_output = stgnn_model(test_seq, return_node_scores=True)
                test_loss = criterion(test_output['graph_logits'], test_label.unsqueeze(0))
                probs = test_output['graph_probs'].cpu().numpy()
                preds = probs.argmax(axis=1)
                true = test_label.unsqueeze(0).cpu().numpy()
                from sklearn.metrics import f1_score, roc_auc_score
                f1 = f1_score(true, preds)
                auc = roc_auc_score(true, probs[:, 1]) if len(np.unique(true)) > 1 else 0
                print(f"  Epoch {epoch}: Loss={np.mean(epoch_losses):.4f}, Test Loss={test_loss.item():.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    print("\n--- Final Evaluation ---")
    gcn_model.eval()
    stgnn_model.eval()

    with torch.no_grad():
        # GCN
        gcn_test = gcn_model([test_graph], return_node_scores=True)
        gcn_probs = gcn_test['graph_probs'].cpu().numpy()
        gcn_preds = gcn_probs.argmax(axis=1)
        gcn_true = test_graph.y.unsqueeze(0).cpu().numpy()

        # ST-GNN
        stgnn_test = stgnn_model([g.to(device) for g in test_seqs[0]], return_node_scores=True)
        stgnn_probs = stgnn_test['graph_probs'].cpu().numpy()
        stgnn_preds = stgnn_probs.argmax(axis=1)
        stgnn_true = test_labels[0].unsqueeze(0).cpu().numpy()

    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

    print(f"\n{'Metric':<20} {'GCN-only':>12} {'ST-GNN':>12} {'Improvement':>12}")
    print("-" * 60)
    for name, (gcn_val, stgnn_val) in [
        ('AUC-ROC', (roc_auc_score(gcn_true, gcn_probs[:, 1]) if len(np.unique(gcn_true)) > 1 else 0,
                     roc_auc_score(stgnn_true, stgnn_probs[:, 1]) if len(np.unique(stgnn_true)) > 1 else 0)),
        ('F1-Score', (f1_score(gcn_true, gcn_preds), f1_score(stgnn_true, stgnn_preds))),
        ('Precision', (precision_score(gcn_true, gcn_preds), precision_score(stgnn_true, stgnn_preds))),
        ('Recall', (recall_score(gcn_true, gcn_preds), recall_score(stgnn_true, stgnn_preds))),
    ]:
        imp = ((stgnn_val - gcn_val) / gcn_val * 100) if gcn_val > 0 else 0
        print(f"{name:<20} {gcn_val:>12.4f} {stgnn_val:>12.4f} {imp:>11.1f}%")

    print("\nDemo complete!")


if __name__ == '__main__':
    train_demo()