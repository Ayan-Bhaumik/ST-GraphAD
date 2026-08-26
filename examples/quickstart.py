#!/usr/bin/env python3
"""
Quickstart example for ST-GraphAD.
Shows how to use the library programmatically.
"""

import torch
from src import load_unsw_nb15, create_model, Trainer, visualize_model_predictions


def main():
    print("ST-GraphAD Quickstart")
    print("="*50)

    # 1. Load data
    print("\n1. Loading UNSW-NB15 dataset...")
    data = load_unsw_nb15(data_dir='data', download=False)

    print(f"   Train graph: {data['train_static'].num_nodes} nodes, {data['train_static'].num_edges} edges")
    print(f"   Test graph: {data['test_static'].num_nodes} nodes, {data['test_static'].num_edges} edges")
    print(f"   Node features: {data['num_node_features']}")

    # 2. Create models
    print("\n2. Creating models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gcn_model = create_model(
        'gcn',
        in_channels=data['num_node_features'],
        hidden_channels=128,
        gcn_layers=3,
        dropout=0.5
    ).to(device)

    stgnn_model = create_model(
        'stgnn',
        in_channels=data['num_node_features'],
        hidden_channels=128,
        gcn_layers=3,
        temporal_layers=2,
        num_heads=4,
        dropout=0.5,
        use_temporal=True
    ).to(device)

    print(f"   GCN parameters: {sum(p.numel() for p in gcn_model.parameters()):,}")
    print(f"   ST-GNN parameters: {sum(p.numel() for p in stgnn_model.parameters()):,}")

    # 3. Quick inference test (without training)
    print("\n3. Testing forward pass...")
    gcn_model.eval()
    stgnn_model.eval()

    with torch.no_grad():
        # GCN on static graph
        train_graph = data['train_static'].to(device)
        gcn_output = gcn_model(train_graph, return_node_scores=True)
        print(f"   GCN graph probs: {gcn_output['graph_probs'].cpu().numpy()}")

        # ST-GNN on temporal sequence
        test_seq = [g.to(device) for g in data['test_temporal'][:3]]
        stgnn_output = stgnn_model(test_seq, return_node_scores=True)
        print(f"   ST-GNN graph probs: {stgnn_output['graph_probs'].cpu().numpy()}")

    # 4. Training example (commented out for quickstart)
    print("\n4. Training (example code):")
    print("""
    trainer = Trainer(gcn_model, device, lr=0.001)
    for epoch in range(100):
        loss = trainer.train_epoch([train_graph], train_graph.y.unsqueeze(0))
        metrics = trainer.evaluate([test_graph], test_graph.y.unsqueeze(0))
        print(f"Epoch {epoch}: Loss={loss:.4f}, AUC={metrics['graph_auc']:.4f}")
    """)

    print("\n5. Run full training with:")
    print("   python main.py --model both --download")

    print("\nQuickstart complete!")


if __name__ == '__main__':
    main()