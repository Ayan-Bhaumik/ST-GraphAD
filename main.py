#!/usr/bin/env python3
"""
Main entry point for ST-GNN Network Intrusion Detection on UNSW-NB15.

Usage:
    python main.py --model gcn        # Train GCN-only baseline
    python main.py --model stgnn      # Train ST-GNN (GCN + Temporal Attention)
    python main.py --model both       # Train and compare both models
    python main.py --visualize        # Generate visualizations from saved models
"""

import argparse
import os
import torch
import numpy as np
from src.data_loader import load_unsw_nb15
from src.models import create_model, GCNOnly, STGNN
from src.train import train_model, compare_models, save_results, print_comparison_table
from src.visualize import visualize_model_predictions, generate_report, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='ST-GNN for Network Intrusion Detection')
    parser.add_argument('--model', type=str, choices=['gcn', 'stgnn', 'both'], default='both',
                        help='Model to train')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden channels')
    parser.add_argument('--layers', type=int, default=3, help='GCN layers')
    parser.add_argument('--temporal-layers', type=int, default=2, help='Temporal attention layers')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--seq-len', type=int, default=5, help='Sequence length for temporal model')
    parser.add_argument('--max-train', type=int, default=200, help='Max training sequences')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--load-model', type=str, help='Load saved model path')
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def get_config(args) -> dict:
    return {
        'hidden_channels': args.hidden,
        'gcn_layers': args.layers,
        'temporal_layers': args.temporal_layers,
        'num_heads': args.heads,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'sequence_length': args.seq_len,
        'max_train_sequences': args.max_train
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    config = get_config(args)

    print("="*60)
    print("ST-GNN Network Intrusion Detection - UNSW-NB15")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Config: {config}")

    # Load data
    print("\nLoading dataset...")
    data = load_unsw_nb15(data_dir=args.data_dir, download=args.download)

    print(f"Train graph: {data['train_static'].num_nodes} nodes, {data['train_static'].num_edges} edges")
    print(f"Test graph: {data['test_static'].num_nodes} nodes, {data['test_static'].num_edges} edges")
    print(f"Temporal graphs - Train: {len(data['train_temporal'])}, Test: {len(data['test_temporal'])}")
    print(f"Node features: {data['num_node_features']}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    if args.model == 'both':
        # Compare both models
        results, gcn_model, stgnn_model = compare_models(data, device, config)

        print_comparison_table(results)
        save_results(results)
        plot_training_curves(results)
        generate_report(results, config)

        # Save models
        torch.save(gcn_model.state_dict(), 'models/gcn_model.pt')
        torch.save(stgnn_model.state_dict(), 'models/stgnn_model.pt')
        print("\nModels saved to models/")

    elif args.model == 'gcn':
        model, result = train_model('gcn', data, config, device)
        print_comparison_table({'gcn': result})
        save_results({'gcn': result})
        torch.save(model.state_dict(), 'models/gcn_model.pt')
        print("\nGCN model saved to models/gcn_model.pt")

    elif args.model == 'stgnn':
        model, result = train_model('stgnn', data, config, device)
        print_comparison_table({'stgnn': result})
        save_results({'stgnn': result})
        torch.save(model.state_dict(), 'models/stgnn_model.pt')
        print("\nST-GNN model saved to models/stgnn_model.pt")

    if args.visualize:
        print("\nGenerating visualizations...")
        if args.model in ['both', 'gcn']:
            gcn_model = create_model('gcn', data['num_node_features'],
                                    hidden_channels=config['hidden_channels'],
                                    gcn_layers=config['gcn_layers'],
                                    dropout=config['dropout'])
            gcn_model.load_state_dict(torch.load('models/gcn_model.pt', map_location=device))
            visualize_model_predictions(gcn_model, data, device, 'results/predictions_gcn')

        if args.model in ['both', 'stgnn']:
            stgnn_model = create_model('stgnn', data['num_node_features'],
                                      hidden_channels=config['hidden_channels'],
                                      gcn_layers=config['gcn_layers'],
                                      temporal_layers=config['temporal_layers'],
                                      num_heads=config['num_heads'],
                                      dropout=config['dropout'],
                                      use_temporal=True)
            stgnn_model.load_state_dict(torch.load('models/stgnn_model.pt', map_location=device))
            visualize_model_predictions(stgnn_model, data, device, 'results/predictions_stgnn')

    print("\nDone!")


if __name__ == '__main__':
    main()