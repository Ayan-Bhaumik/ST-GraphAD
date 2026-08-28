#!/usr/bin/env python3
"""
Main entry point for ST-GraphAD Network Intrusion Detection on UNSW-NB15.

Usage:
    python main.py --model gcn        # Train GCN-only baseline
    python main.py --model stgnn      # Train ST-GNN (GCN + Temporal Attention)
    python main.py --model both       # Train and compare both models
    python main.py --seeds 1 2 3 4 5  # Multi-seed evaluation (5 seeds)
    python main.py --visualize        # Generate visualizations from saved models
"""

import argparse
import os
import torch
import numpy as np
from src.data_loader import load_unsw_nb15
from src.models import create_model, GCNOnly, STGNN
from src.train import (train_model, compare_models, save_results,
                       print_comparison_table, plot_training_curves,
                       multi_seed_evaluation)
from src.visualize import visualize_model_predictions, generate_report


def parse_args():
    parser = argparse.ArgumentParser(description='ST-GraphAD for Network Intrusion Detection')
    parser.add_argument('--model', type=str, choices=['gcn', 'stgnn', 'both'], default='both',
                        help='Model to train')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--download', action='store_true', help='Download dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden channels')
    parser.add_argument('--layers', type=int, default=3, help='GCN layers')
    parser.add_argument('--temporal-layers', type=int, default=2, help='Temporal attention layers')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--seq-len', type=int, default=5, help='Sequence length for temporal model')
    parser.add_argument('--max-train', type=int, default=200, help='Max training sequences')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds for multi-seed evaluation')
    parser.add_argument('--multi-seed', action='store_true',
                        help='Run multi-seed evaluation (requires --seeds)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--load-model', type=str, help='Load saved model path')
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
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

    print("=" * 60)
    print("ST-GraphAD Network Intrusion Detection - UNSW-NB15")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Config: {config}")
    if args.multi_seed:
        print(f"Seeds: {args.seeds}")

    # Load data
    print("\nLoading dataset...")
    data = load_unsw_nb15(data_dir=args.data_dir, download=args.download)

    print(f"\n=== Graph Statistics ===")
    print(f"Train temporal graphs: {len(data['train_temporal'])}")
    print(f"Val temporal graphs: {len(data['val_temporal'])}")
    print(f"Test temporal graphs: {len(data['test_temporal'])}")
    print(f"Train static: {data['train_static'].num_nodes} nodes, {data['train_static'].num_edges} edges")
    print(f"Val static: {data['val_static'].num_nodes} nodes, {data['val_static'].num_edges} edges")
    print(f"Test static: {data['test_static'].num_nodes} nodes, {data['test_static'].num_edges} edges")
    print(f"Node features: {data['num_node_features']}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Sanity checks
    print("\n=== Sanity Checks ===")
    train_idxs = set(data['train_df'].index)
    val_idxs = set(data['val_df'].index)
    test_idxs = set(data['test_df'].index)
    print(f"Train/Val disjoint: {len(train_idxs & val_idxs) == 0}")
    print(f"Train/Test disjoint: {len(train_idxs & test_idxs) == 0}")
    print(f"Val/Test disjoint: {len(val_idxs & test_idxs) == 0}")
    print(f"Val labels have both classes: {len(data['val_temporal'][0].y.unique()) == 2 if len(data['val_temporal']) > 0 else False}")

    if args.multi_seed:
        # Multi-seed evaluation
        print("\n" + "=" * 60)
        print("MULTI-SEED EVALUATION")
        print("=" * 60)
        all_results, aggregate = multi_seed_evaluation(data, device, config, seeds=args.seeds)

        # Save models from last seed
        for model_type in ['gcn', 'stgnn']:
            if model_type in all_results[args.seeds[-1]]:
                torch.save(all_results[args.seeds[-1]][model_type]['model_state'],
                           f'models/{model_type}_model.pt')
                print(f"{model_type.upper()} model saved to models/{model_type}_model.pt")

    elif args.model == 'both':
        # Compare both models (single seed)
        results, gcn_model, stgnn_model = compare_models(data, device, config, seeds=args.seeds)

        print_comparison_table(results, seeds=args.seeds)
        save_results(results, seeds=args.seeds)
        plot_training_curves(results)
        generate_report(results, config)

        # Save models
        for model_type in ['gcn', 'stgnn']:
            if model_type in results:
                torch.save(results[model_type]['model_state'], f'models/{model_type}_model.pt')
                print(f"\n{model_type.upper()} model saved to models/{model_type}_model.pt")

    elif args.model == 'gcn':
        result = train_model('gcn', data, config, device, seed=args.seeds[0])
        print_comparison_table({'gcn': result}, seeds=args.seeds)
        save_results({'gcn': result}, seeds=args.seeds)
        torch.save(result['model_state'], 'models/gcn_model.pt')
        print("\nGCN model saved to models/gcn_model.pt")

    elif args.model == 'stgnn':
        result = train_model('stgnn', data, config, device, seed=args.seeds[0])
        print_comparison_table({'stgnn': result}, seeds=args.seeds)
        save_results({'stgnn': result}, seeds=args.seeds)
        torch.save(result['model_state'], 'models/stgnn_model.pt')
        print("\nST-GNN model saved to models/stgnn_model.pt")

    if args.visualize:
        print("\nGenerating visualizations...")
        try:
            visualize_model_predictions(data['test_static'], device)
        except Exception as e:
            print(f"Visualization error: {e}")


if __name__ == '__main__':
    main()