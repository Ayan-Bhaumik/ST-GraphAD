"""
Training and evaluation pipeline for ST-GNN on UNSW-NB15.
Fixes: node-level validation, proper train/val/test split, multi-seed, per-attack-category eval.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.models import create_model


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Trainer for spatio-temporal GNN models with node-level validation."""

    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 1e-3, weight_decay: float = 5e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_node_aucs = []
        self.val_node_f1s = []
        self.val_node_precisions = []
        self.val_node_recalls = []

    def train_epoch(self, train_graphs: List[Data], labels: torch.Tensor,
                    node_labels: torch.Tensor) -> float:
        """Train for one epoch on node-level labels."""
        self.model.train()
        self.optimizer.zero_grad()

        # Pass the entire sequence to the model; it handles temporal internally
        output = self.model(train_graphs, return_node_scores=True)
        loss = self.criterion(output['node_logits'], node_labels.to(self.device))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def evaluate_node_level(self, seqs: List[List[Data]], labels_list: List[torch.Tensor]) -> Dict:
        """
        Evaluate node-level detection for sequences.
        seqs: list of sequences, where each sequence is a list of Data objects
        labels_list: list of tensors, one per sequence (node labels for the last graph in each sequence)
        """
        self.model.eval()

        all_preds = []
        all_probs = []
        all_true = []

        with torch.no_grad():
            for seq, labels in zip(seqs, labels_list):
                # Forward pass on the entire sequence
                output = self.model(seq, return_node_scores=True)
                probs = output['node_probs'].cpu().numpy()
                preds = probs.argmax(axis=1)
                y_true = labels.cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs[:, 1])
                all_true.extend(y_true)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)

        # AUC: unavailable if only one class present
        if len(np.unique(all_true)) < 2:
            auc = None
        else:
            auc = roc_auc_score(all_true, all_probs)

        f1 = f1_score(all_true, all_preds, average='binary', zero_division=0)
        precision = precision_score(all_true, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_true, all_preds, average='binary', zero_division=0)

        return {
            'node_auc': auc,
            'node_f1': f1,
            'node_precision': precision,
            'node_recall': recall,
            'node_preds': all_preds,
            'node_probs': all_probs,
            'node_true': all_true,
            'confusion_matrix': confusion_matrix(all_true, all_preds).tolist(),
            'n_positive': int(np.sum(all_true == 1)),
            'n_negative': int(np.sum(all_true == 0))
        }


def prepare_temporal_data(train_graphs: List[Data], val_graphs: List[Data], test_graphs: List[Data],
                          sequence_length: int = 5) -> Tuple:
    """Prepare sliding window sequences for temporal training. Returns sequences + per-node labels."""
    def create_sequences(graphs: List[Data], seq_len: int):
        seqs, labels = [], []
        for i in range(len(graphs) - seq_len + 1):
            seq = graphs[i:i + seq_len]
            seqs.append(seq)
            labels.append(seq[-1].y)  # node-level labels
        return seqs, labels

    train_seqs, train_labels = create_sequences(train_graphs, sequence_length)
    val_seqs, val_labels = create_sequences(val_graphs, sequence_length)
    test_seqs, test_labels = create_sequences(test_graphs, sequence_length)

    return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels


def train_model(model_type: str, data: Dict, config: Dict, device: torch.device,
                seed: int = 42) -> Dict:
    """Train a model with given configuration. Uses node-level validation."""
    set_seed(seed)
    in_channels = data['num_node_features']
    seq_len = config.get('sequence_length', 5)

    # Create model
    model = create_model(model_type, in_channels,
                        hidden_channels=config['hidden_channels'],
                        gcn_layers=config['gcn_layers'],
                        temporal_layers=config.get('temporal_layers', 2),
                        num_heads=config.get('num_heads', 4),
                        dropout=config['dropout'],
                        use_temporal=(model_type == 'stgnn'))

    trainer = Trainer(model, device, lr=config['lr'], weight_decay=config['weight_decay'])

    # Prepare sequences (same for both models)
    train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = prepare_temporal_data(
        data['train_temporal'], data['val_temporal'], data['test_temporal'], seq_len
    )

    # Limit training sequences
    max_train = config.get('max_train_sequences', 200)
    if len(train_seqs) > max_train:
        idx = np.random.choice(len(train_seqs), max_train, replace=False)
        train_seqs = [train_seqs[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    # Move labels to device individually (different sizes)
    train_labels_t = [lbl.to(device) for lbl in train_labels]
    val_labels_t = [lbl.to(device) for lbl in val_labels]
    test_labels_t = [lbl.to(device) for lbl in test_labels]

    # Print sequence counts
    print(f"\n=== {model_type.upper()} Sequence Counts ===")
    print(f"Train snapshots: {len(data['train_temporal'])} → Train sequences: {len(train_seqs)}")
    print(f"Val snapshots: {len(data['val_temporal'])} → Val sequences: {len(val_seqs)}")
    print(f"Test snapshots: {len(data['test_temporal'])} → Test sequences: {len(test_seqs)}")

    # Training loop
    best_val_auc = -1.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    max_patience = config.get('patience', 20)
    epochs = config.get('epochs', 200)

    start_time = time.time()

    print(f"\nTraining {model_type.upper()} model (seed={seed})...")
    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}, Test sequences: {len(test_seqs)}")

    for epoch in range(epochs):
        epoch_losses = []
        for seq, label in zip(train_seqs, train_labels_t):
            loss = trainer.train_epoch(seq, label, label)  # node-level
            epoch_losses.append(loss)
        avg_loss = np.mean(epoch_losses)
        trainer.train_losses.append(avg_loss)

        # Node-level validation: use all val sequences
        if val_seqs:
            val_metrics = trainer.evaluate_node_level(val_seqs, val_labels_t)
        else:
            val_metrics = {
                'node_auc': None, 'node_f1': 0.0, 'node_precision': 0.0, 'node_recall': 0.0,
                'node_preds': np.array([]), 'node_probs': np.array([]), 'node_true': np.array([]),
                'confusion_matrix': [[0, 0], [0, 0]], 'n_positive': 0, 'n_negative': 0
            }
        val_node_auc = val_metrics['node_auc'] if val_metrics['node_auc'] is not None else float('nan')

        trainer.val_node_aucs.append(val_node_auc)
        trainer.val_node_f1s.append(val_metrics['node_f1'])
        trainer.val_node_precisions.append(val_metrics['node_precision'])
        trainer.val_node_recalls.append(val_metrics['node_recall'])

        trainer.scheduler.step(val_node_auc if not np.isnan(val_node_auc) else 0.0)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Val Node AUC: {val_node_auc:.4f} | "
                  f"Val Node F1: {val_metrics['node_f1']:.4f} | "
                  f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping on validation node AUC
        if not np.isnan(val_node_auc) and val_node_auc > best_val_auc:
            best_val_auc = val_node_auc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    runtime = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test evaluation: use all test sequences
    if test_seqs:
        test_metrics = trainer.evaluate_node_level(test_seqs, test_labels_t)
    else:
        test_metrics = {
            'node_auc': None, 'node_f1': 0.0, 'node_precision': 0.0, 'node_recall': 0.0,
            'node_preds': np.array([]), 'node_probs': np.array([]), 'node_true': np.array([]),
            'confusion_matrix': [[0, 0], [0, 0]], 'n_positive': 0, 'n_negative': 0
        }

    # Per-attack-category evaluation
    cat_results = evaluate_per_category(model, data['test_temporal'], test_seqs, device)

    return {
        'model_type': model_type,
        'seed': seed,
        'train_losses': trainer.train_losses,
        'val_node_aucs': trainer.val_node_aucs,
        'val_node_f1s': trainer.val_node_f1s,
        'val_node_precisions': trainer.val_node_precisions,
        'val_node_recalls': trainer.val_node_recalls,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'category_results': cat_results,
        'config': config,
        'runtime': runtime,
        'model_state': best_model_state
    }


def evaluate_per_category(model, test_temporal_graphs, test_seqs, device) -> Dict:
    """
    Evaluate node-level detection per attack category.
    NOTE: The current node-label construction aggregates flows into nodes by IP-proto-state tuples,
    so node-level labels are binary (attack/normal), not per-category. Per-category evaluation
    requires storing category info per node in the graph construction step, which is not currently
    available at node granularity. We report this limitation explicitly.
    """
    return {
        'note': 'Per-category node evaluation not available: node labels are binary aggregated (attack/normal). '
                'attack_cat is available at flow level, but node aggregation loses per-category assignment.',
        'available': False
    }


def compare_models(data: Dict, device: torch.device, config: Dict, seeds: List[int] = [42]) -> Tuple[Dict, Optional, Optional]:
    """Train and compare GCN and ST-GNN. Resilient to partial failures."""
    results = {}

    print("\n" + "=" * 60)
    print("ST-GraphAD Model Comparison")
    print("=" * 60)

    # Train GCN
    try:
        gcn_run = train_model('gcn', data, config, device, seed=seeds[0])
        results['gcn'] = gcn_run
        print("✓ GCN training completed")
    except Exception as e:
        print(f"✗ GCN training failed: {e}")
        import traceback
        traceback.print_exc()

    # Train ST-GNN
    try:
        stgnn_run = train_model('stgnn', data, config, device, seed=seeds[0])
        results['stgnn'] = stgnn_run
        print("✓ ST-GNN training completed")
    except Exception as e:
        print(f"✗ ST-GNN training failed: {e}")
        import traceback
        traceback.print_exc()

    gcn_model = None
    stgnn_model = None

    return results, gcn_model, stgnn_model


def print_comparison_table(results: Dict, seeds: List[int] = [42]):
    """Print comparison table. Robust to missing models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)

    if 'gcn' not in results and 'stgnn' not in results:
        print("ERROR: No results available.")
        return

    for model_type in ['gcn', 'stgnn']:
        if model_type not in results:
            print(f"\n{model_type.upper()}: NOT COMPLETED")
            continue

        r = results[model_type]
        tm = r['test_metrics']
        print(f"\n{model_type.upper()} (seed={r['seed']}):")
        print(f"  Test Node AUC-ROC: {tm['node_auc']:.4f}" if tm['node_auc'] is not None else "  Test Node AUC-ROC: N/A")
        print(f"  Test Node F1:       {tm['node_f1']:.4f}")
        print(f"  Test Node Precision: {tm['node_precision']:.4f}")
        print(f"  Test Node Recall:    {tm['node_recall']:.4f}")
        print(f"  Confusion Matrix:    {tm['confusion_matrix']}")
        print(f"  Positive nodes: {tm['n_positive']}, Negative nodes: {tm['n_negative']}")
        print(f"  Best val epoch: {r['best_epoch']}, Best val AUC: {r['best_val_auc']:.4f}")
        print(f"  Runtime: {r['runtime']:.1f}s")

    if 'gcn' in results and 'stgnn' in results:
        gcn_auc = results['gcn']['test_metrics']['node_auc']
        stgnn_auc = results['stgnn']['test_metrics']['node_auc']
        if gcn_auc is not None and stgnn_auc is not None:
            delta = (stgnn_auc - gcn_auc) * 100
            print(f"\nST-GNN vs GCN ΔAUC: {delta:+.2f}%")


def save_results(results: Dict, path: str = 'results/evaluation_results.json', seeds: List[int] = [42]):
    """Save results to structured JSON. Preserves all keys."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(v) for v in obj)
        else:
            return obj

    serializable = {}
    for model_type, r in results.items():
        serializable[model_type] = {
            'model_type': r['model_type'],
            'seed': r['seed'],
            'train_losses': make_serializable(r['train_losses']),
            'val_node_aucs': make_serializable(r['val_node_aucs']),
            'val_node_f1s': make_serializable(r['val_node_f1s']),
            'val_node_precisions': make_serializable(r['val_node_precisions']),
            'val_node_recalls': make_serializable(r['val_node_recalls']),
            'best_epoch': r['best_epoch'],
            'best_val_auc': r['best_val_auc'],
            'test_metrics': make_serializable(r['test_metrics']),
            'category_results': make_serializable(r['category_results']),
            'config': r['config'],
            'runtime': r['runtime']
        }

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {path}")


def plot_training_curves(results: Dict, save_path: str = 'results/training_curves.png'):
    """Plot training curves. Node-level validation, not graph-level."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for model_type in ['gcn', 'stgnn']:
        if model_type not in results:
            continue

        r = results[model_type]
        color = 'blue' if model_type == 'gcn' else 'red'

        # Training loss
        axes[0, 0].plot(r['train_losses'], color=color, label=model_type.upper())
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        # Val node AUC
        axes[0, 1].plot(r['val_node_aucs'], color=color, label=model_type.upper())
        axes[0, 1].set_title('Validation Node AUC-ROC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC-ROC')
        axes[0, 1].legend()

        # Val node F1
        axes[0, 2].plot(r['val_node_f1s'], color=color, label=model_type.upper())
        axes[0, 2].set_title('Validation Node F1')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1')
        axes[0, 2].legend()

        # Val precision & recall
        axes[1, 0].plot(r['val_node_precisions'], color=color, linestyle='--', label=f'{model_type.upper()} Prec')
        axes[1, 0].plot(r['val_node_recalls'], color=color, linestyle='-.', label=f'{model_type.upper()} Rec')
        axes[1, 0].set_title('Validation Node Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()

        # Confusion matrix
        cm = np.array(r['test_metrics']['confusion_matrix'])
        im = axes[1, 1].imshow(cm, cmap='Blues')
        axes[1, 1].set_title(f'{model_type.upper()} Test Confusion Matrix')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(['Normal', 'Attack'])
        axes[1, 1].set_yticklabels(['Normal', 'Attack'])
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center')

        # Placeholder for per-category
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to {save_path}")


def multi_seed_evaluation(data: Dict, device: torch.device, config: Dict,
                          seeds: List[int] = [1, 2, 3, 4, 5]) -> Dict:
    """Run both models across multiple seeds. Saves each seed separately + aggregate."""
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        try:
            results, _, _ = compare_models(data, device, config, seeds=[seed])
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            results = {}

        all_results[seed] = results

        # Save individual seed
        save_results(results, f'results/seed_{seed}_results.json', seeds=[seed])

    # Aggregate
    aggregate = {}
    for model_type in ['gcn', 'stgnn']:
        aucs, f1s, precs, recs = [], [], [], []
        for seed in seeds:
            if model_type in all_results.get(seed, {}):
                tm = all_results[seed][model_type]['test_metrics']
                if tm['node_auc'] is not None:
                    aucs.append(tm['node_auc'])
                f1s.append(tm['node_f1'])
                precs.append(tm['node_precision'])
                recs.append(tm['node_recall'])

        if aucs:
            aggregate[model_type] = {
                'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
                'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                'precision_mean': float(np.mean(precs)), 'precision_std': float(np.std(precs)),
                'recall_mean': float(np.mean(recs)), 'recall_std': float(np.std(recs))
            }

    # Save aggregate
    os.makedirs('results', exist_ok=True)
    with open('results/multi_seed_aggregate.json', 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*60}")
    print("MULTI-SEED AGGREGATE (mean ± std)")
    print(f"{'='*60}")
    for model_type in ['gcn', 'stgnn']:
        if model_type in aggregate:
            a = aggregate[model_type]
            print(f"\n{model_type.upper()}:")
            print(f"  AUC:        {a['auc_mean']:.4f} ± {a['auc_std']:.4f}")
            print(f"  F1:         {a['f1_mean']:.4f} ± {a['f1_std']:.4f}")
            print(f"  Precision:  {a['precision_mean']:.4f} ± {a['precision_std']:.4f}")
            print(f"  Recall:     {a['recall_mean']:.4f} ± {a['recall_std']:.4f}")

    return all_results, aggregate


if __name__ == '__main__':
    from src.data_loader import load_unsw_nb15
    device = torch.device('cpu')
    data = load_unsw_nb15(download=False)
    config = {
        'hidden_channels': 128, 'gcn_layers': 3, 'temporal_layers': 2,
        'num_heads': 4, 'dropout': 0.5, 'lr': 1e-3, 'weight_decay': 5e-4,
        'epochs': 200, 'patience': 20, 'sequence_length': 5, 'max_train_sequences': 200
    }
    # Sanity checks
    assert len(set(data['train_df'].index) & set(data['val_df'].index)) == 0, "Train/Val overlap!"
    print("Sanity check passed: train/val disjoint")
    results, _, _ = compare_models(data, device, config)
    print_comparison_table(results)
    save_results(results)
    plot_training_curves(results)
