"""
Visualization utilities for ST-GNN intrusion detection results.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


def plot_graph_structure(graph: Data, labels: Optional[torch.Tensor] = None,
                         save_path: str = 'results/graph_structure.png',
                         max_nodes: int = 500):
    """Visualize graph structure with node colors by label."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Subsample if too large
    if graph.num_nodes > max_nodes:
        indices = np.random.choice(graph.num_nodes, max_nodes, replace=False)
        subgraph = graph.subgraph(torch.tensor(indices))
    else:
        subgraph = graph

    # Convert to networkx
    G = to_networkx(subgraph, to_undirected=True)

    # Node colors
    if labels is not None:
        node_colors = labels[list(G.nodes())].cpu().numpy()
        cmap = plt.cm.RdYlGn_r
    else:
        node_colors = 'lightblue'
        cmap = None

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap,
                          node_size=30, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)

    if labels is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label='Attack (1) / Normal (0)')

    plt.title(f'Graph Structure ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph structure saved to {save_path}")


def plot_embeddings_tsne(embeddings: torch.Tensor, labels: torch.Tensor,
                         save_path: str = 'results/embeddings_tsne.png',
                         perplexity: int = 30):
    """Plot t-SNE visualization of node embeddings."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    emb_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Reduce dimensionality with PCA first if needed
    if emb_np.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        emb_np = pca.fit_transform(emb_np)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, n_jobs=-1)
    emb_2d = tsne.fit_transform(emb_np)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_np,
                         cmap='RdYlGn_r', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Attack (1) / Normal (0)')
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


def plot_embeddings_pca(embeddings: torch.Tensor, labels: torch.Tensor,
                        save_path: str = 'results/embeddings_pca.png'):
    """Plot PCA visualization of node embeddings."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    emb_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(emb_np)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_np,
                         cmap='RdYlGn_r', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Attack (1) / Normal (0)')
    plt.title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved to {save_path}")


def plot_attention_weights(attention_weights: torch.Tensor,
                           save_path: str = 'results/attention_weights.png'):
    """Plot temporal attention weights."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # attention_weights: [num_heads, seq_len, seq_len]
    attn = attention_weights.detach().cpu().numpy()
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(1, num_heads, figsize=(4*num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        im = axes[h].imshow(attn[h], cmap='Blues', aspect='auto')
        axes[h].set_title(f'Head {h+1}')
        axes[h].set_xlabel('Key Position')
        axes[h].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[h])

    plt.suptitle('Temporal Attention Weights')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention weights saved to {save_path}")


def plot_temporal_anomaly_scores(scores: Dict[str, List[float]],
                                  save_path: str = 'results/temporal_anomaly.png'):
    """Plot anomaly scores over time."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 6))

    for model_name, score_list in scores.items():
        plt.plot(score_list, label=model_name.upper(), marker='o', markersize=3, alpha=0.7)

    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    plt.xlabel('Time Window')
    plt.ylabel('Anomaly Score')
    plt.title('Temporal Anomaly Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Temporal anomaly scores saved to {save_path}")


def plot_feature_importance(feature_names: List[str], importance: np.ndarray,
                             save_path: str = 'results/feature_importance.png',
                             top_k: int = 20):
    """Plot feature importance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get top k features
    indices = np.argsort(importance)[-top_k:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_k} Feature Importances')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature importance saved to {save_path}")


def plot_metrics_comparison(results: Dict, save_path: str = 'results/metrics_comparison.png'):
    """Plot comprehensive metrics comparison."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = list(results.keys())
    metrics = ['graph_auc', 'graph_f1', 'graph_precision', 'graph_recall']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results[m]['final_metrics'].get(metric, 0) for m in models]

        bars = axes[idx].bar(models, values, color=['skyblue', 'lightcoral'], alpha=0.8, edgecolor='black')
        axes[idx].set_title(metric.replace('graph_', '').upper())
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim(0, 1.05)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved to {save_path}")


def plot_roc_curve_detailed(y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                             save_path: str = 'results/roc_curve_detailed.png'):
    """Plot detailed ROC curves."""
    from sklearn.metrics import roc_curve, auc
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))

    for model_name, scores in y_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name.upper()} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Detailed Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Detailed ROC curve saved to {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                                 save_path: str = 'results/pr_curve.png'):
    """Plot precision-recall curves."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))

    for model_name, scores in y_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, linewidth=2, label=f'{model_name.upper()} (AP = {ap:.3f})')

    # Baseline
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (AP = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")


def visualize_model_predictions(model, data: Dict, device: torch.device,
                                 save_dir: str = 'results/predictions'):
    """Generate comprehensive visualizations for model predictions."""
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Get predictions on test data
        if hasattr(model, 'use_temporal') and model.use_temporal:
            output = model(data['test_temporal'][:5], return_node_scores=True)
        else:
            output = model([data['test_static']], return_node_scores=True)

        # Graph-level
        graph_probs = output['graph_probs'].cpu().numpy()
        graph_preds = graph_probs.argmax(axis=1)

        # Node-level
        if 'node_probs' in output:
            node_probs = output['node_probs'].cpu().numpy()
            node_preds = node_probs.argmax(axis=1)
            anomaly_scores = node_probs[:, 1]

            # Plot node embedding space
            plot_embeddings_tsne(output['node_embeddings'], data['test_static'].y,
                                os.path.join(save_dir, 'test_embeddings_tsne.png'))
            plot_embeddings_pca(output['node_embeddings'], data['test_static'].y,
                               os.path.join(save_dir, 'test_embeddings_pca.png'))

            # Plot anomaly score distribution
            plt.figure(figsize=(10, 6))
            plt.hist(anomaly_scores[data['test_static'].y == 0], bins=50, alpha=0.5, label='Normal', density=True)
            plt.hist(anomaly_scores[data['test_static'].y == 1], bins=50, alpha=0.5, label='Attack', density=True)
            plt.xlabel('Anomaly Score')
            plt.ylabel('Density')
            plt.title('Anomaly Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'anomaly_score_distribution.png'), dpi=150)
            plt.close()

        # Graph structure
        plot_graph_structure(data['test_static'], data['test_static'].y,
                            os.path.join(save_dir, 'test_graph_structure.png'))


def generate_report(results: Dict, config: Dict, save_path: str = 'results/report.md'):
    """Generate markdown report."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("# Spatio-Temporal GNN for Network Intrusion Detection\n\n")
        f.write("## UNSW-NB15 Dataset\n\n")
        f.write("### Model Configuration\n\n")
        for key, value in config.items():
            f.write(f"- **{key}**: {value}\n")

        f.write("\n## Results Comparison (Node-Level)\n\n")
        f.write("| Metric | GCN-only | ST-GNN | Improvement |\n")
        f.write("|--------|----------|--------|-------------|\n")

        for model_type in ['gcn', 'stgnn']:
            if model_type not in results:
                continue

        metrics = ['node_auc', 'node_f1', 'node_precision', 'node_recall']
        for metric in metrics:
            gcn = results['gcn']['test_metrics'].get(metric, 0) if 'gcn' in results else 0
            stgnn = results['stgnn']['test_metrics'].get(metric, 0) if 'stgnn' in results else 0
            imp = ((stgnn - gcn) / gcn * 100) if gcn > 0 else 0
            f.write(f"| {metric} | {gcn:.4f} | {stgnn:.4f} | {imp:+.1f}% |\n")

        f.write("\n## Detailed Results\n\n")
        for model_type in ['gcn', 'stgnn']:
            if model_type not in results:
                continue
            r = results[model_type]
            tm = r['test_metrics']
            f.write(f"\n### {model_type.upper()}\n\n")
            f.write(f"- Best validation epoch: {r['best_epoch']}\n")
            f.write(f"- Best validation node AUC: {r['best_val_auc']:.4f}\n")
            f.write(f"- Runtime: {r['runtime']:.1f}s\n")
            f.write(f"- Test confusion matrix: {tm.get('confusion_matrix', 'N/A')}\n")
            f.write(f"- Positive samples: {tm.get('n_positive', 0)}, Negative samples: {tm.get('n_negative', 0)}\n")

        f.write("\n## Visualizations\n\n")
        f.write("![Training Curves](training_curves.png)\n\n")
        f.write("![Confusion Matrices](confusion_matrices.png)\n\n")
        f.write("![ROC Curves](roc_curves.png)\n\n")
        f.write("![Metrics Comparison](metrics_comparison.png)\n\n")

    print(f"Report saved to {save_path}")


if __name__ == '__main__':
    # Test visualizations with dummy data
    print("Visualization module ready. Import and use functions in your pipeline.")