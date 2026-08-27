"""
ST-GraphAD: Spatio-Temporal Graph Neural Network for Network Intrusion Detection
"""

__version__ = "1.0.0"
__author__ = "Ayan Bhaumik"

from .data_loader import load_unsw_nb15, UNSWNB15Loader
from .models import GCNOnly, STGNN, create_model, TemporalAttention, PositionalEncoding
from .train import Trainer, train_model, compare_models
from .visualize import (
    plot_graph_structure,
    plot_embeddings_tsne,
    plot_embeddings_pca,
    plot_attention_weights,
    plot_temporal_anomaly_scores,
    visualize_model_predictions,
    generate_report
)

__all__ = [
    "load_unsw_nb15",
    "UNSWNB15Loader",
    "GCNOnly",
    "STGNN",
    "create_model",
    "TemporalAttention",
    "PositionalEncoding",
    "Trainer",
    "train_model",
    "compare_models",
    "plot_graph_structure",
    "plot_embeddings_tsne",
    "plot_embeddings_pca",
    "plot_attention_weights",
    "plot_temporal_anomaly_scores",
    "visualize_model_predictions",
    "generate_report",
]