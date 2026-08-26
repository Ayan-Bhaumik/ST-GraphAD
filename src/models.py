"""
Spatio-Temporal Graph Neural Network Models for Network Intrusion Detection.

Models:
1. GCN-only: Standard Graph Convolutional Network
2. ST-GNN: GCN + Temporal Attention for spatio-temporal anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Tuple
import math


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder for spatial features."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, dropout: float = 0.5, use_batch_norm: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # No activation after last layer
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence of graph embeddings."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Positional encoding for temporal order
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, embed_dim] - sequence of graph embeddings
            mask: [batch_size, seq_len] - optional padding mask
        Returns:
            output: [batch_size, seq_len, embed_dim] - attended sequence
            attn_weights: [batch_size, num_heads, seq_len, seq_len] - attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        x = self.pos_encoding(x)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        return out, attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Intrusion Detection.

    Architecture:
    1. GCN Encoder: Extracts spatial features from each time window's graph
    2. Temporal Attention: Models temporal dependencies across time windows
    3. Classifier: Predicts anomaly score for each node/time window
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 gcn_layers: int = 3, temporal_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.5,
                 use_temporal: bool = True):
        super().__init__()
        self.use_temporal = use_temporal
        self.hidden_channels = hidden_channels

        # Spatial encoder (GCN)
        self.gcn_encoder = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=gcn_layers,
            dropout=dropout
        )

        # Temporal attention layers
        if use_temporal:
            self.temporal_attention = nn.ModuleList([
                TemporalAttention(hidden_channels, num_heads, dropout)
                for _ in range(temporal_layers)
            ])
            self.temporal_norms = nn.ModuleList([
                nn.LayerNorm(hidden_channels) for _ in range(temporal_layers)
            ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)  # Binary classification: normal vs attack
        )

        # Node-level classifier for per-node anomaly scoring
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )

    def forward_spatial(self, graphs: List[Data]) -> List[torch.Tensor]:
        """Encode each graph in the temporal sequence."""
        embeddings = []
        for graph in graphs:
            x = self.gcn_encoder(graph.x, graph.edge_index)
            embeddings.append(x)
        return embeddings

    def forward_temporal(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply temporal attention over sequence of graph embeddings.

        Args:
            embeddings: List of [num_nodes_i, hidden_channels] tensors
        Returns:
            Temporally enhanced embeddings for the last time window
        """
        # Pad sequences to same length (max nodes across windows)
        max_nodes = max(e.shape[0] for e in embeddings)
        batch_size = len(embeddings)

        padded = torch.zeros(batch_size, max_nodes, self.hidden_channels, device=embeddings[0].device)
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=embeddings[0].device)

        for i, emb in enumerate(embeddings):
            padded[i, :emb.shape[0]] = emb
            mask[i, :emb.shape[0]] = True

        # Apply temporal attention layers
        x = padded
        for attn, norm in zip(self.temporal_attention, self.temporal_norms):
            attn_out, _ = attn(x, mask)
            x = norm(x + attn_out)  # Residual connection

        # Return embeddings for the last time window (most recent)
        last_window_embeddings = x[-1, :embeddings[-1].shape[0]]

        return last_window_embeddings

    def forward(self, graphs: List[Data], return_node_scores: bool = False) -> dict:
        """
        Forward pass.

        Args:
            graphs: List of PyG Data objects (temporal sequence)
            return_node_scores: If True, return per-node anomaly scores

        Returns:
            Dictionary with graph-level and/or node-level predictions
        """
        # Spatial encoding
        spatial_embeddings = self.forward_spatial(graphs)

        if self.use_temporal and len(graphs) > 1:
            # Temporal modeling
            temporal_embeddings = self.forward_temporal(spatial_embeddings)
        else:
            # Use only the last window's spatial embeddings
            temporal_embeddings = spatial_embeddings[-1]

        # Graph-level classification (average pooling)
        graph_embedding = temporal_embeddings.mean(dim=0, keepdim=True)
        graph_logits = self.classifier(graph_embedding)

        output = {
            'graph_logits': graph_logits,
            'graph_probs': F.softmax(graph_logits, dim=-1),
            'node_embeddings': temporal_embeddings
        }

        if return_node_scores:
            node_logits = self.node_classifier(temporal_embeddings)
            output['node_logits'] = node_logits
            output['node_probs'] = F.softmax(node_logits, dim=-1)
            output['anomaly_scores'] = node_logits[:, 1]  # Probability of attack class

        return output


class GCNOnly(nn.Module):
    """GCN-only baseline model (no temporal attention)."""

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 gcn_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.gcn_encoder = GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=gcn_layers,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )

    def forward(self, graph: Data, return_node_scores: bool = False) -> dict:
        embeddings = self.gcn_encoder(graph.x, graph.edge_index)

        graph_embedding = embeddings.mean(dim=0, keepdim=True)
        graph_logits = self.classifier(graph_embedding)

        output = {
            'graph_logits': graph_logits,
            'graph_probs': F.softmax(graph_logits, dim=-1),
            'node_embeddings': embeddings
        }

        if return_node_scores:
            node_logits = self.node_classifier(embeddings)
            output['node_logits'] = node_logits
            output['node_probs'] = F.softmax(node_logits, dim=-1)
            output['anomaly_scores'] = node_logits[:, 1]

        return output


def create_model(model_type: str, in_channels: int, **kwargs) -> nn.Module:
    """Factory function to create models."""
    if model_type == 'gcn':
        return GCNOnly(in_channels=in_channels, **kwargs)
    elif model_type == 'stgnn':
        return STGNN(in_channels=in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Loss functions
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_loss(model_output: dict, labels: torch.Tensor,
                 node_labels: Optional[torch.Tensor] = None,
                 lambda_node: float = 0.5, loss_type: str = 'ce') -> torch.Tensor:
    """Compute combined graph-level and node-level loss."""
    if loss_type == 'focal':
        loss_fn = FocalLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    graph_loss = loss_fn(model_output['graph_logits'], labels)

    if node_labels is not None and 'node_logits' in model_output:
        node_loss = loss_fn(model_output['node_logits'], node_labels)
        total_loss = graph_loss + lambda_node * node_loss
    else:
        total_loss = graph_loss

    return total_loss