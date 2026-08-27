# Spatio-Temporal GNN for Network Intrusion Detection

## UNSW-NB15 Dataset

### Model Configuration

- **hidden_channels**: 128
- **gcn_layers**: 3
- **temporal_layers**: 2
- **num_heads**: 4
- **dropout**: 0.5
- **lr**: 0.001
- **weight_decay**: 0.0005
- **epochs**: 100
- **patience**: 20
- **sequence_length**: 5
- **max_train_sequences**: 200

## Results Comparison

| Metric | GCN-only | ST-GNN | Improvement |
|--------|----------|--------|-------------|
| graph_auc | 0.0000 | 0.0000 | +0.0% |
| graph_f1 | 1.0000 | 1.0000 | +0.0% |
| graph_precision | 1.0000 | 1.0000 | +0.0% |
| graph_recall | 1.0000 | 1.0000 | +0.0% |

## Visualizations

![Training Curves](training_curves.png)

![Confusion Matrices](confusion_matrices.png)

![ROC Curves](roc_curves.png)

![Metrics Comparison](metrics_comparison.png)

