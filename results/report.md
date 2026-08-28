# Spatio-Temporal GNN for Network Intrusion Detection

## UNSW-NB15 Dataset

### Model Configuration

- **hidden_channels**: 256
- **gcn_layers**: 4
- **temporal_layers**: 2
- **num_heads**: 4
- **dropout**: 0.5
- **lr**: 0.001
- **weight_decay**: 0.0005
- **epochs**: 200
- **patience**: 20
- **sequence_length**: 5
- **max_train_sequences**: 200

## Results Comparison (Node-Level)

| Metric | GCN-only | ST-GNN | Improvement |
|--------|----------|--------|-------------|
| node_auc | 0.5016 | 0.5873 | +17.1% |
| node_f1 | 0.8137 | 0.8105 | -0.4% |
| node_precision | 0.7571 | 0.7469 | -1.3% |
| node_recall | 0.8795 | 0.8860 | +0.7% |

## Detailed Results


### GCN

- Best validation epoch: 105
- Best validation node AUC: 0.9928
- Runtime: 106.8s
- Test confusion matrix: [[99, 438], [187, 1365]]
- Positive samples: 1552, Negative samples: 537

### STGNN

- Best validation epoch: 106
- Best validation node AUC: 0.9934
- Runtime: 625.5s
- Test confusion matrix: [[71, 466], [177, 1375]]
- Positive samples: 1552, Negative samples: 537

## Visualizations

![Training Curves](training_curves.png)

![Confusion Matrices](confusion_matrices.png)

![ROC Curves](roc_curves.png)

![Metrics Comparison](metrics_comparison.png)

