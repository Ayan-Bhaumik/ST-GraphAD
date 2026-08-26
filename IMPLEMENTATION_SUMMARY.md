# ST-GraphAD Implementation Summary

## Overview
Complete implementation of a Spatio-Temporal Graph Neural Network for Network Intrusion Detection on UNSW-NB15 dataset.

## Project Structure
```
ST-GraphAD/
├── main.py                    # Main entry point with CLI
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── README.md                  # Documentation
├── IMPLEMENTATION_SUMMARY.md  # This file
├── .gitignore                 # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # UNSW-NB15 loading & graph construction
│   ├── models.py              # GCN, Temporal Attention, ST-GNN
│   ├── train.py               # Training & evaluation pipeline
│   └── visualize.py           # Visualization utilities
├── scripts/
│   ├── demo.py                # Synthetic data demo
│   ├── inference.py           # Inference on new data
│   ├── test_imports.py        # Import verification
│   └── test_model.py          # Model forward pass tests
├── notebooks/
│   └── explore_data.ipynb     # Data exploration notebook
├── data/                      # Dataset (auto-download)
├── models/                    # Saved checkpoints
└── results/                   # Evaluation outputs
```

## Key Components

### 1. Data Loader (`src/data_loader.py`)
- **UNSWNB15Loader**: Downloads, loads, and preprocesses UNSW-NB15
- **Graph Construction**: Converts flows (src/dst IP, ports) to adjacency matrix
- **Node Features**: Aggregates flow statistics per IP (mean bytes, packets, duration, etc.)
- **Temporal Graphs**: Creates time-windowed graph sequences (1-hour windows)
- **Encoders**: LabelEncoder for categorical, StandardScaler for numerical, IP/Port encoders

### 2. Models (`src/models.py`)
- **GCNEncoder**: Multi-layer Graph Convolutional Network with batch norm
- **TemporalAttention**: Multi-head attention with positional encoding
- **STGNN**: Combines GCN + Temporal Attention for spatio-temporal modeling
- **GCNOnly**: Baseline GCN without temporal component
- **FocalLoss**: For handling class imbalance

### 3. Training Pipeline (`src/train.py`)
- **Trainer**: Training loop with early stopping, LR scheduling
- **Sequence Preparation**: Sliding windows for temporal training
- **Evaluation**: AUC-ROC, F1, Precision, Recall at graph and node level
- **Comparison**: Automatic GCN vs ST-GNN comparison

### 4. Visualizations (`src/visualize.py`)
- Graph structure plots (NetworkX)
- t-SNE/PCA of embeddings
- Attention weight heatmaps
- ROC/PR curves
- Metrics comparison bars
- Anomaly score distributions
- Confusion matrices

## Usage Instructions

### 1. Install Dependencies
```bash
# When network is available:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training
```bash
# Compare both models (recommended)
python main.py --model both --download

# Or train individually
python main.py --model gcn
python main.py --model stgnn
```

### 3. Generate Visualizations
```bash
python main.py --model both --visualize
```

### 4. Run Demo (no dataset needed)
```bash
python scripts/demo.py
```

### 5. Test Imports
```bash
python scripts/test_imports.py
```

### 6. Test Model Forward Pass
```bash
python scripts/test_model.py
```

## Key Features Implemented

✅ **Data Loading**: UNSW-NB15 download, preprocessing, graph construction
✅ **Graph Representation**: IPs as nodes, flows as edges, aggregated features
✅ **Temporal Sequences**: Time-windowed graphs for sequence modeling
✅ **GCN Baseline**: Standard graph convolution for spatial features
✅ **Temporal Attention**: Multi-head attention over graph sequence
✅ **ST-GNN**: End-to-end spatio-temporal model
✅ **Evaluation**: AUC-ROC, F1, Precision, Recall (graph & node level)
✅ **Model Comparison**: Automated GCN vs ST-GNN benchmarking
✅ **Visualizations**: 10+ plot types for analysis
✅ **CLI Interface**: Flexible training/inference arguments
✅ **Checkpointing**: Model saving/loading
✅ **Report Generation**: Markdown evaluation report

## Expected Outputs

After running `python main.py --model both`:

```
models/
  ├── gcn_model.pt
  └── stgnn_model.pt

results/
  ├── evaluation_results.json
  ├── training_curves.png
  ├── metrics_comparison.png
  ├── confusion_matrices.png
  ├── roc_curves.png
  ├── report.md
  └── predictions_gcn/
  └── predictions_stgnn/
```

## Configuration Options

```bash
python main.py --model both \
    --epochs 200 \
    --hidden 256 \
    --layers 4 \
    --temporal-layers 3 \
    --heads 8 \
    --seq-len 10 \
    --dropout 0.3 \
    --lr 0.001
```

## Network Requirements

The dataset downloads from:
- `https://raw.githubusercontent.com/unsw-nb15/dataset/master/UNSW_NB15_training-set.csv`
- `https://raw.githubusercontent.com/unsw-nb15/dataset/master/UNSW_NB15_testing-set.csv`

If network is blocked, manually download and place in `data/`.

## Research Contributions

1. **Graph-based NID**: Converts tabular flows to graph structure preserving topology
2. **Temporal Modeling**: Captures attack evolution over time windows
3. **Node-level Detection**: Identifies suspicious IPs, not just graph-level
4. **Ablation Study**: Built-in GCN vs ST-GNN comparison
5. **Interpretability**: Attention weights show temporal importance

## References

- UNSW-NB15 Dataset: Moustafa & Slay (2015)
- GCN: Kipf & Welling (2017)
- Attention: Vaswani et al. (2017)
- PyTorch Geometric: Fey & Lenssen (2019)