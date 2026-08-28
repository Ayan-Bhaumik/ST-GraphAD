# ST-GraphAD: Spatio-Temporal Graph Neural Network for Network Intrusion Detection on UNSW-NB15

A PyTorch Geometric implementation of ST-GraphAD, a Spatio-Temporal Graph Neural Network for network intrusion detection on the UNSW-NB15 dataset. The system converts network flows into dynamic attributed graphs and uses a 3-layer Graph Convolutional Network (GCN) combined with a 2-layer, 4-head temporal attention mechanism to detect anomalies.

## Key Results (Five-Seed Aggregate, Node-Level)

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| GCN-only | 0.612 ± 0.030 | 0.836 ± 0.013 | 0.789 ± 0.015 | 0.889 ± 0.023 |
| **ST-GraphAD** | **0.657 ± 0.078** | **0.858 ± 0.021** | **0.791 ± 0.022** | **0.939 ± 0.037** |

**Improvements (mean to mean):**
- AUC-ROC: +4.5 percentage points
- F1: +2.2 percentage points
- Recall: +5.0 percentage points

Graph-level evaluation is degenerate (~88% attack nodes → majority-vote graph label is "Attack") and is not the primary task.

## Architecture

```
Network Flows → Pseudo-Node Graph Construction → 3-Layer GCN Spatial Encoding → 2-Layer Temporal Attention (4 heads) → Node/Graph Classification
   (UNSW-NB15)     (protocol-service-state tuples)        (128 hidden dims)           (sequence length 5)              (anomaly scores)
```

### Components

1. **Graph Construction**: Network flows are converted to graphs where nodes are pseudo-IPs derived from (protocol, service, role) tuples (because the UNSW-NB15 CSV lacks explicit IP addresses) and edges represent communication flows. This is a documented limitation, not a hidden assumption.

2. **GCN Encoder**: 3-layer Graph Convolutional Network (128 hidden channels, batch norm, dropout=0.5) extracts spatial features from each time window's graph structure.

3. **Temporal Attention**: 2-layer Multi-Head Self-Attention (4 heads, 128 dimensions) with positional encoding models temporal dependencies across a sliding window of 5 graph snapshots.

4. **Classification Heads**: 
   - Graph-level: MLP over mean-pooled final window embeddings
   - Node-level: MLP over final window node embeddings (primary task)

## Features

- ✅ UNSW-NB15 dataset loading and preprocessing with **no leakage** (encoders/scalers fit on training partition only)
- ✅ Flow-to-graph conversion with pseudo-node construction
- ✅ GCN-only baseline (3 layers, 128 hidden, no temporal attention)
- ✅ ST-GraphAD (full architecture: GCN + temporal attention)
- ✅ **Proper train/validation/test split**: 85/15 from training CSV + official test CSV held out
- ✅ Node-level validation AUC-ROC for early stopping and model selection
- ✅ **Five-seed evaluation** (42, 123, 456, 789, 999) with mean±std reporting
- ✅ Comprehensive metrics (AUC-ROC, F1, Precision, Recall) at node level
- ✅ Graph-level metrics reported with explicit degeneracy caveat
- ✅ Model checkpointing and evaluation reports
- ✅ Computational complexity analysis

## Installation

### macOS / Linux

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install pandas scikit-learn networkx matplotlib tqdm
```

## Dataset

The UNSW-NB15 dataset must be placed in the `data/` directory:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Usage

### Train and Evaluate (Five-Seed Final Experiment)

```bash
# Train both models with 5 seeds, 50 epochs each (final experiment configuration)
python main.py --model both --multi-seed --seeds 42 123 456 789 999 --epochs 50 --patience 10
```

This produces:
- `results/seed_<seed>_results.json` — per-seed detailed results
- `results/multi_seed_aggregate.json` — five-seed aggregate (mean ± std)
- `results/evaluation_results.json` — single-seed (last run) results
- `results/report.md` — summary report
- `results/training_curves.png` — training curves
- `models/gcn_model.pt`, `models/stgnn_model.pt` — best checkpoints

### Train Single Model (Single Seed)

```bash
# GCN-only baseline
python main.py --model gcn --epochs 50 --patience 10

# ST-GraphAD
python main.py --model stgnn --epochs 50 --patience 10
```

### Generate Visualizations from Saved Models

```bash
python main.py --visualize
```

## Configuration (Final Experiment)

| Parameter | Value |
|-----------|-------|
| Hidden channels | 128 |
| GCN layers | 3 |
| Temporal attention layers | 2 |
| Attention heads | 4 |
| Dropout | 0.5 |
| Learning rate | 1e-3 |
| Weight decay | 5e-4 |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Sequence length | 5 |
| Max train sequences/epoch | 200 |
| Loss weight (λ) | 0.5 |
| Seeds | 42, 123, 456, 789, 999 |

## Dataset Split

| Split | Flows | Temporal Graphs | Nodes | Edges |
|-------|-------|-----------------|-------|-------|
| Train | 149,039 | 150 | 178 | 298K |
| Validation | 26,302 | 27 | 174 | 53K |
| Test (official) | 82,332 | 83 | 172 | 165K |

- Training CSV split 85/15 stratified by flow label
- Official test CSV **never used** during training/validation
- Preprocessing (LabelEncoder, StandardScaler) fitted on training partition only

## Known Limitations

1. **Pseudo-node construction**: Nodes are (protocol, service, role) tuples, not real IPs. Multiple real hosts collapse to one pseudo-node.
2. **Pseudo-temporal ordering**: Windows derived from CSV row order (1000 flows/window), not real timestamps.
3. **Binary node labels only**: No per-attack-category evaluation implemented.
4. **Higher variance**: ST-GraphAD AUC std 0.078 vs GCN 0.030 — sensitivity to initialization.
5. **CPU-only training**: GPU not available; larger hyperparameter searches infeasible.
6. **No attention interpretability**: Head specialization and attention entropy not quantitatively analyzed.
7. **Single dataset**: Results may not generalize to other NIDS datasets.

## Reproducibility

To reproduce the exact five-seed final experiment:

```bash
python main.py --model both --multi-seed --seeds 42 123 456 789 999 \
    --epochs 50 --patience 10 --hidden 128 --layers 3 --temporal-layers 2 --heads 4
```

Expected outputs (approximately):
- GCN: AUC 0.612 ± 0.030, F1 0.836 ± 0.013
- ST-GraphAD: AUC 0.657 ± 0.078, F1 0.858 ± 0.021

## Project Structure

```
ST-GraphAD/
├── main.py                 # Entry point
├── src/
│   ├── data_loader.py      # UNSW-NB15 loading, preprocessing, graph construction
│   ├── models.py           # GCNOnly, STGNN model definitions
│   ├── train.py            # Training, evaluation, multi-seed orchestration
│   └── visualize.py        # Visualization utilities
├── data/                   # UNSW-NB15 CSV files (not tracked)
├── results/                # Output: JSON results, reports, figures
├── models/                 # Saved model checkpoints
├── manuscript_full_text.txt    # Full manuscript (publication-ready)
├── RESEARCH.md             # Extended research document
└── README.md               # This file
```

## Citation

If you use this work, please cite:

```bibtex
@misc{bhaumik2024stgraphad,
  author       = {Ayan Bhaumik},
  title        = {ST-GraphAD: Spatio-Temporal Graph Neural Network for Network Intrusion Detection on UNSW-NB15},
  year         = {2024},
  note         = {Independent Research},
  url          = {https://github.com/ayanbhaumik/ST-GraphAD}
}
```

## License

MIT License