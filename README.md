# Spatio-Temporal Graph Neural Network for Network Intrusion Detection

A PyTorch-based implementation of a Spatio-Temporal Graph Neural Network (ST-GNN) for
network intrusion detection on the UNSW-NB15 dataset. The system converts network flows
into graph structures and uses Graph Convolutional Networks (GCN) combined with temporal
attention mechanisms to detect anomalies.

## Architecture

```
Network Flows → Graph Construction → GCN Spatial Encoding → Temporal Attention → Classification
   (UNSW-NB15)    (adjacency matrix)   (node embeddings)     (sequence modeling)   (anomaly score)
```

### Components

1. **Graph Construction**: Network flows (source IP, dest IP, ports) are converted to a
   graph adjacency matrix where nodes are IP addresses and edges represent communication.

2. **GCN Encoder**: Graph Convolutional Network extracts spatial features from the graph
   structure, capturing communication patterns between network entities.

3. **Temporal Attention**: Multi-head attention mechanism models temporal dependencies
   across time windows, allowing the model to detect evolving attack patterns.

4. **Classifier**: Predicts anomaly scores for graph-level and node-level detection.

## Features

- ✅ UNSW-NB15 dataset loading and preprocessing
- ✅ Flow-to-graph conversion with multiple node features
- ✅ GCN-only baseline
- ✅ ST-GNN (GCN + Temporal Attention)
- ✅ Comprehensive evaluation (AUC-ROC, F1-score, Precision, Recall)
- ✅ Model comparison (GCN vs ST-GNN)
- ✅ Multiple visualizations (ROC, t-SNE, attention weights, graph structure)
- ✅ Model checkpointing and evaluation report

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
pip install -r requirements.txt
```

**For Apple Silicon (M1/M2/M3/M4) with GPU acceleration (MPS):**
```bash
# PyTorch with MPS support comes with standard install on macOS 12.3+ (PyTorch 2.0+)
pip install torch torch-geometric

# Verify MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**Troubleshooting M4 / Apple Silicon:**

If you encounter issues on M4, try these steps:

```bash
# 1. Ensure you're on Python 3.10+ (M4 requires 3.10+)
python --version

# 2. Update pip and setuptools
pip install --upgrade pip setuptools wheel

# 3. Install with no cache (avoids corrupted wheels)
pip install --no-cache-dir torch torch-geometric

# 4. If torch-geometric fails, install PyG dependencies first
pip install --no-cache-dir pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch-geometric

# 5. Alternative: Use conda (often more reliable on Apple Silicon)
conda create -n st-graphad python=3.11
conda activate st-graphad
conda install pytorch torchvision torchaudio -c pytorch
conda install pyg -c pyg
conda install pandas numpy scikit-learn matplotlib networkx scipy tqdm
```

**Common M4 Issues & Fixes:**

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | Use `--no-cache-dir` or conda |
| `MPS not available` | Requires macOS 13.0+, PyTorch 2.0+ |
| `torch-geometric` install fails | Install PyG dependencies first (step 4 above) |
| `ImportError: libomp.dylib` | `brew install libomp` |
| `RuntimeError: MPS backend out of memory` | Reduce batch size, use CPU: `device = torch.device('cpu')` |

**Force CPU Mode (if MPS causes issues):**
```python
# In main.py or your script, force CPU
device = torch.device('cpu')
# Or set environment variable
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

### Windows

**Option 1: Command Prompt (cmd.exe)**
```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**Option 2: PowerShell**
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**Option 3: Windows with NVIDIA GPU (CUDA)**
```cmd
# Install PyTorch with CUDA 11.8 support
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**For CUDA 12.1:**
```cmd
pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cu121
```

### Manual Install (All Platforms)
```bash
pip install torch torch-geometric pandas numpy scikit-learn matplotlib networkx scipy tqdm
```

### Verify Installation
```bash
# Test imports
python scripts/test_imports.py

# Quick model test
python scripts/test_model.py
```

## Dataset

The UNSW-NB15 dataset is automatically downloaded if not present. You can also manually
download from: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/

Place the files in `data/`:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

## Usage

### Train both models (GCN vs ST-GNN comparison)

```bash
python main.py --model both --download
```

### Train only GCN baseline

```bash
python main.py --model gcn
```

### Train only ST-GNN

```bash
python main.py --model stgnn
```

### Generate visualizations from trained models

```bash
python main.py --model both --visualize
```

### Advanced options

```bash
python main.py --model both \
    --epochs 200 \
    --hidden 256 \
    --layers 4 \
    --temporal-layers 3 \
    --heads 8 \
    --seq-len 10 \
    --dropout 0.3
```

## Project Structure

```
ST-GraphAD/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── src/
│   ├── data_loader.py      # UNSW-NB15 loading & graph construction
│   ├── models.py           # GCN, Temporal Attention, ST-GNN models
│   ├── train.py            # Training & evaluation pipeline
│   ├── visualize.py        # Visualization utilities
├── data/                   # Dataset (auto-downloaded)
├── models/                 # Saved model checkpoints
│   ├── gcn_model.pt
│   └── stgnn_model.pt
└── results/                # Evaluation results & visualizations
    ├── evaluation_results.json
    ├── training_curves.png
    ├── metrics_comparison.png
    └── predictions_*/
```

## Evaluation Metrics

The system reports:

- **AUC-ROC**: Area under the ROC curve (graph-level and node-level)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **Confusion Matrix**: Detailed breakdown of predictions

## Model Comparison

The system automatically compares:

| Model | Description |
|-------|-------------|
| GCN-only | Static graph convolution, no temporal modeling |
| ST-GNN | GCN + Multi-head Temporal Attention |

Expected improvements with temporal attention:
- Better detection of evolving attack patterns
- Improved precision by reducing false positives
- Higher AUC-ROC for sequential anomaly detection

## Visualization Outputs

1. **Training Curves**: Loss, AUC, F1 over epochs
2. **ROC Curves**: Model comparison ROC curves
3. **Confusion Matrices**: Per-model confusion matrix
4. **t-SNE/PCA**: Embedding space visualization
5. **Attention Weights**: Temporal attention patterns
6. **Graph Structure**: Visual representation of network topology
7. **Anomaly Score Distribution**: Normal vs attack score separation

## References

- K. M. Al-Saadi et al., "UNSW-NB15: a comprehensive data set for network intrusion detection systems"
- T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
- A. Vaswani et al., "Attention Is All You Need"

## License

MIT License
