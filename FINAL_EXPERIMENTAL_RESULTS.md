# FINAL EXPERIMENTAL RESULTS — ST-GraphAD

**Authoritative Configuration: Configuration C**  
(5 seeds, 128 hidden channels, 3 GCN layers, 2 temporal attention layers, 4 heads, sequence length 5, 50 epochs, early stopping patience 10 on node-level validation AUC-ROC)

---

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| Hidden channels (d) | 128 |
| GCN layers | 3 |
| Temporal attention layers | 2 |
| Attention heads | 4 |
| Dropout | 0.5 |
| Learning rate | 1×10⁻³ |
| Weight decay | 5×10⁻⁴ |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Sequence length (L) | 5 |
| Max train sequences/epoch | 200 (randomly sampled) |
| Loss weight λ | 0.5 |
| Random seeds | 42, 123, 456, 789, 999 |
| Validation metric | Node-level AUC-ROC |
| Device | CPU (Apple M4, 16 GB) |

---

## Dataset Split

| Split | Source | Flows | Temporal Graphs | Nodes | Edges | Preprocessing |
|-------|--------|-------|-----------------|-------|-------|---------------|
| Train | Training CSV (85%) | 149,039 | 150 | 178 | 298K | Fit encoders/scalers |
| Validation | Training CSV (15%) | 26,302 | 27 | 174 | 53K | Transform only |
| Test (official) | Test CSV (100%) | 82,332 | 83 | 172 | 165K | Transform only |

- Stratified by flow label
- LabelEncoder + StandardScaler fit on **training partition only**
- Official test CSV **never used** during training/validation/hyperparameter tuning

---

## Five-Seed Aggregate Results (Principal Headline Result)

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| **GCN-only** | 0.612 ± 0.030 | 0.836 ± 0.013 | 0.789 ± 0.015 | 0.889 ± 0.023 |
| **ST-GraphAD** | **0.657 ± 0.078** | **0.858 ± 0.021** | **0.791 ± 0.022** | **0.939 ± 0.037** |

**Absolute Improvements (mean to mean):**
- AUC-ROC: +4.5 percentage points (0.657 − 0.612 = 0.045)
- F1: +2.2 percentage points (0.858 − 0.836 = 0.022)
- Precision: +0.2 percentage points (0.791 − 0.789 = 0.002)
- Recall: +5.0 percentage points (0.939 − 0.889 = 0.050)

**Variance Note:** ST-GraphAD exhibits substantially higher variance than GCN (AUC std 0.078 vs 0.030; recall std 0.037 vs 0.023), indicating sensitivity to initialization. The observed improvement is a measured mean difference; no statistical significance test was performed.

---

## Per-Seed Breakdown

| Seed | GCN AUC | GCN F1 | GCN Prec | GCN Rec | ST-GNN AUC | ST-GNN F1 | ST-GNN Prec | ST-GNN Rec |
|------|---------|--------|----------|---------|------------|-----------|-------------|------------|
| 42   | 0.5727  | 0.8424 | 0.7820   | 0.9130  | 0.6782     | 0.8288    | 0.7898      | 0.8718     |
| 123  | 0.6250  | 0.8499 | 0.8055   | 0.8995  | 0.5581     | 0.8632    | 0.7816      | 0.9639     |
| 456  | 0.6232  | 0.8216 | 0.7963   | 0.8486  | 0.5945     | 0.8453    | 0.7578      | 0.9555     |
| 789  | 0.5848  | 0.8186 | 0.7632   | 0.8827  | 0.6703     | 0.8590    | 0.8007      | 0.9265     |
| 999  | 0.6547  | 0.8474 | 0.7980   | 0.9034  | 0.7832     | 0.8932    | 0.8237      | 0.9755     |

All five seeds completed successfully.

---

## Confusion Matrices (Test Set, Aggregated Across All Test Sequences)

### GCN-only (seed=42 reference)
```
[[ 99, 438],
 [187, 1365]]
```
- True Negatives: 99, False Positives: 438
- False Negatives: 187, True Positives: 1365
- Total Positive: 1552, Total Negative: 537

### ST-GraphAD (seed=42 reference)
```
[[ 71, 466],
 [177, 1375]]
```
- True Negatives: 71, False Positives: 466
- False Negatives: 177, True Positives: 1375
- Total Positive: 1552, Total Negative: 537

*Note: Confusion matrices vary by seed; the above are the seed=42 reference (included in the five-seed aggregate).*

---

## Graph-Level Results (Degenerate — Not Primary Task)

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| GCN-only | 0.00 | 1.00 | 1.00 | 1.00 |
| ST-GraphAD | 0.00 | 0.00 | 0.00 | 0.00 |

**Explanation:** Graph label = majority vote over nodes in last window. With ~88% attack nodes, almost every graph snapshot receives label "Attack". AUC=0.00 reflects single-class ground truth; F1/Precision/Recall=1.00 reflects trivial majority prediction. **Graph-level metrics are not meaningful for this task formulation and are not used for model selection.**

---

## Computational Complexity (Configuration C)

| Model | Parameters | Train Time (50 ep, seed=42) | Peak CPU Memory |
|-------|------------|-----------------------------|-----------------|
| GCN-only | ~180K | ~10 sec | ~300 MB |
| ST-GraphAD | ~4.2M | ~86 sec | ~1.2 GB |

- ST-GraphAD: ~23× more parameters, ~4× memory, ~8.6× slower (50 epochs)
- Exploratory run with Configuration D (256 hidden, 4 GCN layers, 200 epochs) produced GCN ~114 sec, ST-GraphAD ~593 sec — **different configuration, not headline**

---

## Training Dynamics (Representative)

| Model | Best Val Epoch | Best Val Node AUC | Final Test Node AUC |
|-------|----------------|-------------------|---------------------|
| GCN-only | 13–21 (varies by seed) | ~0.97–0.97 | 0.57–0.65 |
| ST-GraphAD | 2–48 (varies by seed) | ~0.98–0.99 | 0.56–0.78 |

Early stopping monitors **node-level validation AUC-ROC** (not graph-level). Checkpoint selection uses the same metric.

---

## Source Files

All results traceable to:
- `results/seed_42_results.json` through `results/seed_999_results.json`
- `results/multi_seed_aggregate.json` (computed mean±std)
- `results/evaluation_results.json` (Configuration D — 256 hidden, 4 layers, 200 epochs, seed=42 — **not the headline**)

---

## Limitations Acknowledged

1. Pseudo-node construction (protocol-service-state tuples, not real IPs)
2. Pseudo-temporal ordering (1000-flow windows from CSV row order)
3. Binary node labels only (no per-attack-category evaluation)
4. Higher variance of ST-GraphAD (AUC std 0.078 vs 0.030)
5. CPU-only training (no GPU for larger searches)
6. No quantitative attention interpretability analysis
7. No ablation experiments executed
8. Single dataset (UNSW-NB15 only)