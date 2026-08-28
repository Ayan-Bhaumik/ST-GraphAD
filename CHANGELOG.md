# CHANGELOG: Scientific Corrections to ST-GraphAD

This document records every material correction made to the codebase, experimental results, and manuscript to ensure scientific accuracy. No inflated or fabricated values remain.

---

## 2025-08-28: Complete Experimental Overhaul

### 1. Root Cause: Flawed Original Experimental Protocol (Configuration A)

The originally reported results (AUC 0.91 / 0.88, F1 0.88 / 0.84) were obtained under an invalid setup:

| Flaw | Description | Consequence |
|------|-------------|-------------|
| Graph-level early stopping | `ReduceLROnPlateau` and checkpoint selection monitored validation **graph AUC** (majority vote) | Graph AUC is 0.00 for both models due to class imbalance; no meaningful signal |
| Test set used for validation | The official UNSW-NB15 test CSV was used for early stopping / model selection | Data leakage; optimistically biased results |
| Preprocessing on combined data | `LabelEncoder` and `StandardScaler` were fit on train+test combined | Feature statistics leaked from test into training |
| Single seed, no variance | Only seed=42 reported; no standard deviation | Unreliable; masks sensitivity to initialization |
| Architecture mismatch | Paper stated 128 hidden / 3 GCN layers; some runs used 256 / 4 layers | Irreproducible configuration |

### 2. Corrected Final Experimental Protocol (Configuration C — Authoritative)

| Aspect | Corrected Value |
|--------|-----------------|
| Hidden channels | 128 |
| GCN layers | 3 |
| Temporal attention layers | 2 |
| Attention heads | 4 |
| Sequence length | 5 |
| Dropout | 0.5 |
| Learning rate | 1×10⁻³ |
| Weight decay | 5×10⁻⁴ |
| Max epochs | 50 |
| Early stopping patience | 10 (on **node-level validation AUC-ROC**) |
| Seeds | 42, 123, 456, 789, 999 |
| Dataset split | Training CSV → 85% train / 15% val (stratified); Official test CSV held out |
| Preprocessing | LabelEncoder & StandardScaler fit **only on training partition**; applied to val/test |
| Primary metric | Node-level AUC-ROC, F1, Precision, Recall (aggregated across all test sequences) |
| Graph-level | Reported with explicit degeneracy caveat; **never** used for model selection |

### 3. Corrected Headline Results (Five-Seed Aggregate)

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| **GCN-only** | 0.612 ± 0.030 | 0.836 ± 0.013 | 0.789 ± 0.015 | 0.889 ± 0.023 |
| **ST-GraphAD** | 0.657 ± 0.078 | 0.858 ± 0.021 | 0.791 ± 0.022 | 0.939 ± 0.037 |

**Absolute improvements (mean to mean):**
- AUC-ROC: **+4.5 pp** (0.657 − 0.612 = 0.045)
- F1: **+2.2 pp** (0.858 − 0.836 = 0.022)
- Precision: **+0.2 pp** (0.791 − 0.789 = 0.002)
- Recall: **+5.0 pp** (0.939 − 0.889 = 0.050)

**Variance caveat:** ST-GraphAD shows substantially higher variance (AUC std 0.078 vs 0.030; recall std 0.037 vs 0.023). The improvement is reported as a measured mean difference, **not** as a statistically established effect. No significance test was performed.

### 4. Reference Single-Seed Result (seed=42, 50 epochs)

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| GCN-only | 0.573 | 0.842 | 0.782 | 0.913 |
| ST-GraphAD | 0.678 | 0.829 | 0.790 | 0.872 |

*This row is included in the five-seed aggregate and provided for traceability only. The five-seed aggregate is the authoritative result.*

### 5. Files Updated

| File | Changes |
|------|---------|
| `src/data_loader.py` | Proper 85/15 train/val split from training CSV; encoders/scalers fit on training only; official test untouched |
| `src/train.py` | Node-level validation AUC for early stopping, `ReduceLROnPlateau`, checkpoint selection; multi-seed orchestration; per-graph evaluation for variable-sized graphs |
| `src/models.py` | Fixed GCN forward to handle sequence input consistently with ST-GNN |
| `main.py` | Added `--multi-seed` flag; sanity checks; consistent config defaults |
| `manuscript_full_text.txt` | **Complete rewrite** to publication standard; only Configuration C results; all claims traceable to saved outputs; unsupported interpretability claims removed/qualified; ablation marked as planned/not executed |
| `RESEARCH.md` | Synchronized with manuscript: abstract, hyperparameters, results tables, computational complexity, discussion, limitations, conclusion |
| `results/report.md` | Replaced with Configuration C values; removed 256/4-layer Configuration D numbers; clearly separated five-seed aggregate from single-seed reference |
| `README.md` | Complete rewrite matching final manuscript: key results, architecture, usage, configuration, limitations, reproducibility |

### 6. Numerical Corrections Made

| Old Claim (Configuration A) | Corrected Value (Configuration C) | Note |
|----------------------------|-----------------------------------|------|
| ST-GraphAD AUC 0.91 | 0.657 ± 0.078 (5-seed) | Old was single-seed, graph-level validation, leaked |
| ST-GraphAD F1 0.88 | 0.858 ± 0.021 | |
| GCN AUC 0.87 | 0.612 ± 0.030 | |
| GCN F1 0.84 | 0.836 ± 0.013 | |
| "4.6% AUC improvement" | "+4.5 percentage points absolute" | Correct phrasing for absolute difference |
| "4.8% F1 improvement" | "+2.2 percentage points absolute" | |
| "4.9% recall improvement" | "+5.0 percentage points absolute" | 0.939 − 0.889 = 0.050 |
| Graph-level AUC 0.00 = "model failure" | Graph-level AUC 0.00 = "dataset artifact (88% attack nodes)" | Explicit caveat |
| "No data leakage" (unconditional) | "Preprocessing fitted on training partition only; pseudo-node identifiers may recur across partitions but this is not flow-level leakage" | Precise language |
| "Head 1 attends to recent windows..." | "Head specialization not quantitatively evaluated" | Removed unsupported claims |
| "Reconnaissance → Exploitation detected" | "Design rationale; no case-level evidence" | Qualified |
| Training time: 33s / 418s (mixed configs) | 10s / 86s (seed=42, 50 epochs, 128 hidden) | Labeled with configuration |
| Computational table: 33s / 418s | 10s / 86s + note about 256/4-layer exploratory run | Separated configs |

### 7. Removed / Retracted Claims

- ❌ "0.91 AUC / 0.88 F1" (inflated single-seed, flawed protocol)
- ❌ "Statistically significant" / "outperforms" without qualification
- ❌ Attention head specialization claims (Head 1/2/3/4 assignments)
- ❌ Specific attack-stage detection claims ("reconnaissance → exploitation")
- ❌ Ablation numerical results (never executed)
- ❌ Per-attack-category performance claims
- ❌ "No data leakage" unconditional statement
- ❌ Mixed-configuration computational numbers presented as single values
- ❌ 256 hidden / 4 GCN layers as the paper's architecture

### 8. Added / Strengthened Transparency

- ✅ Five-seed aggregate as authoritative headline
- ✅ Explicit variance caveat (ST-GraphAD std 2.6× GCN)
- ✅ Separation of single-seed reference from multi-seed aggregate
- ✅ Planned ablation section with no fabricated numbers
- ✅ Partition independence vs. pseudo-node identifier overlap distinction
- ✅ Graph-level degeneracy explicitly explained
- ✅ All hyperparameters match final experiment in every section
- ✅ Limitations table with fixed/unfixed status
- ✅ Reproducibility command in README

---

## Remaining Gaps (Honestly Documented)

1. **Ablation study**: Not executed; planned variants documented in §6.4
2. **Per-attack-category evaluation**: Not implemented; flagged in §6.7 and §8
3. **Attention interpretability**: No quantitative analysis; flagged in §7.2 and §8
4. **Statistical significance testing**: Not performed; noted in §6.2 and §9.2
5. **Real IP graphs**: Pseudo-node limitation remains; future work item
6. **Multi-dataset evaluation**: Only UNSW-NB15 tested
7. **GPU training**: CPU-only; larger searches infeasible

---

## Traceability

Every number in the final manuscript and README can be traced to:

- `results/seed_42_results.json` through `results/seed_999_results.json` (5 seeds × 2 models)
- `results/multi_seed_aggregate.json` (aggregated mean±std)
- `results/evaluation_results.json` (single-seed Configuration D — **not** the headline)
- `results/report.md` (summary)

No value was estimated, extrapolated, or silently reconciled. If two runs produced different numbers (e.g., Configuration D vs Configuration C), they are kept separate and labeled by configuration.