# REMAINING LIMITATIONS — ST-GraphAD

This document honestly enumerates all limitations that remain after the corrected experimental protocol. None are hidden or downplayed.

---

## 1. Graph Construction Limitations

### 1.1 Pseudo-Node Construction (Severity: HIGH)
- **What**: Nodes are constructed from `(protocol, service, role)` tuples, not real IP addresses.
- **Why**: UNSW-NB15 CSV release omits explicit source/destination IPs.
- **Consequence**: Multiple real hosts sharing the same tuple collapse to one pseudo-node. The graph topology is a proxy for true communication structure, not ground truth.
- **Mitigation**: Parse PCAP files to extract real IPs/ports; use heterogeneous graphs with distinct node types (IP, port, protocol, service).

### 1.2 Pseudo-Temporal Ordering (Severity: HIGH)
- **What**: Time windows are created by splitting flows into chunks of 1000 rows by CSV order.
- **Why**: The CSV lacks usable timestamps (the `timestamp` field is missing or unreliable).
- **Consequence**: Attack campaigns may be split across windows or unrelated flows merged; temporal attention operates on an arbitrary discretization.
- **Mitigation**: Use real timestamps if available; adopt continuous-time temporal GNNs (TGN, CAW) that don't require fixed windows.

### 1.3 Fixed 1000-Flow Windows (Severity: MEDIUM)
- **What**: Window size is a hardcoded hyperparameter with no domain justification.
- **Consequence**: May not align with natural attack stage durations.
- **Mitigation**: Ablate window size; explore adaptive/event-driven windowing.

---

## 2. Evaluation Limitations

### 2.1 No Per-Attack-Category Node-Level Evaluation (Severity: MEDIUM)
- **What**: Node labels are binary (attack vs. normal); the `attack_cat` field is lost during node aggregation.
- **Consequence**: Cannot claim ST-GraphAD is better/worse on specific attack types (Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoors, Shellcode, Worms).
- **Mitigation**: Implement multi-class node classification; propagate attack_cat to nodes; report category-wise AUC/F1.

### 2.2 Higher Variance of ST-GraphAD (Severity: MEDIUM)
- **What**: ST-GraphAD AUC std = 0.078 vs GCN std = 0.030 (2.6×); Recall std = 0.037 vs 0.023.
- **Consequence**: The +4.5 pp mean AUC improvement is not statistically established; some seeds show ST-GraphAD underperforming GCN.
- **Mitigation**: Run more seeds (e.g., 20+); apply paired significance testing (Wilcoxon, McNemar); consider weight averaging or ensemble.

### 2.3 Graph-Level Metrics Degenerate (Severity: HIGH for graph-level; LOW for primary task)
- **What**: ~88% attack nodes → majority-vote graph label is almost always "Attack".
- **Consequence**: Graph AUC = 0.00, F1 = 1.00 (trivial). Not a model failure — a label construction artifact.
- **Mitigation**: Use graph-level regression (predict attack proportion) or discard graph-level task. Primary task remains node-level.

---

## 3. Interpretability Limitations

### 3.1 No Quantitative Attention Analysis (Severity: MEDIUM)
- **What**: Attention weights were not systematically analyzed (entropy, head specialization, temporal focus patterns).
- **Consequence**: Claims about "what the attention heads learn" are qualitative design rationales, not experimental findings.
- **Mitigation**: Compute attention entropy per head; visualize attention maps; ablate individual heads.

### 3.2 No Case-Level Evidence for Multi-Stage Detection (Severity: MEDIUM)
- **What**: The paper discusses "reconnaissance → exploitation" and "lateral movement" as design motivations.
- **Consequence**: No concrete examples or attention-based evidence showing ST-GraphAD detects these specific patterns.
- **Mitigation**: Curate case studies from test set; visualize attention for known multi-stage attacks; compare with ground-truth attack timelines if available.

---

## 4. Experimental Scope Limitations

### 4.1 No Ablation Experiments Executed (Severity: MEDIUM)
- **What**: Planned variants (1 vs 4 vs 8 heads; 1/2/3 temporal layers; L ∈ {3,5,10}; d ∈ {64,128,256}) were not run.
- **Consequence**: No empirical evidence for architectural choices beyond the single configuration.
- **Mitigation**: Execute ablation with 5 seeds each; report mean±std; include in next revision.

### 4.2 Single Dataset (Severity: HIGH for generalization)
- **What**: Only UNSW-NB15 evaluated.
- **Consequence**: Results may not transfer to CICIDS2017, CSE-CIC-IDS2018, or enterprise traffic.
- **Mitigation**: Evaluate on at least one additional NIDS dataset.

### 4.3 CPU-Only Training (Severity: MEDIUM)
- **What**: MPS/CUDA errors prevented GPU training; limited hyperparameter search and larger model exploration.
- **Consequence**: Could not test larger hidden dimensions, more layers, longer sequences, or more seeds within practical time.
- **Mitigation**: Resolve GPU compatibility; enable larger-scale experiments.

### 4.4 No Statistical Significance Testing (Severity: MEDIUM)
- **What**: Mean±std reported but no confidence intervals, p-values, or paired tests.
- **Consequence**: Cannot formally reject null hypothesis that ST-GraphAD = GCN.
- **Mitigation**: Apply Wilcoxon signed-rank test on per-seed AUC differences; report 95% CIs.

---

## 5. Deployment / Operational Limitations

### 5.1 Sensitivity to Initialization (Severity: HIGH for production)
- **What**: ST-GraphAD's high variance means a single training run may underperform GCN.
- **Consequence**: Unreliable for production without ensembling or extensive seed search.
- **Mitigation**: Use ensemble of 5+ seeds; or find initialization/regularization that reduces variance.

### 5.2 Computational Cost (Severity: MEDIUM)
- **What**: ~8.6× slower than GCN (50 epochs); ~23× more parameters; ~4× memory.
- **Consequence**: May be prohibitive for real-time streaming on edge devices.
- **Mitigation**: Knowledge distillation; attention pruning; quantization; ONNX export.

### 5.3 Inductive Generalization Not Tested (Severity: LOW)
- **What**: The GCN encoder is theoretically inductive (can embed unseen nodes).
- **Consequence**: Not empirically validated on truly unseen pseudo-node combinations.
- **Mitigation**: Evaluate on test windows containing pseudo-nodes absent from training.

---

## 6. Data Leakage / Partition Independence Clarification

### 6.1 Flow-Level Partition Independence: CONFIRMED
- Training, validation, and test flows are disjoint by index (sanity check passed).
- No flow record appears in more than one partition.

### 6.2 Pseudo-Node Identifier Overlap: EXPECTED AND NOT LEAKAGE
- The same `(protocol, service, role)` tuple (and thus the same pseudo-node ID) can appear in multiple partitions.
- This is **not** data leakage — it reflects that different flows across time can involve the same protocol/service combination.
- The leakage control is that **preprocessing transformations are fit on training flows only**.

### 6.3 Preprocessing Leakage: MITIGATED
- LabelEncoder and StandardScaler are fit exclusively on the training partition (149,039 flows).
- Validation and test partitions are transformed using the training-fitted objects.
- No test labels used for training, checkpoint selection, scheduler, or early stopping.

---

## 7. Summary: What Is and Is Not Claimed

| Claim | Status |
|-------|--------|
| ST-GraphAD achieves higher mean node-level AUC than GCN on UNSW-NB15 under the corrected protocol | ✅ Supported by 5-seed experiment |
| The improvement is 4.5 pp absolute (mean to mean) | ✅ 0.657 − 0.612 = 0.045 |
| The improvement is statistically significant | ❌ Not tested; higher variance cautions against this |
| ST-GraphAD detects "reconnaissance → exploitation" sequences | ❌ Design rationale only; no case evidence |
| Attention heads specialize (recent/periodic/degree/global) | ❌ Not quantitatively evaluated |
| ST-GraphAD outperforms on Generic/Exploits/DoS/etc. individually | ❌ Not implemented |
| Results generalize to other datasets | ❌ Not tested |
| The pseudo-node graph equals real network topology | ❌ Explicitly false; known proxy |
| Graph-level AUC 0.00 indicates model failure | ❌ Dataset artifact (88% attack nodes) |

---

## 8. Path to Publication-Ready

Before considering this work publication-ready, the following **must** be completed:

1. [ ] Execute ablation study (5 seeds × each variant)
2. [ ] Implement per-attack-category node evaluation
3. [ ] Perform statistical significance testing (Wilcoxon on 5-seed paired differences)
4. [ ] Add at least one additional dataset (CICIDS2017 or CSE-CIC-IDS2018)
5. [ ] Resolve GPU training for larger-scale exploration
6. [ ] Quantitative attention interpretability analysis
7. [ ] Regenerate all figures from the final Configuration C experiment
8. [ ] Consider ensembling or variance-reduction techniques for ST-GraphAD

Until these are done, the work should be presented as: *"ST-GraphAD shows a promising mean AUC improvement over static GCN under a rigorous protocol, but the improvement is not yet statistically established and requires further validation."*

---

*Last updated: 2025-08-28. All limitations correspond to the Configuration C final experiment (5 seeds, 128 hidden, 3 layers, 50 epochs).*