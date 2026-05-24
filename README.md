# CAMIE: Context-Aware Subgraph Explanations for Multi-Task GNNs

Welcome to the anonymous official implementation codebase for **CAMIE**, a post-hoc explanation framework for **multi-task molecular GNNs**.

Unlike context-agnostic explainers that assign a single fixed explanation to a molecule, CAMIE estimates **assay-specific motif importance** for each **molecule--assay pair**. The key idea is that, in multi-task molecular prediction, the same molecule may rely on different substructures depending on the queried biological assay. CAMIE captures this by conditioning motif scoring on assay context and distilling assay-specific motif-removal responses from a frozen multi-task predictor.

---

## Overview

CAMIE consists of four stages:

1. **Frozen multi-task prediction backbone**  
   A pretrained shared GNN encodes each molecule and produces assay-specific outputs.

2. **Motif decomposition and representation extraction**  
   Each molecule is decomposed into chemically meaningful candidate motifs using BRICS. Graph, motif, and assay-context representations are then constructed.

3. **Context-aware motif scoring**  
   CAMIE learns a shared scorer over `(graph, motif, assay)` tuples using pseudo targets derived from assay-specific motif-removal responses.

4. **Top-k motif explanation**  
   At inference time, CAMIE ranks motifs for a queried assay and returns the top-k motifs as the explanation.

![CAMIE overview](supp/overall_camie.png)

---

## Repository Structure

```directory
CAMIE/
├── baselines/                  # Saliency and aggregation baseline models
│   ├── common/                 # Shared utilities, aggregation and metrics
│   └── models/
│       └── gradient/           # Gradient-based saliency (SA) baseline
├── datasets/                   # Preprocessing and dataset classes
│   ├── preprocess_toxcast.py   # Maps biological context and annotations to ToxCast
│   ├── toxcast_dataset.py      # Preprocesses and splits ToxCast deepchem dataset
│   ├── toxcast_graph_dataset.py# PyG InMemoryDataset definition for ToxCast
│   └── utils.py                # Graph helper functions
├── evaluate/                   # Evaluation scripts
│   └── compute_fidelity_f1_single.py # Computes F1-Fidelity under S1/S2 masking
├── models/
│   └── gnn/                    # Backbone GNN definitions and extraction script
│       ├── gnn.py              # PyG GIN/GCN architectures
│       ├── run_toxcast.py      # GNN backbone training & evaluation
│       └── extract_toxcast_emb.py # Extracts GNN graph and node embeddings
├── subgraph/                   # Subgraph & context processing
│   ├── context/
│   │   └── hierarchical_dataset.py # Generates motif-assay tables
│   ├── motifs/
│   │   ├── extract.py          # Decomposes SMILES into motifs (BRICS/Tree)
│   │   └── build_motif_emb.py  # Builds motif embeddings via node pooling
│   └── scoring/                # Joint Scorer implementation
│       ├── scoring_dataset.py  # Prepares pseudo-targets (masking difference)
│       ├── train_mse.py        # Trains the JointMLPScorer model
│       ├── scores_mse.py       # Predicts motif joint scores
│       └── decomposition_mse.py# Hard S1/S2 partition of motifs
├── utils/
│   ├── chemutils.py            # RDKit decomposition helpers
│   └── utils.py                # Generic utilities
│
├──
└── environment.yml             # Conda environment configuration
```

---

## Environment Setup

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate SCAR
```

If your environment name is different, replace `SCAR` with the actual name defined in `environment.yml`.

---

## Dataset

We use the **ToxCast multi-task molecular assay benchmark**.

### Main statistics

- **# molecules**: 8,578
- **# assays/tasks**: 30
- **# observed molecule--assay pairs**: 59,131
- **# motif--assay scoring instances**: 413,114

### Split statistics

- **train pairs**: 49,110
- **valid pairs**: 4,681
- **test pairs**: 5,340

- **train molecules**: 6,862
- **valid molecules**: 858
- **test molecules**: 858

### Additional statistics

- **overall positive ratio**: 0.116
- **pairs per assay (min / median / max)**: 97 / 502 / 7,934
- **assays per molecule (min / median / max)**: 1 / 3 / 30
- **avg. motifs per molecule--assay pair**: 6.99
- **avg. motifs per molecule**: 7.61

We use **Bemis--Murcko scaffold splitting** with a **7:1:2** ratio and **BRICS** motif decomposition.

---

## Baselines

We compare CAMIE against representative post-hoc explanation baselines.

### Optimization-based
- **eXEL-group**
- **eXEL-lasso**

### Gradient-based
- **SA**
- **GBP**
- **GradCAM**

### Perturbation-based
- **GNNExplainer**
- **PGExplainer**

For gradient- and perturbation-based baselines, node- or edge-level scores are converted to motif-level scores using a shared aggregation protocol.

---

## Execution Pipeline

### Step 1. Train the CAMIE scorer

```bash
python -m subgraph.scoring.train_mse \
  --scoring_table_dir assets/scoring/scoring_dataset/motif_context_scoring_table_seed0.csv \
  --out_dir assets/scoring/joint_mlp/seed0/joint_ckpt \
  --model_type joint_mlp \
  --seed 0
```

### Step 2. Score motif--assay tuples

```bash
python -m subgraph.scoring.scores_mse \
  --scoring_table_dir assets/scoring/scoring_dataset/motif_context_scoring_table_seed0.csv \
  --joint_ckpt_dir assets/scoring/joint_mlp/seed0/joint_ckpt \
  --out_dir assets/scoring/joint_mlp/seed0/scores \
  --model_type joint_mlp \
  --seed 0
```

### Step 3. Decompose into top-k motifs

```bash
python -m subgraph.scoring.decomposition_mse \
  --scored_table_dir assets/scoring/joint_mlp/seed0/scores/scored_table_joint_mlp_seed0.csv \
  --score_col score_joint_mlp \
  --rule_name joint_mlp \
  --out_dir assets/scoring/joint_mlp/seed0/decomposition \
  --seed 0
```

### Step 4. Compute Fidelity F1

```bash
python -m baselines.evaluate.compute_fidelity_f1_single \
  --decomp_csv assets/scoring/decomposition/ablation/mse/motif_decomposition_table_seed0.csv \
  --out_dir assets/baselines/fidelity_f1_single/seed0/mse \
  --model mse \
  --seed 0
```
---

## Main Outputs

### Scored motif table
```text
assets/scoring/joint_mlp/seed0/scores/scored_table_joint_mlp_seed0.csv
```

### Motif decomposition table
```text
assets/scoring/decomposition/ablation/mse/motif_decomposition_table_seed0.csv
```

### Fidelity summaries
```text
assets/baselines/fidelity_f1_single/seed0/mse/compact_fidelity_f1_summary_mse.csv
```

---

## Reproducibility Notes

- All main results are reported over **10 random seeds (0--9)**.
- Main comparisons use **top-k = 2**.
- Probabilities are thresholded at **0.5** for F1-based fidelity evaluation.
- Gradient- and perturbation-based baselines use the same frozen multi-task GNN backbone.

---

## Notes

- CAMIE is a **post-hoc** explainer and does **not** retrain the original multi-task predictor.
- The pseudo target is derived from the predictor's assay-specific motif-removal response, so CAMIE explanations depend on the behavior of the frozen backbone.
- Explanation resolution is limited by the predefined motif candidates produced by BRICS decomposition.
