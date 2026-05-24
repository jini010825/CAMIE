# CAMIE: Context-conditioned Explanatory Motif Identification under Biological Assays

Welcome to the anonymous official implementation codebase for **CAMIE** (Context-conditioned Explanatory Motif Identification under Biological Assays). This framework identifies and evaluates context-dependent molecular subgraphs (motifs) that explain Graph Neural Network (GNN) predictions under multi-task biological assay contexts.

---

## Overview

Molecular graph neural networks (GNNs) are widely used for predicting chemical properties and toxicities. However, molecular subgraphs can exhibit context-dependent importances. A motif explaining chemical toxicity in one biological assay might be entirely neutral or irrelevant in another assay. 

**CAMIE** addresses this by modeling **motif importance under specific assay contexts**:
1. **Backbone GNN Training**: Trains multi-task molecular GNNs (GIN/GCN) on toxicological datasets (e.g., ToxCast).
2. **Hierarchical Motif Decomposition**: Decomposes molecules into chemically valid subgraphs (motifs) using BRICS or Junction Tree decomposition.
3. **Assay Context Embeddings**: Generates semantic embeddings of assay text contexts (e.g., target genes, biology processes).
4. **Joint MLP Scorer**: Trains a joint scorer (`JointMLPScorer`) using pseudo-importance labels (predictive logit/probability differences under subgraph masking) to capture non-linear interactions between motif subgraphs, molecular graphs, and assay contexts.
5. **Subgraph Decomposition & Evaluation**: Partitions molecular graphs into active explanation subgraphs ($S_2$) and background fragments ($S_1$) to measure F1-Fidelity.

```
                  ┌────────────────────────┐
                  │ Molecular Graph (zg)   │
                  └───────────┬────────────┘
                              ▼
┌──────────────┐  ┌────────────────────────┐  ┌──────────────────────┐
│ Motif (zm)   │─►│     JointMLPScorer     │◄─│  Assay Context (za)  │
└──────────────┘  └────────────────────────┘  └──────────────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │ Predicted Motif Importance │
                └────────────────────────────┘
```

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
└── environment.yml             # Conda environment configuration
```

---

## Environment Setup

Set up the Conda environment using the provided `environment.yml` configuration:

```bash
# Clone the repository and navigate inside
cd CAMIE

# Create the Conda environment
conda env create -f environment.yml

# Activate the environment
conda activate base
```

---

## Execution Pipeline

Follow these steps sequentially to preprocess the data, train the backbone, extract motifs, train the scorer, and evaluate fidelity:

### Step 1: Data Preprocessing
Preprocess the ToxCast raw data, align annotations, and map biological assay metadata:
```bash
python datasets/preprocess_toxcast.py
```

Create dataset splits (Train/Valid/Test) and convert them to PyG graph formats:
```bash
python datasets/toxcast_dataset.py --split_mode scaffold
```

### Step 2: Train Backbone GNN
Train a multi-task GIN (or GCN) backbone on the preprocessed ToxCast graph dataset:
```bash
python models/gnn/run_toxcast.py --model gin --epochs 100 --seed 0
```

Extract the learned node-level and graph-level molecular embeddings from the trained model checkpoint:
```bash
python models/gnn/extract_toxcast_emb.py --model gin --seed 0 --split_mode all
```

### Step 3: Hierarchical Motif Extraction & Embedding
Decompose target chemical SMILES into chemical motifs using BRICS decomposition:
```bash
python subgraph/motifs/extract.py --method brics
```

Construct the aligned motif embeddings by pooling the corresponding backbone GNN node embeddings:
```bash
python subgraph/motifs/build_motif_emb.py --split_mode all --seed 0
```

### Step 4: Build Hierarchical Tables
Map the hierarchical relationship tables and connect the assay context embeddings:
```bash
python subgraph/context/hierarchical_dataset.py --decomposition_method brics
```

### Step 5: Train Joint Scorer & Generate Explanations
1. **Prepare Scoring Dataset**: Perform GNN inference under single-motif zero-out masking to compute pseudo-targets (logit and probability changes):
   ```bash
   python subgraph/scoring/scoring_dataset.py --model gin --seed 0
   ```

2. **Train Scorer**: Train the `JointMLPScorer` model to fit the pseudo-target distribution based on molecular, motif, and assay context features:
   ```bash
   python subgraph/scoring/train_mse.py --seed 0 --epochs 50
   ```

3. **Predict Motif Scores**: Predict the joint motif-context importance scores for all candidate pairs:
   ```bash
   python subgraph/scoring/scores_mse.py --seed 0 --split_mode all
   ```

4. **Hard Decomposition**: Perform S1/S2 hard decomposition based on the predicted scorer scores:
   ```bash
   python subgraph/scoring/decomposition_mse.py --seed 0 --k_values 2
   ```

---

## Evaluation & Verification

To evaluate the explanatory quality of the identified explanation subgraphs ($S_2$ active components) versus background fragments ($S_1$), run the F1-Fidelity evaluator. It applies masking to the chemical graph and computes GNN classification prediction drops on the target biological assays:

```bash
python evaluate/compute_fidelity_f1_single.py \
    --decomp_csv assets/scoring/decomposition/ablation/mse/motif_decomposition_table_seed0.csv \
    --seed 0 \
    --k 2
```

Results (including rule-level, split-wise, and assay-wise F1-Fidelity reports) will be saved in `assets/baselines/fidelity_f1_single/`.

---

## Baselines
To compare the joint scorer against gradient-based Saliency Attribution (SA) baselines, run:
```bash
python baselines/models/gradient/saliency.py --seed 0 --abs_grad
```
