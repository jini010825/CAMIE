#%%
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets.toxcast_graph_dataset import ToxCastSharedDataset
from models.gnn.gnn import GNN_graph
from subgraph.scoring.utils import zero_mask_atom_indices, collect_atom_indices_for_set


# =========================================================
# args
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Compute Fidelity_F1 from one decomposition table")

    # data / backbone model
    p.add_argument("--toxcast_all_csv", type=str, default="datasets/raw_data/chem_dataset/toxcast_all.csv")
    p.add_argument("--assay_table_csv", type=str, default="datasets/processed/toxcast/hierarchical/assay_table.csv")
    p.add_argument("--motif_mapping_pkl", type=str, default="assets/motif_split/smiles_to_hierarchical_mapping.pkl")
    
    p.add_argument("--ckpt_dir", type=str, default="assets/toxcast_gnn/ckpt")
    p.add_argument("--backbone_model", type=str, default="gin", choices=["gin", "gcn"])
    p.add_argument("--num_layer", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=300)
    p.add_argument("--drop_ratio", type=float, default=0.5)

    # decomposition input
    p.add_argument("--decomp_csv", type=str, required=True)

    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--model",
        type=str,
        default="mse",
    )
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_num_motifs", type=int, default=1)

    p.add_argument("--out_dir", type=str, default="assets/baselines/fidelity_f1_single")

    return p.parse_args()

# =========================================================
# forward utils
# =========================================================
@torch.no_grad()
def forward_one(model, data, assay_idx: int, device):
    model.eval()
    data = data.to(device)

    logits = model(data).view(-1)   # [T]
    probs = torch.sigmoid(logits)

    logit_c = float(logits[assay_idx].item())
    prob_c = float(probs[assay_idx].item())
    return logit_c, prob_c


# =========================================================
# load decomposition
# =========================================================
def load_decomp(csv_path: str, k: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if k is not None and "k" in df.columns:
        df = df[df["k"] == k].copy()
    return df


# =========================================================
# metric helpers
# =========================================================
def filter_pair_df_by_min_num_motifs(pair_df: pd.DataFrame, min_num_motifs: int) -> pd.DataFrame:
    if min_num_motifs <= 1:
        return pair_df.copy()
    return pair_df[pair_df["num_motifs"] >= min_num_motifs].copy()


def safe_f1(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(f1_score(y_true, y_pred, zero_division=0))


def build_f1_rule_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for rule in sorted(pair_df["rule"].unique()):
        sub = pair_df[pair_df["rule"] == rule].copy()

        y_true = sub["label"].astype(int).values
        y_orig = sub["pred_orig"].astype(int).values
        y_s2 = sub["pred_masked_s2"].astype(int).values
        y_s1 = sub["pred_masked_s1"].astype(int).values

        f1_orig = safe_f1(y_true, y_orig)
        f1_s2 = safe_f1(y_true, y_s2)
        f1_s1 = safe_f1(y_true, y_s1)

        rows.append({
            "rule": rule,
            "k": int(sub["k"].iloc[0]) if len(sub) > 0 else np.nan,
            "n_pairs": len(sub),
            "f1_orig": f1_orig,
            "f1_masked_s2": f1_s2,
            "fidelity_f1_s2": f1_orig - f1_s2 if not np.isnan(f1_orig) and not np.isnan(f1_s2) else np.nan,
            "f1_masked_s1": f1_s1,
            "fidelity_f1_s1": f1_orig - f1_s1 if not np.isnan(f1_orig) and not np.isnan(f1_s1) else np.nan,
        })

    return pd.DataFrame(rows)


def build_f1_split_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for rule in sorted(pair_df["rule"].unique()):
        for split in sorted(pair_df["split"].unique()):
            sub = pair_df[(pair_df["rule"] == rule) & (pair_df["split"] == split)].copy()

            y_true = sub["label"].astype(int).values
            y_orig = sub["pred_orig"].astype(int).values
            y_s2 = sub["pred_masked_s2"].astype(int).values
            y_s1 = sub["pred_masked_s1"].astype(int).values

            f1_orig = safe_f1(y_true, y_orig)
            f1_s2 = safe_f1(y_true, y_s2)
            f1_s1 = safe_f1(y_true, y_s1)

            rows.append({
                "rule": rule,
                "split": split,
                "k": int(sub["k"].iloc[0]) if len(sub) > 0 else np.nan,
                "n_pairs": len(sub),
                "f1_orig": f1_orig,
                "f1_masked_s2": f1_s2,
                "fidelity_f1_s2": f1_orig - f1_s2 if not np.isnan(f1_orig) and not np.isnan(f1_s2) else np.nan,
                "f1_masked_s1": f1_s1,
                "fidelity_f1_s1": f1_orig - f1_s1 if not np.isnan(f1_orig) and not np.isnan(f1_s1) else np.nan,
            })

    return pd.DataFrame(rows)


def build_f1_assay_summary(pair_df: pd.DataFrame, min_pairs: int = 5) -> pd.DataFrame:
    rows = []

    for rule in sorted(pair_df["rule"].unique()):
        for assay in sorted(pair_df["assay"].unique()):
            sub = pair_df[(pair_df["rule"] == rule) & (pair_df["assay"] == assay)].copy()

            if len(sub) < min_pairs:
                continue

            y_true = sub["label"].astype(int).values
            y_orig = sub["pred_orig"].astype(int).values
            y_s2 = sub["pred_masked_s2"].astype(int).values
            y_s1 = sub["pred_masked_s1"].astype(int).values

            f1_orig = safe_f1(y_true, y_orig)
            f1_s2 = safe_f1(y_true, y_s2)
            f1_s1 = safe_f1(y_true, y_s1)

            rows.append({
                "rule": rule,
                "assay": assay,
                "k": int(sub["k"].iloc[0]) if len(sub) > 0 else np.nan,
                "n_pairs": len(sub),
                "f1_orig": f1_orig,
                "f1_masked_s2": f1_s2,
                "fidelity_f1_s2": f1_orig - f1_s2 if not np.isnan(f1_orig) and not np.isnan(f1_s2) else np.nan,
                "f1_masked_s1": f1_s1,
                "fidelity_f1_s1": f1_orig - f1_s1 if not np.isnan(f1_orig) and not np.isnan(f1_s1) else np.nan,
            })

    return pd.DataFrame(rows)


# =========================================================
# main
# =========================================================
def main():
    args = get_args()

    decomp_csv_path = Path(args.decomp_csv)
    if not decomp_csv_path.exists():
        raise FileNotFoundError(f"decomp_csv not found: {decomp_csv_path}")

    out_dir = Path(args.out_dir)
    #out_dir = Path(args.out_dir) / f"seed{args.seed}" / f"{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # -------------------------------------------------
    # 1. dataset
    # -------------------------------------------------
    dataset = ToxCastSharedDataset(
        toxcast_all_csv=args.toxcast_all_csv,
        assay_table_csv=args.assay_table_csv,
    )
    assay_to_task_idx = {a: i for i, a in enumerate(dataset.task_names)}

    smiles_to_dataset_idx = {}
    for i in range(len(dataset)):
        d = dataset.get(i)
        smiles_to_dataset_idx[d.smiles] = i

    print(f"[INFO] dataset size: {len(dataset)}")
    print(f"[INFO] num tasks   : {len(dataset.task_names)}")

    # -------------------------------------------------
    # 2. backbone model
    # -------------------------------------------------
    model = GNN_graph(
        num_tasks=len(dataset.task_names),
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        gnn_type=args.backbone_model,
    ).to(device)

    ckpt_path = Path(args.ckpt_dir) / f"toxcast_shared_{args.backbone_model}_best_seed{args.seed}.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[INFO] loaded checkpoint: {ckpt_path}")

    # -------------------------------------------------
    # 3. decomposition + motif mapping
    # -------------------------------------------------
    decomp = load_decomp(args.decomp_csv, args.k)

    with open(args.motif_mapping_pkl, "rb") as f:
        motif_mapping = pickle.load(f)

    print(f"[INFO] loaded decomposition: {args.decomp_csv}")
    print(f"[INFO] decomposition rows after filter: {len(decomp)}")
    print(f"[INFO] rules: {sorted(decomp['rule'].unique())}")
    print(f"[INFO] k    : {args.k}")

    required_cols = [
        "molecule_id", "smiles", "assay", "split", "label",
        "rule", "k", "set_assignment", "motif_local_id"
    ]
    for c in required_cols:
        if c not in decomp.columns:
            raise ValueError(f"Required column missing in decomposition table: {c}")

    group_cols = ["molecule_id", "smiles", "assay", "split", "label", "rule", "k"]
    grouped = decomp.groupby(group_cols, sort=False, dropna=False)

    print(f"[INFO] number of (molecule, assay, rule, k) groups: {grouped.ngroups}")

    # -------------------------------------------------
    # 4. pair-level masking predictions
    # -------------------------------------------------
    pair_rows = []

    for group_key, g in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Computing Fidelity_F1 masking predictions",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
    ):
        molecule_id, smiles, assay, split, label, rule, k = group_key

        if smiles not in smiles_to_dataset_idx:
            continue
        if smiles not in motif_mapping:
            continue
        if assay not in assay_to_task_idx:
            continue

        dataset_idx = smiles_to_dataset_idx[smiles]
        assay_idx = assay_to_task_idx[assay]

        data = dataset.get(dataset_idx)
        motif_info = motif_mapping[smiles]

        orig_logit, orig_prob = forward_one(model, data, assay_idx, device)

        s2_atoms = collect_atom_indices_for_set(g, motif_info, "S2")
        s1_atoms = collect_atom_indices_for_set(g, motif_info, "S1")

        data_mask_s2 = zero_mask_atom_indices(data, s2_atoms)
        data_mask_s1 = zero_mask_atom_indices(data, s1_atoms)

        masked_s2_logit, masked_s2_prob = forward_one(model, data_mask_s2, assay_idx, device)
        masked_s1_logit, masked_s1_prob = forward_one(model, data_mask_s1, assay_idx, device)

        pred_orig = int(orig_prob >= args.threshold)
        pred_masked_s2 = int(masked_s2_prob >= args.threshold)
        pred_masked_s1 = int(masked_s1_prob >= args.threshold)

        pair_rows.append({
            "molecule_id": molecule_id,
            "smiles": smiles,
            "assay": assay,
            "split": split,
            "label": int(label),
            "rule": rule,
            "k": int(k),
            "num_motifs": int(len(g)),
            "s2_size": int((g["set_assignment"] == "S2").sum()),
            "s1_size": int((g["set_assignment"] == "S1").sum()),
            "num_s2_atoms_masked": int(len(s2_atoms)),
            "num_s1_atoms_masked": int(len(s1_atoms)),
            "orig_logit": float(orig_logit),
            "orig_prob": float(orig_prob),
            "masked_s2_logit": float(masked_s2_logit),
            "masked_s2_prob": float(masked_s2_prob),
            "masked_s1_logit": float(masked_s1_logit),
            "masked_s1_prob": float(masked_s1_prob),
            "pred_orig": pred_orig,
            "pred_masked_s2": pred_masked_s2,
            "pred_masked_s1": pred_masked_s1,
            "logit_diff_s2": float(abs(orig_logit - masked_s2_logit)),
            "logit_diff_s1": float(abs(orig_logit - masked_s1_logit)),
            "prob_diff_s2": float(abs(orig_prob - masked_s2_prob)),
            "prob_diff_s1": float(abs(orig_prob - masked_s1_prob)),
        })

    pair_df = pd.DataFrame(pair_rows)

    pair_csv = out_dir / f"fidelity_f1_pair_table_{args.model}.csv"
    pair_df.to_csv(pair_csv, index=False)
    print(f"[OK] saved pair-level masking predictions: {pair_csv}")

    summary_df = filter_pair_df_by_min_num_motifs(pair_df, args.min_num_motifs)

    print(f"[INFO] applied min_num_motifs >= {args.min_num_motifs}")
    print(f"[INFO] pair rows used for summary: {len(pair_df)} -> {len(summary_df)}")

    # -------------------------------------------------
    # 5. rule-level summary
    # -------------------------------------------------
    rule_summary = build_f1_rule_summary(summary_df)
    rule_csv = out_dir / f"fidelity_f1_rule_summary_{args.model}.csv"
    rule_summary.to_csv(rule_csv, index=False)

    print("\n=== Rule-level Fidelity_F1 ===")
    print(rule_summary)

    # -------------------------------------------------
    # 6. split-wise summary
    # -------------------------------------------------
    split_summary = build_f1_split_summary(summary_df)
    split_csv = out_dir / f"fidelity_f1_split_summary_{args.model}.csv"
    split_summary.to_csv(split_csv, index=False)

    print("\n=== Split-wise Fidelity_F1 ===")
    print(split_summary)

    # -------------------------------------------------
    # 7. assay-wise summary
    # -------------------------------------------------
    assay_summary = build_f1_assay_summary(summary_df, min_pairs=5)
    assay_csv = out_dir / f"fidelity_f1_assay_summary_{args.model}.csv"
    assay_summary.to_csv(assay_csv, index=False)

    print("\n=== Assay-wise Fidelity_F1 (min_pairs=5) ===")
    print(assay_summary.head())

    # -------------------------------------------------
    # 8. compact summary
    # -------------------------------------------------
    compact = rule_summary.copy()
    compact_csv = out_dir / f"compact_fidelity_f1_summary_{args.model}.csv"
    compact.to_csv(compact_csv, index=False)

    print("\n=== Compact Fidelity_F1 summary ===")
    print(compact)

    print("\n[OK] saved:")
    print(" -", pair_csv)
    print(" -", rule_csv)
    print(" -", split_csv)
    print(" -", assay_csv)
    print(" -", compact_csv)
#%%
if __name__ == "__main__":
    main()