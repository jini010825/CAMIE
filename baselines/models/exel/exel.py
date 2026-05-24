#%%
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from baselines.common.utils import (
    GROUP_COLS,
    topk_idx,
    compute_gap,
    compute_oracle_overlap,
    mean_pool_or_zero,
)
#%%
def get_args():
    p = argparse.ArgumentParser("Build eXEL-GroupLasso baseline on node embeddings")

    p.add_argument("--scoring_table_dir", type=str, default="assets/scoring/scoring_dataset")
    p.add_argument("--motif_assay_table_csv", type=str, default="datasets/processed/toxcast/hierarchical/motif_assay_table.csv")
    p.add_argument("--graph_emb_dir", type=str, default="assets/toxcast_gnn/gnn_emb")
    p.add_argument("--motif_emb_dir", type=str, default="assets/toxcast_gnn/motif_emb")
    p.add_argument("--out_dir", type=str, default="assets/baselines/exel_group")
    p.add_argument("--k_values", type=int, nargs="+", default=[2])
    p.add_argument("--target_col", type=str, default="pseudo_target_logit_diff", choices=["pseudo_target_logit_diff", "pseudo_target_prob_diff"])
    p.add_argument("--lambda_grid", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    p.add_argument("--select_on_split", type=str, default="valid", choices=["train", "valid", "test"])
    p.add_argument("--selection_metric", type=str, default="oracle_overlap", choices=["oracle_overlap", "gap"])
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()
#%%
def parse_atom_indices(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return list(ast.literal_eval(x))
#%%
# -------------------------------------------------
# group lasso solver
# -------------------------------------------------
def group_soft_threshold(v: np.ndarray, thresh: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= thresh:
        return np.zeros_like(v)
    return (1.0 - thresh / norm) * v
#%%
def prox_group_lasso(alpha: np.ndarray, groups: list[np.ndarray], step: float, lam: float) -> np.ndarray:
    out = alpha.copy()
    for g in groups:
        out[g] = group_soft_threshold(out[g], step * lam)
    return out
#%%
def fit_group_lasso_coeff(
    node_emb: np.ndarray,      # [n, d]
    graph_emb: np.ndarray,     # [d]
    groups: list[np.ndarray],  # node index groups
    lam: float,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Solve:
        min_a 0.5 || z_G - H^T a ||_2^2 + lam * sum_g ||a_g||_2

    H: [n, d], H^T: [d, n], a: [n]
    """
    H = node_emb
    z = graph_emb
    n = H.shape[0]

    alpha = np.zeros(n, dtype=np.float64)

    # Lipschitz constant of grad of 0.5||z - H^T a||^2
    # grad = H(H^T a - z)
    # L = ||H H^T||_2 = ||H||_2^2
    svals = np.linalg.svd(H, compute_uv=False)
    L = float((svals[0] ** 2) if len(svals) > 0 else 1.0)
    step = 1.0 / max(L, 1e-8)

    prev_obj = np.inf

    for _ in range(max_iter):
        residual = H.T @ alpha - z               # [d]
        grad = H @ residual                      # [n]
        alpha_next = alpha - step * grad
        alpha_next = prox_group_lasso(alpha_next, groups, step, lam)

        # objective
        residual_next = H.T @ alpha_next - z
        loss = 0.5 * float(np.dot(residual_next, residual_next))
        reg = float(sum(np.linalg.norm(alpha_next[g]) for g in groups))
        obj = loss + lam * reg

        if abs(prev_obj - obj) < tol:
            alpha = alpha_next
            break

        alpha = alpha_next
        prev_obj = obj

    return alpha.astype(np.float32)
#%%
def group_scores_from_alpha(alpha: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    return np.array([np.linalg.norm(alpha[g]) for g in groups], dtype=np.float32)
#%%
# -------------------------------------------------
# build group mapping
# -------------------------------------------------
def build_group_table(scoring_df: pd.DataFrame, motif_assay_df: pd.DataFrame) -> pd.DataFrame:
    """
    scoring table + motif_assay_table(atom_indices) merge
    """
    key_cols = ["molecule_id", "smiles", "assay", "split", "label", "motif_local_id", "motif_id"]

    motif_assay_sub = motif_assay_df[
        ["molecule_id", "smiles", "assay", "split", "label", "motif_local_id", "motif_id", "atom_indices"]
    ].copy()

    merged = scoring_df.merge(
        motif_assay_sub,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    if merged["atom_indices"].isna().any():
        missing = int(merged["atom_indices"].isna().sum())
        raise ValueError(f"{missing} rows missing atom_indices after merge")

    merged["atom_indices"] = merged["atom_indices"].apply(parse_atom_indices)
    return merged
#%%
# -------------------------------------------------
# alpha/lambda selection
# -------------------------------------------------
def evaluate_lambda_on_split(
    df: pd.DataFrame,
    graph_emb: np.ndarray,
    node_emb_all: np.ndarray,
    node_ptr: np.ndarray,
    lam: float,
    k: int,
    target_col: str,
    metric: str,
    max_iter: int,
    tol: float,
) -> float:
    vals = []

    grouped = df.groupby(GROUP_COLS, sort=False)

    for _, g in grouped:
        g = g.copy().sort_values("motif_local_id").reset_index(drop=True)

        graph_idx = int(g["graph_emb_idx"].iloc[0])
        z_g = graph_emb[graph_idx]

        start = int(node_ptr[graph_idx])
        end = int(node_ptr[graph_idx + 1])
        node_emb = node_emb_all[start:end]

        groups = []
        valid_rows = []
        for i, row in g.iterrows():
            atom_idx = np.array(row["atom_indices"], dtype=int)
            atom_idx = atom_idx[(atom_idx >= 0) & (atom_idx < len(node_emb))]
            if len(atom_idx) == 0:
                continue
            groups.append(atom_idx)
            valid_rows.append(i)

        if len(groups) == 0:
            continue

        g = g.iloc[valid_rows].reset_index(drop=True)

        alpha = fit_group_lasso_coeff(
            node_emb=node_emb,
            graph_emb=z_g,
            groups=groups,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
        )
        scores = group_scores_from_alpha(alpha, groups)
        selected = topk_idx(scores, min(k, len(scores)))

        pseudo_t = g[target_col].astype(float).values

        if metric == "gap":
            val = compute_gap(pseudo_t, selected)
        else:
            val = compute_oracle_overlap(
                values=pseudo_t,
                selected_idx=selected,
                k=min(k, len(scores)),
                tie_break_ids=g["motif_id"].tolist(),
            )

        if not np.isnan(val):
            vals.append(val)

    if len(vals) == 0:
        return -np.inf
    return float(np.mean(vals))
#%%
# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    t_total_start = time.perf_counter()

    args = get_args()

    out_dir = Path(args.out_dir) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scoring_table_csv = Path(args.scoring_table_dir) / f"motif_context_scoring_table_seed{args.seed}.csv"
    scoring_df = pd.read_csv(scoring_table_csv).copy()
    motif_assay_df = pd.read_csv(args.motif_assay_table_csv).copy()

    graph_emb_npy = Path(args.graph_emb_dir) / f"seed{args.seed}" / "all_graph_emb.npy"
    node_emb_npy = Path(args.graph_emb_dir) / f"seed{args.seed}" / "all_node_emb.npy"
    node_ptr_npy = Path(args.graph_emb_dir) / f"seed{args.seed}" / "all_node_ptr.npy" 

    graph_emb = np.load(graph_emb_npy)
    node_emb_all = np.load(node_emb_npy)
    node_ptr = np.load(node_ptr_npy)

    df = build_group_table(scoring_df, motif_assay_df)

    emb_dim = int(graph_emb.shape[1])

    # -----------------------------
    # select best lambda
    # -----------------------------
    select_df = df[df["split"] == args.select_on_split].copy()
    if len(select_df) == 0:
        raise ValueError(f"No rows found for split={args.select_on_split}")

    best_lambda = None
    best_score = -np.inf
    lambda_rows = []

    print(f"[INFO] selecting lambda on split={args.select_on_split}, metric={args.selection_metric}")

    for lam in args.lambda_grid:
        tune_k = int(args.k_values[0])

        score = evaluate_lambda_on_split(
            df=select_df,
            graph_emb=graph_emb,
            node_emb_all=node_emb_all,
            node_ptr=node_ptr,
            lam=lam,
            k=tune_k,
            target_col=args.target_col,
            metric=args.selection_metric,
            max_iter=args.max_iter,
            tol=args.tol,
        )

        lambda_rows.append({
            "lambda": lam,
            "selection_split": args.select_on_split,
            "selection_metric": args.selection_metric,
            "k": tune_k,
            "score": score,
        })

        print(f"[lambda={lam:.6g}] score={score:.6f}")

        if score > best_score:
            best_score = score
            best_lambda = lam

    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(out_dir / "lambda_search.csv", index=False)

    print(f"[INFO] best lambda = {best_lambda} (score={best_score:.6f})")

    # -----------------------------
    # compute row-level scores
    # -----------------------------
    t_local_opt_start = time.perf_counter()

    score_rows = []

    grouped = df.groupby(GROUP_COLS, sort=False)

    for _, g in tqdm(grouped, total=grouped.ngroups, desc="Computing eXEL-Group scores"):
        g = g.copy().sort_values("motif_local_id").reset_index(drop=True)

        graph_idx = int(g["graph_emb_idx"].iloc[0])
        z_g = graph_emb[graph_idx]

        start = int(node_ptr[graph_idx])
        end = int(node_ptr[graph_idx + 1])
        node_emb = node_emb_all[start:end]

        groups = []
        valid_rows = []
        for i, row in g.iterrows():
            atom_idx = np.array(row["atom_indices"], dtype=int)
            atom_idx = atom_idx[(atom_idx >= 0) & (atom_idx < len(node_emb))]
            if len(atom_idx) == 0:
                continue
            groups.append(atom_idx)
            valid_rows.append(i)

        if len(groups) == 0:
            continue

        g = g.iloc[valid_rows].reset_index(drop=True)

        alpha = fit_group_lasso_coeff(
            node_emb=node_emb,
            graph_emb=z_g,
            groups=groups,
            lam=best_lambda,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        scores = group_scores_from_alpha(alpha, groups)

        g["score_exel_group"] = scores
        score_rows.append(g)

    local_optimization_seconds = time.perf_counter() - t_local_opt_start
    inference_seconds = local_optimization_seconds

    scored_df = pd.concat(score_rows, axis=0).reset_index(drop=True)
    scored_df.to_csv(out_dir / "scored_motif_context_table_exel_group.csv", index=False)

    # -----------------------------
    # hard decomposition
    # -----------------------------
    assign_rows = []
    pooled_meta_rows = []
    s2_emb_rows = []
    s1_emb_rows = []

    grouped_scored = scored_df.groupby(GROUP_COLS, sort=False)

    for k in args.k_values:
        print(f"[INFO] building decomposition for k={k}")

        for group_key, group_df in tqdm(grouped_scored, total=grouped_scored.ngroups, desc=f"exel-group-top{k}"):
            molecule_id, smiles, assay, split, label = group_key

            g = group_df.copy().sort_values(
                by=["score_exel_group", "motif_id"],
                ascending=[False, True],
            ).reset_index(drop=True)

            num_motifs = len(g)
            s2_size = min(k, num_motifs)

            g["rank_in_group"] = np.arange(1, num_motifs + 1)
            g["rule"] = "exel_group"
            g["k"] = k
            g["score"] = g["score_exel_group"].astype(float)
            g["set_assignment"] = "S1"
            if s2_size > 0:
                g.loc[:s2_size - 1, "set_assignment"] = "S2"

            keep_cols = [
                "motif_assay_id",
                "motif_id",
                "motif_local_id",
                "molecule_id",
                "motif_smiles",
                "smiles",
                "assay",
                "split",
                "label",
                "motif_emb_idx",
                "rule",
                "k",
                "rank_in_group",
                "score",
                "set_assignment",
                "pseudo_target_logit_diff",
                "pseudo_target_prob_diff",
                "num_atoms_in_motif",
            ]
            assign_rows.extend(g[keep_cols].to_dict("records"))

            s2_idx = g.loc[g["set_assignment"] == "S2", "motif_emb_idx"].astype(int).values
            s1_idx = g.loc[g["set_assignment"] == "S1", "motif_emb_idx"].astype(int).values

            motif_emb_npy = Path(args.motif_emb_dir) / f"seed{args.seed}" / "motif_emb.npy"
            motif_emb = np.load(motif_emb_npy)
            z_s2 = motif_emb[s2_idx] if len(s2_idx) > 0 else np.empty((0, emb_dim), dtype=np.float32)
            z_s1 = motif_emb[s1_idx] if len(s1_idx) > 0 else np.empty((0, emb_dim), dtype=np.float32)

            s2_pooled, s2_empty = mean_pool_or_zero(z_s2, emb_dim)
            s1_pooled, s1_empty = mean_pool_or_zero(z_s1, emb_dim)

            pooled_id = len(pooled_meta_rows)
            pooled_meta_rows.append({
                "pooled_id": pooled_id,
                "molecule_id": molecule_id,
                "smiles": smiles,
                "assay": assay,
                "split": split,
                "label": label,
                "rule": "exel_group",
                "k": k,
                "num_motifs": num_motifs,
                "s2_size": int(len(s2_idx)),
                "s1_size": int(len(s1_idx)),
                "s2_empty": int(s2_empty),
                "s1_empty": int(s1_empty),
            })

            s2_emb_rows.append(s2_pooled)
            s1_emb_rows.append(s1_pooled)

    assign_df = pd.DataFrame(assign_rows)
    pooled_meta_df = pd.DataFrame(pooled_meta_rows)
    s2_emb = np.stack(s2_emb_rows, axis=0).astype(np.float32)
    s1_emb = np.stack(s1_emb_rows, axis=0).astype(np.float32)

    assign_df.to_csv(out_dir / "motif_decomposition_table.csv", index=False)
    pooled_meta_df.to_csv(out_dir / "pooled_representation_meta.csv", index=False)
    np.save(out_dir / "s2_emb.npy", s2_emb)
    np.save(out_dir / "s1_emb.npy", s1_emb)

    n_pairs = int(scored_df.groupby(GROUP_COLS, sort=False).ngroups)
    total_seconds = time.perf_counter() - t_total_start

    device_str = "cpu"
    gpu_name = None

    summary = {
        "best_lambda": best_lambda,
        "best_selection_score": best_score,
        "selection_split": args.select_on_split,
        "selection_metric": args.selection_metric,
        "target_col": args.target_col,
        "k_values": list(args.k_values),
        "n_rows_scored": int(len(scored_df)),
        "n_assignment_rows": int(len(assign_df)),
        "n_pooled_rows": int(len(pooled_meta_df)),
        "runtime": {
            "train_seconds": 0.0,
            "inference_seconds": float(inference_seconds),
            "local_optimization_seconds": float(local_optimization_seconds),
            "total_seconds": float(total_seconds),
            "inference_seconds_per_pair": float(inference_seconds / max(1, n_pairs)),
            "local_optimization_seconds_per_pair": float(local_optimization_seconds / max(1, n_pairs)),
            "n_pairs": int(n_pairs),
            "n_scored_rows": int(len(scored_df)),
            "device": device_str,
            "gpu_name": gpu_name,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] saved to {out_dir}")
#%%
if __name__ == "__main__":
    main()