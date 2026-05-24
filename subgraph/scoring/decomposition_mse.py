#%%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from baselines.common.utils import save_json
#%%
def get_args():
    p = argparse.ArgumentParser("Build hard S1/S2 decomposition from scored motif-context table")

    p.add_argument("--scored_table_dir", type=str, default="assets/scoring/decomposition/scores")
    p.add_argument("--motif_emb_dir", type=str, default="assets/toxcast_gnn/motif_emb")
    p.add_argument("--out_dir", type=str, default="assets/scoring/decomposition")

    p.add_argument("--k_values", type=int, nargs="+", default=[2],
                   help="top-k values for S2 decomposition, e.g. --k_values 1 2 3")
    
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss_rule", type=str, default="mse")

    return p.parse_args()
#%%
def mean_pool_or_zero(x: np.ndarray, emb_dim: int) -> tuple[np.ndarray, int]:
    """
    x: [N, d]
    return:
      pooled vector
      empty flag (1 if empty, else 0)
    """
    if x.shape[0] == 0:
        return np.zeros((emb_dim,), dtype=np.float32), 1
    return x.mean(axis=0).astype(np.float32), 0
#%%
def main():
    t_total_start = time.perf_counter()
    decomp_seconds_by_rule = {}

    args = get_args()

    out_dir = Path(args.out_dir)
    ablation_dir = out_dir / "ablation" / f"{args.loss_rule}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    scored_table_csv = Path(args.scored_table_dir)
    #scored_table_csv = Path(args.scored_table_dir) / f"scored_table_{args.loss_rule}_seed{args.seed}.csv"
    df = pd.read_csv(scored_table_csv).copy()
    motif_emb_path = Path(args.motif_emb_dir) / f"seed{args.seed}" / f"motif_emb.npy"
    motif_emb = np.load(motif_emb_path)   # [Nm, dm]
    emb_dim = motif_emb.shape[1]

    required_cols = [
        "motif_assay_id",
        "motif_id",
        "molecule_id",
        "smiles",
        "assay",
        "split",
        "label",
        "motif_emb_idx",
        "score_joint_mlp",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    rules = {
        "joint_mlp": "score_joint_mlp",
    }

    assign_rows = []
    pooled_meta_rows = []
    s2_emb_rows = []
    s1_emb_rows = []

    # group by graph-assay pair
    group_cols = ["molecule_id", "smiles", "assay", "split", "label"]
    grouped = df.groupby(group_cols, sort=False)

    for rule_name, score_col in rules.items():
        print(f"\n[INFO] processing rule = {rule_name} ({score_col})")

        t_rule_start = time.perf_counter()

        for k in args.k_values:
            print(f"[INFO] processing k = {k}")

            for group_key, group_df in tqdm(grouped, total=grouped.ngroups, desc=f"{rule_name}-top{k}"):
                molecule_id, smiles, assay, split, label = group_key

                g = group_df.copy()

                # Sort in descending order of score, with motif_id as a tie-breaker
                g = g.sort_values(
                    by=[score_col, "motif_id"],
                    ascending=[False, True],
                ).reset_index(drop=True)

                num_motifs = len(g)
                s2_size = min(k, num_motifs)

                g["rank_in_group"] = np.arange(1, num_motifs + 1)
                g["rule"] = rule_name
                g["k"] = k
                g["score"] = g[score_col].astype(float)

                # hard partition
                g["set_assignment"] = "S1"
                if s2_size > 0:
                    g.loc[:s2_size - 1, "set_assignment"] = "S2"

                # save motif-level assignment rows
                assign_keep_cols = [
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
                assign_keep_cols = [c for c in assign_keep_cols if c in g.columns]
                assign_rows.extend(g[assign_keep_cols].to_dict("records"))

                # pooled S2/S1 embeddings
                s2_idx = g.loc[g["set_assignment"] == "S2", "motif_emb_idx"].astype(int).values
                s1_idx = g.loc[g["set_assignment"] == "S1", "motif_emb_idx"].astype(int).values

                z_s2 = motif_emb[s2_idx] if len(s2_idx) > 0 else np.empty((0, emb_dim), dtype=np.float32)
                z_s1 = motif_emb[s1_idx] if len(s1_idx) > 0 else np.empty((0, emb_dim), dtype=np.float32)

                s2_pooled, s2_empty = mean_pool_or_zero(z_s2, emb_dim)
                s1_pooled, s1_empty = mean_pool_or_zero(z_s1, emb_dim)

                pooled_idx = len(pooled_meta_rows)

                pooled_meta_rows.append({
                    "pooled_id": pooled_idx,
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "assay": assay,
                    "split": split,
                    "label": label,
                    "rule": rule_name,
                    "k": k,
                    "num_motifs": num_motifs,
                    "s2_size": int(len(s2_idx)),
                    "s1_size": int(len(s1_idx)),
                    "s2_empty": int(s2_empty),
                    "s1_empty": int(s1_empty),
                })
                s2_emb_rows.append(s2_pooled)
                s1_emb_rows.append(s1_pooled)

        decomp_seconds_by_rule[rule_name] = time.perf_counter() - t_rule_start

    # -------------------------------------------------
    # save outputs
    # -------------------------------------------------
    assign_df = pd.DataFrame(assign_rows)
    pooled_meta_df = pd.DataFrame(pooled_meta_rows)
    s2_emb = np.stack(s2_emb_rows, axis=0).astype(np.float32)
    s1_emb = np.stack(s1_emb_rows, axis=0).astype(np.float32)

    assign_csv = ablation_dir / f"motif_decomposition_table_seed{args.seed}.csv"
    pooled_meta_csv = ablation_dir / f"pooled_representation_meta_seed{args.seed}.csv"
    s2_emb_npy = out_dir / f"s2_emb_{args.loss_rule}_seed{args.seed}.npy"
    s1_emb_npy = out_dir / f"s1_emb_{args.loss_rule}_seed{args.seed}.npy"

    assign_df.to_csv(assign_csv, index=False)
    pooled_meta_df.to_csv(pooled_meta_csv, index=False)
    np.save(s2_emb_npy, s2_emb)
    np.save(s1_emb_npy, s1_emb)

    # summary
    summary_rows = []

    for rule_name in pooled_meta_df["rule"].unique():
        for k in sorted(pooled_meta_df["k"].unique()):
            sub = pooled_meta_df[
                (pooled_meta_df["rule"] == rule_name)
                & (pooled_meta_df["k"] == k)
            ]

            if len(sub) == 0:
                continue

            summary_rows.append({
                "rule": rule_name,
                "k": int(k),
                "n_pairs": int(len(sub)),
                "mean_num_motifs": float(sub["num_motifs"].mean()),
                "mean_s2_size": float(sub["s2_size"].mean()),
                "mean_s1_size": float(sub["s1_size"].mean()),
                "ratio_s1_empty": float(sub["s1_empty"].mean()),
                "ratio_s2_empty": float(sub["s2_empty"].mean()),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = ablation_dir / f"decomposition_ablation_seed{args.seed}.csv"
    summary_df.to_csv(summary_csv, index=False)

    # runtime / global summary
    total_seconds = time.perf_counter() - t_total_start

    summary_json = {
        "n_assignment_rows": int(len(assign_df)),
        "n_pooled_rows": int(len(pooled_meta_df)),
        "s2_emb_shape": list(s2_emb.shape),
        "s1_emb_shape": list(s1_emb.shape),
        "rules": list(rules.keys()),
        "k_values": [int(k) for k in args.k_values],
        "summary_csv": str(summary_csv),
        "runtime": {
            "decomposition_seconds_by_rule": {
                str(rule): float(sec)
                for rule, sec in decomp_seconds_by_rule.items()
            },
            "decomposition_total_seconds": float(sum(decomp_seconds_by_rule.values())),
            "script_total_seconds": float(total_seconds),
            "n_pairs": int(grouped.ngroups),
            "n_decomposition_rows": int(len(assign_df)),
        },
    }

    save_json(
        summary_json,
        out_dir / f"summary_{args.loss_rule}_seed{args.seed}.json",
    )
    print(f"\n[OK] saved assignment table : {assign_csv}")
    print(f"[OK] saved pooled meta     : {pooled_meta_csv}")
    print(f"[OK] saved s2 emb          : {s2_emb_npy}")
    print(f"[OK] saved s1 emb          : {s1_emb_npy}")
    print(f"[OK] saved summary csv     : {summary_csv}")
#%%
if __name__ == "__main__":
    main()