#%%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import time

from subgraph.scoring.train_mse import JointMLPScorer
from utils.utils import l2_normalize
#%%
def get_args():
    p = argparse.ArgumentParser("Build row-level decomposition scores for graph/context/joint rules")

    p.add_argument("--scoring_table_dir", type=str, default="assets/scoring/scoring_dataset")

    p.add_argument("--graph_emb_dir", type=str, default="assets/toxcast_gnn/gnn_emb")
    p.add_argument("--motif_emb_dir", type=str, default="assets/toxcast_gnn/motif_emb")
    p.add_argument("--assay_emb_npy", type=str, default="assets/assay_emb/biobert_emb/hierarchy_core_mean.npy")

    p.add_argument("--joint_ckpt_dir", type=str, default="assets/scoring/joint_mlp")

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--out_dir", type=str, default="assets/scoring/decomposition/scores")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_mode", type=str, default="all", choices=["train", "valid", "test", "all"])
    p.add_argument("--loss_rule", type=str, default="mse")
    
    return p.parse_args()
#%%
def cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N, d]
    b: [N, d]
    return: [N]
    """
    a_n = l2_normalize(a, axis=1)
    b_n = l2_normalize(b, axis=1)
    return np.sum(a_n * b_n, axis=1)
#%%
@torch.no_grad()
def predict_joint_scores(
    model: nn.Module,
    motif_rows: np.ndarray,
    graph_rows: np.ndarray,
    assay_rows: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()

    n = motif_rows.shape[0]
    preds = []

    for start in tqdm(range(0, n, batch_size), desc="Predicting joint MLP scores"):
        end = min(start + batch_size, n)

        z_m = torch.tensor(motif_rows[start:end], dtype=torch.float, device=device)
        z_g = torch.tensor(graph_rows[start:end], dtype=torch.float, device=device)
        z_a = torch.tensor(assay_rows[start:end], dtype=torch.float, device=device)

        pred = model(z_m, z_g, z_a)
        preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)
#%%
def main():
    t_total_start = time.perf_counter()

    args = get_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------
    # 1. load table and embeddings
    # -------------------------------------------------
    scoring_table_csv = Path(args.scoring_table_dir)
    #scoring_table_csv = Path(args.scoring_table_dir) / f"motif_context_scoring_table_seed{args.seed}.csv"
    df = pd.read_csv(scoring_table_csv).copy()

    graph_emb_path = Path(args.graph_emb_dir) / f"seed{args.seed}" / f"{args.split_mode}_graph_emb.npy"
    graph_emb = np.load(graph_emb_path)   # [Ng, dg]
    motif_emb_path = Path(args.motif_emb_dir) / f"seed{args.seed}" / f"motif_emb.npy"
    motif_emb = np.load(motif_emb_path)   # [Nm, dm]
    assay_emb = np.load(args.assay_emb_npy)   # [Na, da]

    print(f"[INFO] scoring table rows : {len(df)}")
    print(f"[INFO] graph_emb shape    : {graph_emb.shape}")
    print(f"[INFO] motif_emb shape    : {motif_emb.shape}")
    print(f"[INFO] assay_emb shape    : {assay_emb.shape}")

    # -------------------------------------------------
    # 2. gather row-wise embeddings
    # -------------------------------------------------
    g_idx = df["graph_emb_idx"].astype(int).values
    m_idx = df["motif_emb_idx"].astype(int).values
    a_idx = df["assay_emb_idx"].astype(int).values

    z_g = graph_emb[g_idx]   # [N, dg]
    z_m = motif_emb[m_idx]   # [N, dm]
    z_a = assay_emb[a_idx]   # [N, da]

    # -------------------------------------------------
    # 3. joint MLP score
    # -------------------------------------------------
    motif_dim = motif_emb.shape[1]
    graph_dim = graph_emb.shape[1]
    assay_dim = assay_emb.shape[1]

    model = JointMLPScorer(
        motif_dim=motif_dim,
        graph_dim=graph_dim,
        assay_dim=assay_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    joint_ckpt_path = Path(args.joint_ckpt_dir) / f"best_joint_mlp_scorer_seed{args.seed}.pt"
    state_dict = torch.load(joint_ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    t_joint_start = time.perf_counter()
    score_joint_mlp = predict_joint_scores(
        model=model,
        motif_rows=z_m,
        graph_rows=z_g,
        assay_rows=z_a,
        device=device,
        batch_size=args.batch_size,
    )
    joint_mlp_score_seconds = time.perf_counter() - t_joint_start

    # -------------------------------------------------
    # 4. save scored table
    # -------------------------------------------------
    out_df = df.copy()
    out_df["score_joint_mlp"] = score_joint_mlp

    out_csv = out_dir / f"scored_table_{args.loss_rule}_seed{args.seed}.csv"
    out_df.to_csv(out_csv, index=False)

    total_seconds = time.perf_counter() - t_total_start

    summary = {
        "n_rows": int(len(out_df)),
        "graph_emb_shape": list(graph_emb.shape),
        "motif_emb_shape": list(motif_emb.shape),
        "assay_emb_shape": list(assay_emb.shape),
        "score_columns": ["score_joint_mlp"],
        "joint_ckpt_path": str(joint_ckpt_path),

        "runtime": {
            "score_seconds_by_rule": {
                "joint_mlp": float(joint_mlp_score_seconds),
            },
            "score_total_seconds": float(joint_mlp_score_seconds),
            "script_total_seconds": float(total_seconds),
            "n_rows": int(len(out_df)),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if torch.cuda.is_available() and "cuda" in str(device)
                else None
            ),
        },
    }
    with open(out_dir / f'summary_{args.loss_rule}_seed{args.seed}.json', "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] saved scored table: {out_csv}")
    print(f"[OK] saved summary     : {out_dir / f'summary_{args.loss_rule}_seed{args.seed}.json'}")
#%%
if __name__ == "__main__":
    main()