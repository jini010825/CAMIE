from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time

import torch.nn.functional as F

from datasets.toxcast_graph_dataset import ToxCastSharedDataset
from models.gnn.gnn import GNN_graph

from baselines.common.utils import (
    GROUP_COLS,
    ensure_dir,
    read_csv,
    save_json,
    set_seed,
    validate_columns,
)

from baselines.common.explain_utils import (
    get_device,
    ensure_batch,
    get_assay_logit,
    get_node_embeddings_and_logits,
    normalize_torch,
    zero_model_grads,
    detach_to_numpy,
)

from baselines.common.motif_aggregation import (
    node_scores_to_motif_scores,
)

from baselines.common.decomposition_writer import (
    save_decomposition_outputs,
)


# =========================================================
# Arguments
# =========================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="SA / Saliency baseline for ToxCast multi-task motif importance"
    )

    # -----------------------------
    # Data paths
    # -----------------------------
    parser.add_argument(
        "--toxcast_all_csv",
        type=str,
        default="datasets/raw_data/chem_dataset/toxcast_all.csv",
    )
    parser.add_argument(
        "--assay_table_csv",
        type=str,
        default="datasets/processed/toxcast/hierarchical/assay_table.csv",
    )
    parser.add_argument(
        "--motif_assay_table_csv",
        type=str,
        default="datasets/processed/toxcast/hierarchical/motif_assay_table.csv",
    )
    parser.add_argument(
        "--scoring_table_dir",
        type=str,
        default="assets/scoring/scoring_dataset",
        help=(
            "Existing motif-context scoring table. "
            "This table is used as the base motif-assay rows. "
            "If it does not contain atom_indices, atom_indices will be merged "
            "from motif_assay_table_csv."
        ),
    )

    # -----------------------------
    # Backbone model
    # -----------------------------
    parser.add_argument("--ckpt_dir", type=str, default="assets/toxcast_gnn/ckpt")
    parser.add_argument("--backbone_model", type=str, default="gin")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--drop_ratio", type=float, default=0.5)
    parser.add_argument("--JK", type=str, default="last")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        choices=["mean", "sum", "max", "attention", "set2set"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    # -----------------------------
    # SA options
    # -----------------------------
    parser.add_argument(
        "--score_norm",
        type=str,
        default="minmax",
        choices=["none", "minmax", "zscore"],
        help="Normalize node saliency scores within each molecule-assay pair.",
    )
    parser.add_argument(
        "--motif_agg",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="Aggregate node saliency scores into motif score.",
    )
    parser.add_argument(
        "--abs_grad",
        action="store_true",
        help=(
            "Use abs gradient before norm. Usually not needed because L2 norm "
            "already removes sign, but kept for ablation."
        ),
    )
    parser.add_argument(
        "--attribution_target",
        type=str,
        default="input_emb",
        choices=["input_emb", "final_node_emb"],
        help=(
            "Attribution target for saliency. "
            "input_emb computes gradients w.r.t. continuous initial atom embeddings. "
            "final_node_emb is the old variant and may collapse under mean pooling."
        ),
    )
    parser.add_argument(
        "--score_reduce",
        type=str,
        default="l2",
        choices=["l2", "sum", "mean", "max"],
        help="How to reduce gradient vector into scalar node saliency.",
    )

    # -----------------------------
    # Decomposition
    # -----------------------------
    parser.add_argument("--k_values", type=int, nargs="+", default=[2])
    parser.add_argument("--rule", type=str, default="sa")

    # -----------------------------
    # Output
    # -----------------------------
    parser.add_argument("--out_dir", type=str, default="assets/baselines/gradient/sa")
    parser.add_argument(
        "--save_raw_node_scores",
        action="store_true",
        help="Save raw node-level SA scores before motif aggregation.",
    )
    parser.add_argument(
        "--raw_node_score_name",
        type=str,
        default="raw_node_scores_sa.csv",
    )
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help="Only save raw node-level scores and skip motif-level table/decomposition.",
    )

    return parser.parse_args()


# =========================================================
# Loaders
# =========================================================
def build_dataset(args) -> ToxCastSharedDataset:
    dataset = ToxCastSharedDataset(
        toxcast_all_csv=args.toxcast_all_csv,
        assay_table_csv=args.assay_table_csv,
    )
    return dataset


def build_assay_to_task_idx(dataset: ToxCastSharedDataset) -> dict[str, int]:
    return {assay: i for i, assay in enumerate(dataset.task_names)}


def build_smiles_to_dataset_idx(dataset: ToxCastSharedDataset) -> dict[str, int]:
    smiles_to_idx = {}

    for i in range(len(dataset)):
        data = dataset.get(i)
        smiles_to_idx[data.smiles] = i

    return smiles_to_idx


def load_backbone(args, num_tasks: int, device: torch.device) -> GNN_graph:
    model = GNN_graph(
        num_tasks=num_tasks,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        gnn_type=args.backbone_model,
        drop_ratio=args.drop_ratio,
        JK=args.JK,
        residual=args.residual,
        graph_pooling=args.graph_pooling,
    ).to(device)

    ckpt_path = (
        Path(args.ckpt_dir)
        / f"toxcast_shared_{args.backbone_model}_best_seed{args.seed}.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    print(f"[INFO] Loaded backbone checkpoint: {ckpt_path}")

    return model


# =========================================================
# Dataframe preparation
# =========================================================
def prepare_scoring_dataframe(
    scoring_df: pd.DataFrame,
    motif_assay_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return motif-level rows with atom_indices.

    Preferred base table:
        motif_context_scoring_table.csv

    If atom_indices is missing from scoring_df, merge it from motif_assay_table.
    """

    required_base_cols = GROUP_COLS + [
        "motif_id",
        "motif_local_id",
    ]
    validate_columns(scoring_df, required_base_cols, name="scoring_df")

    if "atom_indices" in scoring_df.columns:
        return scoring_df.copy()

    validate_columns(
        motif_assay_df,
        required_base_cols + ["atom_indices"],
        name="motif_assay_df",
    )

    merge_cols = required_base_cols.copy()

    # It is safer to use motif_assay_id as a key if it exists on both sides
    if "motif_assay_id" in scoring_df.columns and "motif_assay_id" in motif_assay_df.columns:
        merge_cols = ["motif_assay_id"] + merge_cols

    motif_cols = merge_cols + ["atom_indices"]

    # Avoid duplicates
    motif_sub = motif_assay_df[motif_cols].drop_duplicates(subset=merge_cols).copy()

    merged = scoring_df.merge(
        motif_sub,
        on=merge_cols,
        how="left",
        validate="many_to_one",
    )

    missing = int(merged["atom_indices"].isna().sum())
    if missing > 0:
        raise ValueError(
            f"atom_indices merge failed for {missing} rows. "
            "Check motif_assay_table_csv and scoring_table_csv keys."
        )

    return merged


def forward_from_initial_node_embedding(
    model: GNN_graph,
    data,
    h0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward GNN_graph from initial continuous atom embedding h0.

    This bypasses model.gnn_node.atom_encoder(data.x), so gradients can be
    computed w.r.t. h0 instead of discrete atom features.
    """
    data = ensure_batch(data)

    edge_index = data.edge_index
    edge_attr = data.edge_attr

    gnn_node = model.gnn_node
    h_list = [h0]

    for layer in range(gnn_node.num_layer):
        h = gnn_node.convs[layer](h_list[layer], edge_index, edge_attr)
        h = gnn_node.batch_norms[layer](h)

        if layer == gnn_node.num_layer - 1:
            h = F.dropout(h, gnn_node.drop_ratio, training=gnn_node.training)
        else:
            h = F.dropout(F.relu(h), gnn_node.drop_ratio, training=gnn_node.training)

        if gnn_node.residual:
            h = h + h_list[layer]

        h_list.append(h)

    if gnn_node.JK == "last":
        node_emb = h_list[-1]
    elif gnn_node.JK == "sum":
        node_emb = 0
        for h in h_list:
            node_emb = node_emb + h
    else:
        raise ValueError(f"Unknown JK: {gnn_node.JK}")

    graph_emb = model.pool(node_emb, data.batch)
    logits = model.graph_pred_linear(graph_emb)

    return node_emb, graph_emb, logits


# =========================================================
# Saliency core
# =========================================================
def reduce_node_gradient(
    grad: torch.Tensor,
    mode: str = "l2",
) -> torch.Tensor:
    if mode == "l2":
        return torch.norm(grad, p=2, dim=-1)
    if mode == "sum":
        return grad.abs().sum(dim=-1)
    if mode == "mean":
        return grad.abs().mean(dim=-1)
    if mode == "max":
        return grad.abs().max(dim=-1).values

    raise ValueError(f"Unknown score_reduce: {mode}")


def compute_sa_node_scores(
    model: GNN_graph,
    data,
    assay_idx: int,
    score_norm: str = "minmax",
    abs_grad: bool = False,
    attribution_target: str = "input_emb",
    score_reduce: str = "l2",
) -> np.ndarray:
    """
    Saliency attribution for molecular GNN.

    Recommended default:
        attribution_target = input_emb

    This computes saliency w.r.t. the continuous initial atom embedding
    h0 = AtomEncoder(data.x). This avoids the collapse that can occur when
    using gradients w.r.t. final node embeddings under mean graph pooling.
    """

    model.eval()
    zero_model_grads(model)

    data = ensure_batch(data)

    if attribution_target == "input_emb":
        with torch.no_grad():
            h0_base = model.gnn_node.atom_encoder(data.x)

        h0 = h0_base.detach().clone().requires_grad_(True)

        _, _, logits = forward_from_initial_node_embedding(
            model=model,
            data=data,
            h0=h0,
        )

        target_tensor = h0

    elif attribution_target == "final_node_emb":
        node_emb, _, logits = get_node_embeddings_and_logits(model, data)
        node_emb.retain_grad()
        target_tensor = node_emb

    else:
        raise ValueError(f"Unknown attribution_target: {attribution_target}")

    target_logit = get_assay_logit(logits, assay_idx=assay_idx)
    target_logit.backward()

    if target_tensor.grad is None:
        raise RuntimeError(
            f"{attribution_target}.grad is None. "
            "Check whether attribution target is connected to target logit."
        )

    grad = target_tensor.grad

    if abs_grad:
        grad = grad.abs()

    node_scores = reduce_node_gradient(
        grad=grad,
        mode=score_reduce,
    )

    node_scores = normalize_torch(node_scores, mode=score_norm)

    return detach_to_numpy(node_scores).astype(float)


def score_one_pair(
    model: GNN_graph,
    dataset: ToxCastSharedDataset,
    group_df: pd.DataFrame,
    smiles_to_dataset_idx: dict[str, int],
    assay_to_task_idx: dict[str, int],
    device: torch.device,
    score_norm: str,
    motif_agg: str,
    abs_grad: bool,
    attribution_target: str,
    score_reduce: str,
    save_raw_node_scores: bool = False,
    raw_only: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Compute SA motif scores for one molecule-assay pair.
    """

    first = group_df.iloc[0]
    smiles = first["smiles"]
    assay = first["assay"]

    if assay not in assay_to_task_idx:
        raise KeyError(f"Assay not found in dataset.task_names: {assay}")

    if smiles not in smiles_to_dataset_idx:
        raise KeyError(f"SMILES not found in dataset: {smiles}")

    data_idx = smiles_to_dataset_idx[smiles]
    data = dataset.get(data_idx).to(device)

    assay_idx = assay_to_task_idx[assay]

    node_scores = compute_sa_node_scores(
        model=model,
        data=data,
        assay_idx=assay_idx,
        score_norm=score_norm,
        abs_grad=abs_grad,
        attribution_target=attribution_target,
        score_reduce=score_reduce,
    )

    raw_node_df = None

    if save_raw_node_scores:
        raw_rows = []

        meta = {
            "molecule_id": first["molecule_id"],
            "smiles": first["smiles"],
            "assay": first["assay"],
            "split": first["split"],
            "label": first["label"],
            "assay_idx": int(assay_idx),
            "data_idx": int(data_idx),
            "num_nodes": int(len(node_scores)),
            "sa_score_norm": score_norm,
            "sa_abs_grad": bool(abs_grad),
            "sa_attribution_target": attribution_target,
            "sa_score_reduce": score_reduce,
        }

        for col in ["graph_emb_idx", "assay_emb_idx"]:
            if col in first.index:
                meta[col] = first[col]

        for node_idx, score in enumerate(node_scores):
            raw_rows.append(
                {
                    **meta,
                    "node_idx": int(node_idx),
                    "score_sa_node": float(score),
                }
            )

        raw_node_df = pd.DataFrame(raw_rows)

    if raw_only:
        return None, raw_node_df
    
    motif_scores = node_scores_to_motif_scores(
        motif_df=group_df,
        node_scores=node_scores,
        atom_indices_col="atom_indices",
        agg=motif_agg,
    )

    out = group_df.copy()
    out["score_sa"] = motif_scores
    out["sa_num_nodes"] = int(len(node_scores))
    out["sa_score_norm"] = score_norm
    out["sa_motif_agg"] = motif_agg
    out["sa_attribution_target"] = attribution_target
    out["sa_score_reduce"] = score_reduce

    return out, raw_node_df


# =========================================================
# Main
# =========================================================
def main():
    t_total_start = time.perf_counter()

    args = get_args()
    if args.raw_only:
        args.save_raw_node_scores = True
    set_seed(args.seed)

    device = get_device(args.device)

    out_dir = ensure_dir(Path(args.out_dir) / f"seed{args.seed}")

    print(f"[INFO] device: {device}")
    print(f"[INFO] output dir: {out_dir}")

    # -------------------------------------------------
    # 1. Dataset / model
    # -------------------------------------------------
    dataset = build_dataset(args)
    assay_to_task_idx = build_assay_to_task_idx(dataset)
    smiles_to_dataset_idx = build_smiles_to_dataset_idx(dataset)

    print(f"[INFO] dataset size: {len(dataset)}")
    print(f"[INFO] num tasks: {len(dataset.task_names)}")

    model = load_backbone(
        args=args,
        num_tasks=len(dataset.task_names),
        device=device,
    )

    # -------------------------------------------------
    # 2. Load base tables
    # -------------------------------------------------
    scoring_table_csv = Path(args.scoring_table_dir) / f"motif_context_scoring_table_seed{args.seed}.csv"
    scoring_df = read_csv(scoring_table_csv)
    motif_assay_df = read_csv(args.motif_assay_table_csv)

    scored_base_df = prepare_scoring_dataframe(
        scoring_df=scoring_df,
        motif_assay_df=motif_assay_df,
    )

    validate_columns(
        scored_base_df,
        GROUP_COLS + ["atom_indices"],
        name="scored_base_df",
    )

    print(f"[INFO] base motif rows: {len(scored_base_df)}")
    print(
        f"[INFO] molecule-assay pairs: "
        f"{scored_base_df.groupby(GROUP_COLS, sort=False).ngroups}"
    )

    # -------------------------------------------------
    # 3. Compute motif-level SA scores
    # -------------------------------------------------
    t_infer_start = time.perf_counter()

    scored_groups = []
    raw_node_groups = []

    grouped = scored_base_df.groupby(GROUP_COLS, sort=False, dropna=False)

    for _, group_df in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Computing SA scores",
        dynamic_ncols=True,
    ):
        scored_g, raw_node_g = score_one_pair(
            model=model,
            dataset=dataset,
            group_df=group_df,
            smiles_to_dataset_idx=smiles_to_dataset_idx,
            assay_to_task_idx=assay_to_task_idx,
            device=device,
            score_norm=args.score_norm,
            motif_agg=args.motif_agg,
            abs_grad=args.abs_grad,
            attribution_target=args.attribution_target,
            score_reduce=args.score_reduce,
            save_raw_node_scores=args.save_raw_node_scores,
            raw_only=args.raw_only,
        )

        if scored_g is not None:
            scored_groups.append(scored_g)

        if raw_node_g is not None:
            raw_node_groups.append(raw_node_g)

    inference_seconds = time.perf_counter() - t_infer_start

    raw_node_path = None
    n_raw_node_rows = 0

    if args.save_raw_node_scores:
        if len(raw_node_groups) == 0:
            raise RuntimeError("No raw node scores were collected.")

        raw_node_df = pd.concat(raw_node_groups, axis=0).reset_index(drop=True)
        raw_node_path = out_dir / args.raw_node_score_name
        raw_node_df.to_csv(raw_node_path, index=False)
        n_raw_node_rows = int(len(raw_node_df))

        print(f"[OK] saved raw node scores: {raw_node_path}")
        print(f"[INFO] raw node rows: {n_raw_node_rows}")

    if args.raw_only:
        total_seconds = time.perf_counter() - t_total_start
        n_pairs = int(scored_base_df.groupby(GROUP_COLS, sort=False).ngroups)

        summary = {
            "method": args.rule,
            "mode": "raw_only",
            "score_type": "node",
            "raw_node_score_path": str(raw_node_path),
            "n_raw_node_rows": int(n_raw_node_rows),
            "score_norm": args.score_norm,
            "abs_grad": bool(args.abs_grad),
            "n_pairs": int(n_pairs),
            "attribution_target": args.attribution_target,
            "score_reduce": args.score_reduce,
            "runtime": {
                "inference_seconds": float(inference_seconds),
                "total_seconds": float(total_seconds),
                "inference_seconds_per_pair": float(inference_seconds / max(1, n_pairs)),
                "device": str(device),
            },
        }

        summary_path = out_dir / "summary_raw_node_scores.json"
        save_json(summary, summary_path)
        print(f"[OK] saved raw-only summary: {summary_path}")
        print("[DONE] SA raw node score extraction finished.")
        return

    scored_df = pd.concat(scored_groups, axis=0).reset_index(drop=True)
    scored_path = out_dir / "scored_motif_context_table_sa.csv"
    scored_df.to_csv(scored_path, index=False)

    print(f"[OK] saved scored table: {scored_path}")

    # -------------------------------------------------
    # 4. Build S1/S2 decomposition outputs
    # -------------------------------------------------
    decomp_path, pooled_path = save_decomposition_outputs(
        scored_df=scored_df,
        out_dir=out_dir,
        score_col="score_sa",
        rule=args.rule,
        k_values=args.k_values,
        decomp_name="motif_decomposition_table.csv",
        pooled_name="pooled_representation_meta.csv",
    )

    print(f"[OK] saved decomposition table: {decomp_path}")
    print(f"[OK] saved pooled meta: {pooled_path}")

    # -------------------------------------------------
    # 5. Summary
    # -------------------------------------------------
    total_seconds = time.perf_counter() - t_total_start
    n_pairs = int(scored_df.groupby(GROUP_COLS, sort=False).ngroups)

    gpu_name = None
    if torch.cuda.is_available() and "cuda" in str(device):
        gpu_name = torch.cuda.get_device_name(device)

    summary = {
        "method": args.rule,
        "definition": (
            "SA node score is computed from gradients of the assay logit with "
            "respect to the selected attribution target. Default target is the "
            "continuous initial atom embedding h0 = AtomEncoder(data.x)."
        ),
        "scoring_table_csv": scoring_table_csv,
        "motif_assay_table_csv": args.motif_assay_table_csv,
        "toxcast_all_csv": args.toxcast_all_csv,
        "assay_table_csv": args.assay_table_csv,
        "ckpt_dir": args.ckpt_dir,
        "seed": int(args.seed),
        "backbone_model": args.backbone_model,
        "num_layer": int(args.num_layer),
        "emb_dim": int(args.emb_dim),
        "drop_ratio": float(args.drop_ratio),
        "JK": args.JK,
        "residual": bool(args.residual),
        "graph_pooling": args.graph_pooling,
        "score_norm": args.score_norm,
        "motif_agg": args.motif_agg,
        "abs_grad": bool(args.abs_grad),
        "k_values": [int(k) for k in args.k_values],
        "n_scored_rows": int(len(scored_df)),
        "n_pairs": int(scored_df.groupby(GROUP_COLS, sort=False).ngroups),
        "scored_path": str(scored_path),
        "decomp_path": str(decomp_path),
        "pooled_path": str(pooled_path),
        "attribution_target": args.attribution_target,
        "score_reduce": args.score_reduce,
        "runtime": {
            "train_seconds": 0.0,
            "inference_seconds": float(inference_seconds),
            "total_seconds": float(total_seconds),
            "inference_seconds_per_pair": float(inference_seconds / max(1, n_pairs)),
            "n_pairs": int(n_pairs),
            "n_scored_rows": int(len(scored_df)),
            "device": str(device),
            "gpu_name": gpu_name,
        },
    }

    summary_path = out_dir / "summary.json"
    save_json(summary, summary_path)

    print(f"[OK] saved summary: {summary_path}")
    print("[DONE] SA baseline finished.")


if __name__ == "__main__":
    main()