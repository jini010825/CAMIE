from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import time

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
        description=(
            "Simplified Guided Backprop baseline for "
            "ToxCast multi-task motif importance"
        )
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
    # GBP options
    # -----------------------------
    parser.add_argument(
        "--attribution_target",
        type=str,
        default="input_emb",
        choices=["input_emb", "final_node_emb"],
        help=(
            "Attribution target for GBP. "
            "input_emb follows Captum-style attribution on continuous atom embeddings; "
            "final_node_emb is the previous node-embedding variant."
        ),
    )
    parser.add_argument(
        "--gbp_rule",
        type=str,
        default="pos_grad_pos_act",
        choices=["pos_grad", "pos_grad_pos_act", "pos_grad_abs_act", "grad_times_act_pos"],
        help="How to construct Guided Backprop node attribution.",
    )
    parser.add_argument(
        "--score_norm",
        type=str,
        default="minmax",
        choices=["none", "minmax", "zscore"],
        help="Normalize node GBP scores within each molecule-assay pair.",
    )
    parser.add_argument(
        "--motif_agg",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="Aggregate node GBP scores into motif score.",
    )
    parser.add_argument(
        "--positive_only",
        action="store_true",
        default=True,
        help=(
            "Keep only positive gradient components. "
            "Default True for simplified Guided Backprop."
        ),
    )
    parser.add_argument(
        "--no_positive_only",
        action="store_false",
        dest="positive_only",
        help=(
            "Disable positive-gradient filtering. "
            "This becomes equivalent to node-embedding saliency."
        ),
    )
    parser.add_argument(
        "--score_reduce",
        type=str,
        default="sum",
        choices=["sum", "l2", "mean", "max"],
        help=(
            "How to reduce positive gradient vector into node score. "
            "For GBP, sum is recommended to differ from vanilla SA."
        ),
    )

    # -----------------------------
    # Decomposition
    # -----------------------------
    parser.add_argument("--k_values", type=int, nargs="+", default=[2])
    parser.add_argument("--rule", type=str, default="gbp")

    # -----------------------------
    # Output
    # -----------------------------
    parser.add_argument("--out_dir", type=str, default="assets/baselines/gradient/gbp")
    parser.add_argument(
        "--save_raw_node_scores",
        action="store_true",
        help="Save raw node-level GBP scores before motif aggregation.",
    )
    parser.add_argument(
        "--raw_node_score_name",
        type=str,
        default="raw_node_scores_gbp.csv",
        help="Filename for raw node-level scores.",
    )
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help=(
            "Only save raw node-level scores and skip motif-level scored table "
            "and decomposition outputs. Useful for aggregation sensitivity."
        ),
    )

    return parser.parse_args()


# =========================================================
# Loaders
# =========================================================
def build_dataset(args) -> ToxCastSharedDataset:
    return ToxCastSharedDataset(
        toxcast_all_csv=args.toxcast_all_csv,
        assay_table_csv=args.assay_table_csv,
    )


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

    if "motif_assay_id" in scoring_df.columns and "motif_assay_id" in motif_assay_df.columns:
        merge_cols = ["motif_assay_id"] + merge_cols

    motif_cols = merge_cols + ["atom_indices"]
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


# =========================================================
# Guided Backprop core
# =========================================================
def forward_from_initial_node_embedding(
    model: GNN_graph,
    data,
    h0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward GNN_graph from initial continuous atom embedding h0.

    This bypasses model.gnn_node.atom_encoder(data.x) so that attribution can
    be computed w.r.t. h0 instead of discrete atom features.

    Returns:
        node_emb, graph_emb, logits
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

def compute_gbp_node_scores(
    model: GNN_graph,
    data,
    assay_idx: int,
    score_norm: str = "minmax",
    attribution_target: str = "input_emb",
    gbp_rule: str = "pos_grad_pos_act",
) -> np.ndarray:
    """
    Captum-style Guided Backprop variant for molecular GNN.

    Because data.x is discrete atom categorical features, we compute attribution
    w.r.t. the initial continuous atom embedding h0 = AtomEncoder(data.x).

    Options:
        attribution_target:
            input_emb: use initial atom embedding h0.
            final_node_emb: use final node embedding h.

        gbp_rule:
            pos_grad:
                a_i = sum_d ReLU(grad_i,d)

            pos_grad_pos_act:
                a_i = sum_d ReLU(grad_i,d) * ReLU(act_i,d)

            pos_grad_abs_act:
                a_i = sum_d ReLU(grad_i,d) * |act_i,d|

            grad_times_act_pos:
                a_i = sum_d ReLU(grad_i,d * act_i,d)
    """

    model.eval()
    zero_model_grads(model)

    data = ensure_batch(data)

    if attribution_target == "input_emb":
        # Continuous atom embedding attribution.
        with torch.no_grad():
            h0_base = model.gnn_node.atom_encoder(data.x)

        h0 = h0_base.detach().clone().requires_grad_(True)

        _, _, logits = forward_from_initial_node_embedding(
            model=model,
            data=data,
            h0=h0,
        )

        act = h0

    elif attribution_target == "final_node_emb":
        node_emb, _, logits = get_node_embeddings_and_logits(model, data)
        node_emb.retain_grad()
        act = node_emb

    else:
        raise ValueError(f"Unknown attribution_target: {attribution_target}")

    target_logit = get_assay_logit(logits, assay_idx=assay_idx)
    target_logit.backward()

    if act.grad is None:
        raise RuntimeError(
            "Attribution tensor grad is None. "
            "Check whether attribution target is connected to target logit."
        )

    grad = act.grad

    if gbp_rule == "pos_grad":
        guided = torch.relu(grad)
    elif gbp_rule == "pos_grad_pos_act":
        guided = torch.relu(grad) * torch.relu(act.detach())
    elif gbp_rule == "pos_grad_abs_act":
        guided = torch.relu(grad) * act.detach().abs()
    elif gbp_rule == "grad_times_act_pos":
        guided = torch.relu(grad * act.detach())
    else:
        raise ValueError(f"Unknown gbp_rule: {gbp_rule}")

    node_scores = guided.sum(dim=-1)
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
    attribution_target: str,
    gbp_rule: str,
    save_raw_node_scores: bool = False,
    raw_only: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Compute GBP motif scores for one molecule-assay pair.

    Returns:
        scored motif-level dataframe
        raw node-level score dataframe, if save_raw_node_scores=True
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

    node_scores = compute_gbp_node_scores(
        model=model,
        data=data,
        assay_idx=assay_idx,
        score_norm=score_norm,
        attribution_target=attribution_target,
        gbp_rule=gbp_rule,
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
            "gbp_score_norm": score_norm,
            "gbp_attribution_target": attribution_target,
            "gbp_rule": gbp_rule,
        }

        for col in ["graph_emb_idx", "assay_emb_idx"]:
            if col in first.index:
                meta[col] = first[col]

        for node_idx, score in enumerate(node_scores):
            raw_rows.append(
                {
                    **meta,
                    "node_idx": int(node_idx),
                    "score_gbp_node": float(score),
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
    out["score_gbp"] = motif_scores
    out["gbp_num_nodes"] = int(len(node_scores))
    out["gbp_score_norm"] = score_norm
    out["gbp_motif_agg"] = motif_agg
    out["gbp_attribution_target"] = attribution_target
    out["gbp_rule"] = gbp_rule

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
    # 3. Compute motif-level GBP scores
    # -------------------------------------------------
    t_infer_start = time.perf_counter()

    scored_groups = []
    raw_node_groups = []

    grouped = scored_base_df.groupby(GROUP_COLS, sort=False, dropna=False)

    for _, group_df in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Computing GBP scores",
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
            attribution_target=args.attribution_target,
            gbp_rule=args.gbp_rule,
            save_raw_node_scores=args.save_raw_node_scores,
            raw_only=args.raw_only,
        )

        if scored_g is not None:
            scored_groups.append(scored_g)

        if raw_node_g is not None:
            raw_node_groups.append(raw_node_g)

    inference_seconds = time.perf_counter() - t_infer_start

    # -------------------------------------------------
    # 3-A. Raw-only mode: save raw node scores and exit
    # -------------------------------------------------
    raw_node_path = None
    n_raw_node_rows = 0

    if args.save_raw_node_scores:
        if len(raw_node_groups) == 0:
            raise RuntimeError(
                "save_raw_node_scores=True, but no raw node scores were collected."
            )

        raw_node_df = pd.concat(raw_node_groups, axis=0).reset_index(drop=True)
        raw_node_path = out_dir / args.raw_node_score_name
        raw_node_df.to_csv(raw_node_path, index=False)
        n_raw_node_rows = int(len(raw_node_df))

        print(f"[OK] saved raw node scores: {raw_node_path}")
        print(f"[INFO] raw node rows: {n_raw_node_rows}")

    if args.raw_only:
        total_seconds = time.perf_counter() - t_total_start
        n_pairs = int(scored_base_df.groupby(GROUP_COLS, sort=False).ngroups)

        gpu_name = None
        if torch.cuda.is_available() and "cuda" in str(device):
            gpu_name = torch.cuda.get_device_name(device)

        summary = {
            "method": args.rule,
            "mode": "raw_only",
            "definition": "GBP raw node-level scores only; motif aggregation and decomposition skipped.",
            "scoring_table_csv": str(scoring_table_csv),
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
            "attribution_target": args.attribution_target,
            "gbp_rule": args.gbp_rule,
            "score_reduce": args.score_reduce,
            "positive_only": bool(args.positive_only),
            "save_raw_node_scores": True,
            "raw_node_score_path": str(raw_node_path),
            "n_raw_node_rows": int(n_raw_node_rows),
            "n_pairs": int(n_pairs),
            "runtime": {
                "train_seconds": 0.0,
                "inference_seconds": float(inference_seconds),
                "total_seconds": float(total_seconds),
                "inference_seconds_per_pair": float(inference_seconds / max(1, n_pairs)),
                "n_pairs": int(n_pairs),
                "device": str(device),
                "gpu_name": gpu_name,
            },
        }

        summary_path = out_dir / "summary_raw_node_scores.json"
        save_json(summary, summary_path)

        print(f"[OK] saved raw-only summary: {summary_path}")
        print("[DONE] GBP raw node score extraction finished.")
        return

    scored_df = pd.concat(scored_groups, axis=0).reset_index(drop=True)
    scored_path = out_dir / "scored_motif_context_table_gbp.csv"
    scored_df.to_csv(scored_path, index=False)

    print(f"[OK] saved scored table: {scored_path}")

    raw_node_path = None
    n_raw_node_rows = 0

    if args.save_raw_node_scores:
        if len(raw_node_groups) == 0:
            raise RuntimeError("save_raw_node_scores=True, but no raw node scores were collected.")

        raw_node_df = pd.concat(raw_node_groups, axis=0).reset_index(drop=True)
        raw_node_path = out_dir / args.raw_node_score_name
        raw_node_df.to_csv(raw_node_path, index=False)
        n_raw_node_rows = int(len(raw_node_df))

        print(f"[OK] saved raw node scores: {raw_node_path}")
        print(f"[INFO] raw node rows: {n_raw_node_rows}")

    # -------------------------------------------------
    # 4. Build S1/S2 decomposition outputs
    # -------------------------------------------------
    decomp_path, pooled_path = save_decomposition_outputs(
        scored_df=scored_df,
        out_dir=out_dir,
        score_col="score_gbp",
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
            "Simplified Guided Backprop on final node embeddings: "
            "guided_grad = ReLU(d logit_assay / d h_i); "
            "node_score_i = ||guided_grad_i||_2"
        ),
        "note": (
            "This is not full layer-wise Guided Backprop because current GNN "
            "uses functional F.relu. It is a positive-gradient node-embedding "
            "attribution baseline."
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
        "positive_only": bool(args.positive_only),
        "k_values": [int(k) for k in args.k_values],
        "n_scored_rows": int(len(scored_df)),
        "n_pairs": int(scored_df.groupby(GROUP_COLS, sort=False).ngroups),
        "scored_path": str(scored_path),
        "decomp_path": str(decomp_path),
        "pooled_path": str(pooled_path),
        "save_raw_node_scores": bool(args.save_raw_node_scores),
        "raw_node_score_path": str(raw_node_path) if raw_node_path is not None else None,
        "n_raw_node_rows": int(n_raw_node_rows),
        "score_reduce": args.score_reduce,
        "attribution_target": args.attribution_target,
        "gbp_rule": args.gbp_rule,
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
    print("[DONE] GBP baseline finished.")


if __name__ == "__main__":
    main()