from __future__ import annotations

import argparse
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

from torch_geometric.nn.conv import MessagePassing

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
    normalize_numpy,
    zero_model_grads,
)

from baselines.common.motif_aggregation import (
    edge_scores_to_motif_scores,
    edge_scores_to_node_scores,
)

from baselines.common.decomposition_writer import (
    save_decomposition_outputs,
)


# =========================================================
# Arguments
# =========================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="GNNExplainer baseline for ToxCast multi-task motif importance"
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
    )

    # -----------------------------
    # Backbone model
    # -----------------------------
    parser.add_argument("--ckpt_dir", type=str, default="assets/toxcast_gnn/ckpt")
    parser.add_argument("--backbone_model", type=str, default="gin", choices=["gin", "gcn"])
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
    # GNNExplainer options
    # -----------------------------
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "maximize"],
        help=(
            "mse: preserve original assay logit. "
            "maximize: maximize masked assay logit. "
            "For your fidelity setting, mse is recommended."
        ),
    )

    parser.add_argument("--edge_size", type=float, default=0.005)
    parser.add_argument("--edge_ent", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument(
        "--score_norm",
        type=str,
        default="minmax",
        choices=["none", "minmax", "zscore"],
    )
    parser.add_argument(
        "--edge_to_node_agg",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
    )
    parser.add_argument(
        "--motif_agg",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
    )

    # -----------------------------
    # Debug / subset options
    # -----------------------------
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional debug limit for number of molecule-assay pairs.",
    )
    parser.add_argument(
        "--split_filter",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="Optional split filter for quick test. Default uses all rows.",
    )

    # -----------------------------
    # Decomposition
    # -----------------------------
    parser.add_argument("--k_values", type=int, nargs="+", default=[2])
    parser.add_argument("--rule", type=str, default="gnnexplainer")

    # -----------------------------
    # Output
    # -----------------------------
    parser.add_argument("--out_dir", type=str, default="assets/baselines/perturbation/gnnexplainer")
    parser.add_argument(
        "--save_raw_edge_scores",
        action="store_true",
        help="Save raw edge-level GNNExplainer scores before motif aggregation.",
    )
    parser.add_argument(
        "--raw_edge_score_name",
        type=str,
        default="raw_edge_scores_gnnexplainer.csv",
    )
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help="Only save raw edge-level scores and skip motif-level table/decomposition.",
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
# Edge mask utilities
# =========================================================
def get_message_passing_modules(model: torch.nn.Module) -> list[MessagePassing]:
    modules = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            modules.append(module)
    return modules


def set_masks(model: torch.nn.Module, edge_mask: torch.Tensor, edge_index: torch.Tensor) -> list[MessagePassing]:
    """
    Register edge mask to all PyG MessagePassing modules.

    This follows the classic PyG GNNExplainer-style masking interface:
        module.__explain__ = True
        module.__edge_mask__ = edge_mask
        module.__loop_mask__ = ...
        module._apply_sigmoid = True

    Depending on PyG version, some attributes may be ignored, but this works
    for many MessagePassing-based GNN layers.
    """
    modules = get_message_passing_modules(model)

    loop_mask = edge_index[0] != edge_index[1]

    for module in modules:
        module.__explain__ = True
        module.__edge_mask__ = edge_mask
        module.__loop_mask__ = loop_mask
        module._apply_sigmoid = True

    return modules


def clear_masks(model: torch.nn.Module) -> None:
    for module in get_message_passing_modules(model):
        module.__explain__ = False
        module.__edge_mask__ = None
        module.__loop_mask__ = None
        module._apply_sigmoid = True


@contextmanager
def temporary_edge_mask(model: torch.nn.Module, edge_mask: torch.Tensor, edge_index: torch.Tensor):
    set_masks(model, edge_mask=edge_mask, edge_index=edge_index)
    try:
        yield
    finally:
        clear_masks(model)


def init_edge_mask(num_edges: int, device: torch.device) -> torch.nn.Parameter:
    """
    GNNExplainer-style edge mask initialization.
    """
    std = torch.nn.init.calculate_gain("relu") * np.sqrt(2.0 / max(1, 2 * num_edges))
    mask = torch.empty(num_edges, device=device)
    torch.nn.init.normal_(mask, mean=0.0, std=std)
    return torch.nn.Parameter(mask)


def edge_mask_regularization(
    edge_prob: torch.Tensor,
    edge_size: float,
    edge_ent: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Size + entropy regularization.
    """
    size_loss = edge_size * edge_prob.mean()

    ent = -edge_prob * torch.log(edge_prob + eps) - (1.0 - edge_prob) * torch.log(1.0 - edge_prob + eps)
    ent_loss = edge_ent * ent.mean()

    return size_loss + ent_loss


def forward_target_logit(
    model: GNN_graph,
    data,
    assay_idx: int,
) -> torch.Tensor:
    data = ensure_batch(data)
    logits = model(data)
    return get_assay_logit(logits, assay_idx=assay_idx)


# =========================================================
# GNNExplainer core
# =========================================================
def explain_one_graph_assay(
    model: GNN_graph,
    data,
    assay_idx: int,
    epochs: int,
    lr: float,
    loss_type: str,
    edge_size: float,
    edge_ent: float,
    eps: float,
) -> tuple[np.ndarray, dict]:
    """
    Optimize edge mask for one molecule-assay pair.

    Returns:
        edge_scores: np.ndarray, shape [num_edges]
        info: optimization diagnostics
    """

    model.eval()
    data = ensure_batch(data)

    num_edges = int(data.edge_index.size(1))
    if num_edges == 0:
        return np.zeros((0,), dtype=float), {
            "original_logit": float("nan"),
            "final_masked_logit": float("nan"),
            "final_loss": float("nan"),
            "num_edges": 0,
            "failed": 0,
        }

    # freeze model parameters
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        original_logit = forward_target_logit(
            model=model,
            data=data,
            assay_idx=assay_idx,
        ).detach()

    edge_mask = init_edge_mask(num_edges=num_edges, device=data.edge_index.device)
    optimizer = torch.optim.Adam([edge_mask], lr=lr)

    final_loss_value = None
    final_masked_logit_value = None

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        zero_model_grads(model)

        with temporary_edge_mask(model, edge_mask=edge_mask, edge_index=data.edge_index):
            masked_logit = forward_target_logit(
                model=model,
                data=data,
                assay_idx=assay_idx,
            )

        edge_prob = torch.sigmoid(edge_mask)

        if loss_type == "mse":
            pred_loss = F.mse_loss(masked_logit, original_logit)
        elif loss_type == "maximize":
            pred_loss = -masked_logit
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        reg_loss = edge_mask_regularization(
            edge_prob=edge_prob,
            edge_size=edge_size,
            edge_ent=edge_ent,
            eps=eps,
        )

        loss = pred_loss + reg_loss
        loss.backward()
        optimizer.step()

        final_loss_value = float(loss.detach().cpu())
        final_masked_logit_value = float(masked_logit.detach().cpu())

    edge_scores = torch.sigmoid(edge_mask).detach().cpu().numpy().astype(float)

    info = {
        "original_logit": float(original_logit.detach().cpu()),
        "final_masked_logit": final_masked_logit_value,
        "final_loss": final_loss_value,
        "num_edges": int(num_edges),
        "failed": 0,
    }

    return edge_scores, info


def build_raw_edge_score_df(
    group_df: pd.DataFrame,
    data,
    edge_scores: np.ndarray,
    assay_idx: int,
    data_idx: int,
    info: dict,
    args,
    score_col: str,
) -> pd.DataFrame:
    first = group_df.iloc[0]
    edge_index_np = data.edge_index.detach().cpu().numpy()

    rows = []

    meta = {
        "molecule_id": first["molecule_id"],
        "smiles": first["smiles"],
        "assay": first["assay"],
        "split": first["split"],
        "label": first["label"],
        "assay_idx": int(assay_idx),
        "data_idx": int(data_idx),
        "num_nodes": int(data.x.size(0)),
        "num_edges": int(data.edge_index.size(1)),
        "score_norm": args.score_norm,
        "edge_to_node_agg": args.edge_to_node_agg,
        "failed": int(info.get("failed", 0)),
        "original_logit": info.get("original_logit", np.nan),
    }

    for col in ["graph_emb_idx", "assay_emb_idx"]:
        if col in first.index:
            meta[col] = first[col]

    for edge_idx, score in enumerate(edge_scores):
        rows.append(
            {
                **meta,
                "edge_idx": int(edge_idx),
                "src": int(edge_index_np[0, edge_idx]),
                "dst": int(edge_index_np[1, edge_idx]),
                score_col: float(score),
            }
        )

    return pd.DataFrame(rows)


def score_one_pair(
    model: GNN_graph,
    dataset: ToxCastSharedDataset,
    group_df: pd.DataFrame,
    smiles_to_dataset_idx: dict[str, int],
    assay_to_task_idx: dict[str, int],
    device: torch.device,
    args,
    save_raw_edge_scores: bool = False,
    raw_only: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Compute GNNExplainer motif scores for one molecule-assay pair.
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
    data = ensure_batch(data)

    assay_idx = assay_to_task_idx[assay]

    try:
        edge_scores, info = explain_one_graph_assay(
            model=model,
            data=data,
            assay_idx=assay_idx,
            epochs=args.epochs,
            lr=args.lr,
            loss_type=args.loss_type,
            edge_size=args.edge_size,
            edge_ent=args.edge_ent,
            eps=args.eps,
        )

        edge_scores = normalize_numpy(edge_scores, mode=args.score_norm)

        motif_scores = edge_scores_to_motif_scores(
            motif_df=group_df,
            edge_index=data.edge_index,
            edge_scores=edge_scores,
            num_nodes=int(data.x.size(0)),
            atom_indices_col="atom_indices",
            edge_to_node_agg=args.edge_to_node_agg,
            node_to_motif_agg=args.motif_agg,
        )

        node_scores = edge_scores_to_node_scores(
            edge_index=data.edge_index,
            edge_scores=edge_scores,
            num_nodes=int(data.x.size(0)),
            agg=args.edge_to_node_agg,
        )

        failed = 0

    except Exception as e:
        # Keep pipeline running. Failed pairs get zero score.
        motif_scores = np.zeros((len(group_df),), dtype=float)
        node_scores = np.zeros((int(data.x.size(0)),), dtype=float)
        edge_scores = np.zeros((int(data.edge_index.size(1)),), dtype=float)

        info = {
            "original_logit": float("nan"),
            "final_masked_logit": float("nan"),
            "final_loss": float("nan"),
            "num_edges": int(data.edge_index.size(1)),
            "failed": 1,
            "error": repr(e),
        }
        failed = 1
    
    raw_edge_df = None

    if save_raw_edge_scores:
        raw_edge_df = build_raw_edge_score_df(
            group_df=group_df,
            data=data,
            edge_scores=edge_scores,
            assay_idx=assay_idx,
            data_idx=data_idx,
            info=info,
            args=args,
            score_col="score_gnnexplainer_edge",
        )

    if raw_only:
        return None, raw_edge_df

    out = group_df.copy()
    out["score_gnnexplainer"] = motif_scores
    out["gnnexplainer_num_nodes"] = int(data.x.size(0))
    out["gnnexplainer_num_edges"] = int(data.edge_index.size(1))
    out["gnnexplainer_original_logit"] = info.get("original_logit", np.nan)
    out["gnnexplainer_final_masked_logit"] = info.get("final_masked_logit", np.nan)
    out["gnnexplainer_final_loss"] = info.get("final_loss", np.nan)
    out["gnnexplainer_failed"] = int(failed)
    out["gnnexplainer_score_norm"] = args.score_norm
    out["gnnexplainer_edge_to_node_agg"] = args.edge_to_node_agg
    out["gnnexplainer_motif_agg"] = args.motif_agg

    return out, raw_edge_df


# =========================================================
# Main
# =========================================================
def main():
    t_total_start = time.perf_counter()

    args = get_args()
    if args.raw_only:
        args.save_raw_edge_scores = True
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

    if args.split_filter is not None:
        scored_base_df = scored_base_df[scored_base_df["split"] == args.split_filter].copy()
        print(f"[INFO] applied split_filter={args.split_filter}, rows={len(scored_base_df)}")

    print(f"[INFO] base motif rows: {len(scored_base_df)}")
    print(
        f"[INFO] molecule-assay pairs: "
        f"{scored_base_df.groupby(GROUP_COLS, sort=False).ngroups}"
    )

    # -------------------------------------------------
    # 3. Compute motif-level GNNExplainer scores
    # -------------------------------------------------
    scored_groups = []
    raw_edge_groups = []

    grouped_items = list(scored_base_df.groupby(GROUP_COLS, sort=False, dropna=False))

    if args.max_pairs is not None:
        grouped_items = grouped_items[: int(args.max_pairs)]
        print(f"[INFO] max_pairs={args.max_pairs}, running pairs={len(grouped_items)}")

    t_local_opt_start = time.perf_counter()

    for _, group_df in tqdm(
        grouped_items,
        total=len(grouped_items),
        desc="Computing GNNExplainer scores",
        dynamic_ncols=True,
    ):
        scored_g, raw_edge_g = score_one_pair(
            model=model,
            dataset=dataset,
            group_df=group_df,
            smiles_to_dataset_idx=smiles_to_dataset_idx,
            assay_to_task_idx=assay_to_task_idx,
            device=device,
            args=args,
            save_raw_edge_scores=args.save_raw_edge_scores,
            raw_only=args.raw_only,
        )

        if scored_g is not None:
            scored_groups.append(scored_g)

        if raw_edge_g is not None:
            raw_edge_groups.append(raw_edge_g)
    
    local_optimization_seconds = time.perf_counter() - t_local_opt_start
    inference_seconds = local_optimization_seconds

    raw_edge_path = None
    n_raw_edge_rows = 0

    if args.save_raw_edge_scores:
        if len(raw_edge_groups) == 0:
            raise RuntimeError("No raw edge scores were collected.")

        raw_edge_df = pd.concat(raw_edge_groups, axis=0).reset_index(drop=True)
        raw_edge_path = out_dir / args.raw_edge_score_name
        raw_edge_df.to_csv(raw_edge_path, index=False)
        n_raw_edge_rows = int(len(raw_edge_df))

        print(f"[OK] saved raw edge scores: {raw_edge_path}")
        print(f"[INFO] raw edge rows: {n_raw_edge_rows}")

    if args.raw_only:
        total_seconds = time.perf_counter() - t_total_start
        n_pairs = len(grouped_items)

        summary = {
            "method": args.rule,
            "mode": "raw_only",
            "score_type": "edge",
            "raw_edge_score_path": str(raw_edge_path),
            "n_raw_edge_rows": int(n_raw_edge_rows),
            "score_norm": args.score_norm,
            "edge_to_node_agg": args.edge_to_node_agg,
            "n_pairs": int(n_pairs),
            "runtime": {
                "inference_seconds": float(inference_seconds),
                "total_seconds": float(total_seconds),
                "inference_seconds_per_pair": float(inference_seconds / max(1, n_pairs)),
                "device": str(device),
            },
        }

        summary_path = out_dir / "summary_raw_edge_scores.json"
        save_json(summary, summary_path)
        print(f"[OK] saved raw-only summary: {summary_path}")
        print("[DONE] GNNExplainer raw edge score extraction finished.")
        return

    scored_df = pd.concat(scored_groups, axis=0).reset_index(drop=True)

    scored_path = out_dir / "scored_motif_context_table_gnnexplainer.csv"
    scored_df.to_csv(scored_path, index=False)

    print(f"[OK] saved scored table: {scored_path}")

    # -------------------------------------------------
    # 4. Build S1/S2 decomposition outputs
    # -------------------------------------------------
    decomp_path, pooled_path = save_decomposition_outputs(
        scored_df=scored_df,
        out_dir=out_dir,
        score_col="score_gnnexplainer",
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
    n_failed_pairs = int(
        scored_df.groupby(GROUP_COLS, sort=False)["gnnexplainer_failed"]
        .max()
        .sum()
    )
    
    n_pairs = int(scored_df.groupby(GROUP_COLS, sort=False).ngroups)
    total_seconds = time.perf_counter() - t_total_start
    
    gpu_name = None
    if torch.cuda.is_available() and "cuda" in str(device):
        gpu_name = torch.cuda.get_device_name(device)
    
    summary = {
        "method": args.rule,
        "definition": (
            "GNNExplainer-style edge mask optimization for assay-specific logit. "
            "Loss preserves original assay logit using MSE plus edge size and entropy regularization."
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
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "loss_type": args.loss_type,
        "edge_size": float(args.edge_size),
        "edge_ent": float(args.edge_ent),
        "score_norm": args.score_norm,
        "edge_to_node_agg": args.edge_to_node_agg,
        "motif_agg": args.motif_agg,
        "k_values": [int(k) for k in args.k_values],
        "split_filter": args.split_filter,
        "max_pairs": args.max_pairs,
        "n_scored_rows": int(len(scored_df)),
        "n_pairs": int(scored_df.groupby(GROUP_COLS, sort=False).ngroups),
        "n_failed_pairs": n_failed_pairs,
        "scored_path": str(scored_path),
        "decomp_path": str(decomp_path),
        "pooled_path": str(pooled_path),
        "runtime": {
            "train_seconds": 0.0,
            "inference_seconds": float(inference_seconds),
            "local_optimization_seconds": float(local_optimization_seconds),
            "total_seconds": float(total_seconds),
            "inference_seconds_per_pair": float(inference_seconds / max(1, n_pairs)),
            "local_optimization_seconds_per_pair": float(local_optimization_seconds / max(1, n_pairs)),
            "epochs_per_pair": int(args.epochs),
            "n_pairs": int(n_pairs),
            "n_scored_rows": int(len(scored_df)),
            "device": str(device),
            "gpu_name": gpu_name,
        },
    }

    summary_path = out_dir / "summary.json"
    save_json(summary, summary_path)

    print(f"[OK] saved summary: {summary_path}")
    print(f"[INFO] failed pairs: {n_failed_pairs}")
    print("[DONE] GNNExplainer baseline finished.")


if __name__ == "__main__":
    main()