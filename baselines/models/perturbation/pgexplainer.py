from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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
    get_node_embeddings_and_logits,
    normalize_numpy,
    zero_model_grads,
    detach_to_numpy,
)

from baselines.common.motif_aggregation import (
    edge_scores_to_motif_scores,
    edge_scores_to_node_scores,
)

from baselines.common.decomposition_writer import (
    save_decomposition_outputs,
)


# =========================================================
# PGExplainer MLP
# =========================================================
class PGExplainerMLP(nn.Module):
    """
    Shared PGExplainer network.

    Edge feature:
        z_e = [h_u, h_v, h_u * h_v]

    Output:
        edge_logit
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        in_dim = emb_dim * 3

        layers = []
        dim = in_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_feat: [num_edges, 3 * emb_dim]

        Returns:
            edge_logits: [num_edges]
        """
        return self.net(edge_feat).view(-1)


# =========================================================
# Arguments
# =========================================================
def get_args():
    parser = argparse.ArgumentParser(
        description="PGExplainer baseline for ToxCast multi-task motif importance"
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
    # PGExplainer training options
    # -----------------------------
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--explainer_layers", type=int, default=2)
    parser.add_argument("--explainer_dropout", type=float, default=0.0)

    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Split used to train the PGExplainer MLP. Default should be train.",
    )
    parser.add_argument(
        "--max_train_pairs",
        type=int,
        default=10000,
        help="Number of train molecule-assay pairs used to train PGExplainer.",
    )
    parser.add_argument(
        "--max_explain_pairs",
        type=int,
        default=None,
        help="Optional debug limit for inference/explanation pairs.",
    )

    # -----------------------------
    # PGExplainer loss options
    # -----------------------------
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "maximize"],
        help=(
            "mse: preserve original assay logit. "
            "maximize: maximize masked assay logit. "
            "For fidelity setting, mse is recommended."
        ),
    )
    parser.add_argument("--edge_size", type=float, default=0.005)
    parser.add_argument("--edge_ent", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--temp", type=float, default=1.0)

    # -----------------------------
    # Inference / aggregation options
    # -----------------------------
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
    # Decomposition
    # -----------------------------
    parser.add_argument("--k_values", type=int, nargs="+", default=[2])
    parser.add_argument("--rule", type=str, default="pgexplainer")

    # -----------------------------
    # Output
    # -----------------------------
    parser.add_argument(
        "--out_dir",
        type=str,
        default="assets/baselines/perturbation/pgexplainer",
    )
    parser.add_argument(
        "--save_raw_edge_scores",
        action="store_true",
        help="Save raw edge-level PGExplainer scores before motif aggregation.",
    )
    parser.add_argument(
        "--raw_edge_score_name",
        type=str,
        default="raw_edge_scores_pgexplainer.csv",
    )
    parser.add_argument(
        "--raw_only",
        action="store_true",
        help="Only save raw edge-level scores and skip motif-level table/decomposition.",
    )

    return parser.parse_args()


# =========================================================
# Dataset / model loaders
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

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[INFO] Loaded and froze backbone checkpoint: {ckpt_path}")

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


def build_group_items(
    df: pd.DataFrame,
    max_pairs: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 0,
):
    items = list(df.groupby(GROUP_COLS, sort=False, dropna=False))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(items)

    if max_pairs is not None:
        items = items[: int(max_pairs)]

    return items


# =========================================================
# PyG MessagePassing edge mask utilities
# =========================================================
def get_message_passing_modules(model: torch.nn.Module) -> list[MessagePassing]:
    modules = []

    for module in model.modules():
        if isinstance(module, MessagePassing):
            modules.append(module)

    return modules


def set_masks(
    model: torch.nn.Module,
    edge_mask: torch.Tensor,
    edge_index: torch.Tensor,
    apply_sigmoid: bool = False,
) -> None:
    """
    Register edge mask to all PyG MessagePassing modules.

    edge_mask is already probability from sigmoid(PGExplainerMLP), so
    apply_sigmoid=False is used by default.
    """

    loop_mask = edge_index[0] != edge_index[1]

    for module in get_message_passing_modules(model):
        module.__explain__ = True
        module.__edge_mask__ = edge_mask
        module.__loop_mask__ = loop_mask
        module._apply_sigmoid = bool(apply_sigmoid)


def clear_masks(model: torch.nn.Module) -> None:
    for module in get_message_passing_modules(model):
        module.__explain__ = False
        module.__edge_mask__ = None
        module.__loop_mask__ = None
        module._apply_sigmoid = True


@contextmanager
def temporary_edge_mask(
    model: torch.nn.Module,
    edge_mask: torch.Tensor,
    edge_index: torch.Tensor,
    apply_sigmoid: bool = False,
):
    set_masks(
        model=model,
        edge_mask=edge_mask,
        edge_index=edge_index,
        apply_sigmoid=apply_sigmoid,
    )
    try:
        yield
    finally:
        clear_masks(model)


# =========================================================
# PGExplainer core utilities
# =========================================================
def forward_target_logit(
    model: GNN_graph,
    data,
    assay_idx: int,
) -> torch.Tensor:
    data = ensure_batch(data)
    logits = model(data)
    return get_assay_logit(logits, assay_idx=assay_idx)


def build_edge_features(
    node_emb: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Edge feature:
        [h_u, h_v, h_u * h_v]
    """

    src, dst = edge_index[0], edge_index[1]

    h_src = node_emb[src]
    h_dst = node_emb[dst]

    edge_feat = torch.cat(
        [
            h_src,
            h_dst,
            h_src * h_dst,
        ],
        dim=-1,
    )

    return edge_feat


def edge_mask_regularization(
    edge_prob: torch.Tensor,
    edge_size: float,
    edge_ent: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    size_loss = edge_size * edge_prob.mean()

    ent = (
        -edge_prob * torch.log(edge_prob + eps)
        - (1.0 - edge_prob) * torch.log(1.0 - edge_prob + eps)
    )
    ent_loss = edge_ent * ent.mean()

    return size_loss + ent_loss


def compute_original_logit_and_node_emb(
    model: GNN_graph,
    data,
    assay_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backbone is frozen.
    Node embedding is detached because PGExplainer only trains the explainer MLP.
    """

    data = ensure_batch(data)

    with torch.no_grad():
        node_emb, _, logits = get_node_embeddings_and_logits(model, data)
        original_logit = get_assay_logit(logits, assay_idx=assay_idx).detach()
        node_emb = node_emb.detach()

    return original_logit, node_emb


def pg_forward_edge_prob(
    explainer: PGExplainerMLP,
    node_emb: torch.Tensor,
    edge_index: torch.Tensor,
    temp: float = 1.0,
) -> torch.Tensor:
    edge_feat = build_edge_features(
        node_emb=node_emb,
        edge_index=edge_index,
    )
    edge_logits = explainer(edge_feat)
    edge_prob = torch.sigmoid(edge_logits / temp)
    return edge_prob


def train_one_pair(
    model: GNN_graph,
    explainer: PGExplainerMLP,
    optimizer: torch.optim.Optimizer,
    data,
    assay_idx: int,
    args,
) -> dict:
    """
    One optimization step for one molecule-assay pair.
    """

    model.eval()
    explainer.train()

    data = ensure_batch(data)

    num_edges = int(data.edge_index.size(1))

    if num_edges == 0:
        return {
            "loss": 0.0,
            "pred_loss": 0.0,
            "reg_loss": 0.0,
            "original_logit": float("nan"),
            "masked_logit": float("nan"),
            "num_edges": 0,
            "skipped": 1,
        }

    original_logit, node_emb = compute_original_logit_and_node_emb(
        model=model,
        data=data,
        assay_idx=assay_idx,
    )

    optimizer.zero_grad(set_to_none=True)
    zero_model_grads(model)

    edge_prob = pg_forward_edge_prob(
        explainer=explainer,
        node_emb=node_emb,
        edge_index=data.edge_index,
        temp=args.temp,
    )

    with temporary_edge_mask(
        model=model,
        edge_mask=edge_prob,
        edge_index=data.edge_index,
        apply_sigmoid=False,
    ):
        masked_logit = forward_target_logit(
            model=model,
            data=data,
            assay_idx=assay_idx,
        )

    if args.loss_type == "mse":
        pred_loss = F.mse_loss(masked_logit, original_logit)
    elif args.loss_type == "maximize":
        pred_loss = -masked_logit
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    reg_loss = edge_mask_regularization(
        edge_prob=edge_prob,
        edge_size=args.edge_size,
        edge_ent=args.edge_ent,
        eps=args.eps,
    )

    loss = pred_loss + reg_loss
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "pred_loss": float(pred_loss.detach().cpu()),
        "reg_loss": float(reg_loss.detach().cpu()),
        "original_logit": float(original_logit.detach().cpu()),
        "masked_logit": float(masked_logit.detach().cpu()),
        "num_edges": int(num_edges),
        "skipped": 0,
        "edge_prob_mean": float(edge_prob.detach().mean().cpu()),
        "edge_prob_std": float(edge_prob.detach().std().cpu()),
        "edge_prob_min": float(edge_prob.detach().min().cpu()),
        "edge_prob_max": float(edge_prob.detach().max().cpu()),
    }


def infer_edge_scores_one_pair(
    model: GNN_graph,
    explainer: PGExplainerMLP,
    data,
    assay_idx: int,
    args,
) -> tuple[np.ndarray, dict]:
    """
    Generate edge scores for one molecule-assay pair.
    """

    model.eval()
    explainer.eval()

    data = ensure_batch(data)

    num_edges = int(data.edge_index.size(1))

    if num_edges == 0:
        return np.zeros((0,), dtype=float), {
            "original_logit": float("nan"),
            "masked_logit": float("nan"),
            "num_edges": 0,
            "failed": 0,
        }

    try:
        original_logit, node_emb = compute_original_logit_and_node_emb(
            model=model,
            data=data,
            assay_idx=assay_idx,
        )

        with torch.no_grad():
            edge_prob = pg_forward_edge_prob(
                explainer=explainer,
                node_emb=node_emb,
                edge_index=data.edge_index,
                temp=args.temp,
            )

        with torch.no_grad():
            with temporary_edge_mask(
                model=model,
                edge_mask=edge_prob,
                edge_index=data.edge_index,
                apply_sigmoid=False,
            ):
                masked_logit = forward_target_logit(
                    model=model,
                    data=data,
                    assay_idx=assay_idx,
                )

        edge_scores = detach_to_numpy(edge_prob).astype(float)

        info = {
            "original_logit": float(original_logit.detach().cpu()),
            "masked_logit": float(masked_logit.detach().cpu()),
            "num_edges": int(num_edges),
            "failed": 0,
        }

        return edge_scores, info

    except Exception as e:
        edge_scores = np.zeros((num_edges,), dtype=float)
        info = {
            "original_logit": float("nan"),
            "masked_logit": float("nan"),
            "num_edges": int(num_edges),
            "failed": 1,
            "error": repr(e),
        }
        return edge_scores, info


# =========================================================
# Training and inference
# =========================================================
def train_pgexplainer(
    model: GNN_graph,
    explainer: PGExplainerMLP,
    dataset: ToxCastSharedDataset,
    train_items,
    smiles_to_dataset_idx: dict[str, int],
    assay_to_task_idx: dict[str, int],
    device: torch.device,
    args,
) -> list[dict]:
    optimizer = torch.optim.Adam(
        explainer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    logs = []

    for epoch in range(1, int(args.train_epochs) + 1):
        epoch_logs = []

        rng = np.random.default_rng(args.seed + epoch)
        shuffled_items = list(train_items)
        rng.shuffle(shuffled_items)

        pbar = tqdm(
            shuffled_items,
            total=len(shuffled_items),
            desc=f"Training PGExplainer epoch {epoch}/{args.train_epochs}",
            dynamic_ncols=True,
        )

        for _, group_df in pbar:
            first = group_df.iloc[0]
            smiles = first["smiles"]
            assay = first["assay"]

            if assay not in assay_to_task_idx:
                continue

            if smiles not in smiles_to_dataset_idx:
                continue

            data_idx = smiles_to_dataset_idx[smiles]
            data = dataset.get(data_idx).to(device)
            data = ensure_batch(data)

            assay_idx = assay_to_task_idx[assay]

            log = train_one_pair(
                model=model,
                explainer=explainer,
                optimizer=optimizer,
                data=data,
                assay_idx=assay_idx,
                args=args,
            )

            epoch_logs.append(log)

            if len(epoch_logs) % 100 == 0:
                recent = epoch_logs[-100:]
                mean_loss = np.mean([x["loss"] for x in recent])
                pbar.set_postfix({"loss": f"{mean_loss:.4f}"})

        if len(epoch_logs) == 0:
            summary = {
                "epoch": epoch,
                "loss": float("nan"),
                "pred_loss": float("nan"),
                "reg_loss": float("nan"),
                "skipped": 0,
                "edge_prob_mean": float(np.mean([x["edge_prob_mean"] for x in epoch_logs if x["skipped"] == 0])),
                "edge_prob_std": float(np.mean([x["edge_prob_std"] for x in epoch_logs if x["skipped"] == 0])),
                "edge_prob_min": float(np.mean([x["edge_prob_min"] for x in epoch_logs if x["skipped"] == 0])),
                "edge_prob_max": float(np.mean([x["edge_prob_max"] for x in epoch_logs if x["skipped"] == 0])),
            }
        else:
            summary = {
                "epoch": epoch,
                "loss": float(np.mean([x["loss"] for x in epoch_logs])),
                "pred_loss": float(np.mean([x["pred_loss"] for x in epoch_logs])),
                "reg_loss": float(np.mean([x["reg_loss"] for x in epoch_logs])),
                "skipped": int(np.sum([x["skipped"] for x in epoch_logs])),
                "edge_prob_mean": float(np.mean([x["edge_prob_mean"] for x in epoch_logs if x["skipped"] == 0])),
                "edge_prob_std": float(np.mean([x["edge_prob_std"] for x in epoch_logs if x["skipped"] == 0])),
                "edge_prob_min": float(np.mean([x["edge_prob_min"] for x in epoch_logs if x["skipped"] == 0])),
                "edge_prob_max": float(np.mean([x["edge_prob_max"] for x in epoch_logs if x["skipped"] == 0])),
            }

        logs.append(summary)

        print(
            f"[EPOCH {epoch}] "
            f"loss={summary['loss']:.6f} "
            f"pred={summary['pred_loss']:.6f} "
            f"reg={summary['reg_loss']:.6f} "
            f"skipped={summary['skipped']}"
            f"edge_prob_mean={summary['edge_prob_mean']:.4f} " #0.1~0.6
            f"edge_prob_std={summary['edge_prob_std']:.4f} " #<0.05
        )

    return logs


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
        "masked_logit": info.get("masked_logit", np.nan),
        "train_split": args.train_split,
        "max_train_pairs": args.max_train_pairs,
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
    explainer: PGExplainerMLP,
    dataset: ToxCastSharedDataset,
    group_df: pd.DataFrame,
    smiles_to_dataset_idx: dict[str, int],
    assay_to_task_idx: dict[str, int],
    device: torch.device,
    args,
    save_raw_edge_scores: bool = False,
    raw_only: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
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

    edge_scores, info = infer_edge_scores_one_pair(
        model=model,
        explainer=explainer,
        data=data,
        assay_idx=assay_idx,
        args=args,
    )

    edge_scores = normalize_numpy(edge_scores, mode=args.score_norm)

    try:
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

        failed = int(info.get("failed", 0))

    except Exception as e:
        motif_scores = np.zeros((len(group_df),), dtype=float)
        node_scores = np.zeros((int(data.x.size(0)),), dtype=float)
        failed = 1
        info["failed"] = 1
        info["error"] = repr(e)

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
            score_col="score_pgexplainer_edge",
        )

    if raw_only:
        return None, raw_edge_df

    out = group_df.copy()
    out["score_pgexplainer"] = motif_scores
    out["pgexplainer_num_nodes"] = int(data.x.size(0))
    out["pgexplainer_num_edges"] = int(data.edge_index.size(1))
    out["pgexplainer_original_logit"] = info.get("original_logit", np.nan)
    out["pgexplainer_masked_logit"] = info.get("masked_logit", np.nan)
    out["pgexplainer_failed"] = int(failed)
    out["pgexplainer_score_norm"] = args.score_norm
    out["pgexplainer_edge_to_node_agg"] = args.edge_to_node_agg
    out["pgexplainer_motif_agg"] = args.motif_agg
    out["pgexplainer_train_split"] = args.train_split
    out["pgexplainer_max_train_pairs"] = args.max_train_pairs

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

    explainer = PGExplainerMLP(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.explainer_layers,
        dropout=args.explainer_dropout,
    ).to(device)

    print(
        f"[INFO] PGExplainer MLP: emb_dim={args.emb_dim}, "
        f"hidden_dim={args.hidden_dim}, layers={args.explainer_layers}, "
        f"dropout={args.explainer_dropout}"
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
    # 3. Build train pair subset
    # -------------------------------------------------
    train_df = scored_base_df[scored_base_df["split"] == args.train_split].copy()

    if len(train_df) == 0:
        raise ValueError(f"No rows found for train_split={args.train_split}")

    train_items = build_group_items(
        train_df,
        max_pairs=args.max_train_pairs,
        shuffle=True,
        seed=args.seed,
    )

    print(f"[INFO] train split rows: {len(train_df)}")
    print(f"[INFO] train pairs used for PGExplainer: {len(train_items)}")

    # -------------------------------------------------
    # 4. Train PGExplainer MLP
    # -------------------------------------------------
    t_train_start = time.perf_counter()

    train_logs = train_pgexplainer(
        model=model,
        explainer=explainer,
        dataset=dataset,
        train_items=train_items,
        smiles_to_dataset_idx=smiles_to_dataset_idx,
        assay_to_task_idx=assay_to_task_idx,
        device=device,
        args=args,
    )

    train_seconds = time.perf_counter() - t_train_start

    train_log_path = out_dir / "pgexplainer_train_log.csv"
    pd.DataFrame(train_logs).to_csv(train_log_path, index=False)

    explainer_ckpt_path = out_dir / "pgexplainer.pt"
    torch.save(
        {
            "state_dict": explainer.state_dict(),
            "args": vars(args),
            "train_logs": train_logs,
        },
        explainer_ckpt_path,
    )

    print(f"[OK] saved PGExplainer checkpoint: {explainer_ckpt_path}")
    print(f"[OK] saved train log: {train_log_path}")
    print(f"[INFO] train_seconds: {train_seconds:.2f}")

    # -------------------------------------------------
    # 5. Inference over all pairs
    # -------------------------------------------------
    explain_items = build_group_items(
        scored_base_df,
        max_pairs=args.max_explain_pairs,
        shuffle=False,
        seed=args.seed,
    )

    if args.max_explain_pairs is not None:
        print(f"[INFO] max_explain_pairs={args.max_explain_pairs}")

    t_infer_start = time.perf_counter()

    scored_groups = []
    raw_edge_groups = []

    for _, group_df in tqdm(
        explain_items,
        total=len(explain_items),
        desc="Inferring PGExplainer scores",
        dynamic_ncols=True,
    ):
        scored_g, raw_edge_g = score_one_pair(
            model=model,
            explainer=explainer,
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

    score_inference_seconds = time.perf_counter() - t_infer_start

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
        n_pairs = len(explain_items)

        summary = {
            "method": args.rule,
            "mode": "raw_only",
            "score_type": "edge",
            "raw_edge_score_path": str(raw_edge_path),
            "n_raw_edge_rows": int(n_raw_edge_rows),
            "score_norm": args.score_norm,
            "edge_to_node_agg": args.edge_to_node_agg,
            "train_split": args.train_split,
            "max_train_pairs": args.max_train_pairs,
            "max_explain_pairs": args.max_explain_pairs,
            "n_pairs": int(n_pairs),
            "runtime": {
                "train_seconds": float(train_seconds),
                "inference_seconds": float(score_inference_seconds),
                "total_seconds": float(total_seconds),
                "inference_seconds_per_pair": float(score_inference_seconds / max(1, n_pairs)),
                "device": str(device),
            },
        }

        summary_path = out_dir / "summary_raw_edge_scores.json"
        save_json(summary, summary_path)
        print(f"[OK] saved raw-only summary: {summary_path}")
        print("[DONE] PGExplainer raw edge score extraction finished.")
        return

    scored_df = pd.concat(scored_groups, axis=0).reset_index(drop=True)
    scored_path = out_dir / "scored_motif_context_table_pgexplainer.csv"
    scored_df.to_csv(scored_path, index=False)

    print(f"[OK] saved scored table: {scored_path}")
    print(f"[INFO] score_inference_seconds: {score_inference_seconds:.2f}")

    # -------------------------------------------------
    # 6. Build S1/S2 decomposition outputs
    # -------------------------------------------------
    t_decomp_start = time.perf_counter()

    decomp_path, pooled_path = save_decomposition_outputs(
        scored_df=scored_df,
        out_dir=out_dir,
        score_col="score_pgexplainer",
        rule=args.rule,
        k_values=args.k_values,
        decomp_name="motif_decomposition_table.csv",
        pooled_name="pooled_representation_meta.csv",
    )

    decomposition_seconds = time.perf_counter() - t_decomp_start

    print(f"[OK] saved decomposition table: {decomp_path}")
    print(f"[OK] saved pooled meta: {pooled_path}")
    print(f"[INFO] decomposition_seconds: {decomposition_seconds:.2f}")

    # -------------------------------------------------
    # 7. Summary
    # -------------------------------------------------
    n_failed_pairs = int(
        scored_df.groupby(GROUP_COLS, sort=False)["pgexplainer_failed"]
        .max()
        .sum()
    )

    n_pairs = int(scored_df.groupby(GROUP_COLS, sort=False).ngroups)
    total_seconds = time.perf_counter() - t_total_start
    inference_seconds = score_inference_seconds + decomposition_seconds
    method_total_seconds = train_seconds + inference_seconds

    gpu_name = None
    if torch.cuda.is_available() and "cuda" in str(device):
        gpu_name = torch.cuda.get_device_name(device)

    summary = {
        "method": args.rule,
        "definition": (
            "Shared PGExplainer MLP. Edge feature is [h_u, h_v, h_u*h_v]. "
            "The explainer is trained on train split pairs to preserve "
            "assay-specific original logits under edge masking."
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
        "hidden_dim": int(args.hidden_dim),
        "explainer_layers": int(args.explainer_layers),
        "explainer_dropout": float(args.explainer_dropout),
        "train_epochs": int(args.train_epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_split": args.train_split,
        "max_train_pairs": args.max_train_pairs,
        "max_explain_pairs": args.max_explain_pairs,
        "loss_type": args.loss_type,
        "edge_size": float(args.edge_size),
        "edge_ent": float(args.edge_ent),
        "temp": float(args.temp),
        "score_norm": args.score_norm,
        "edge_to_node_agg": args.edge_to_node_agg,
        "motif_agg": args.motif_agg,
        "k_values": [int(k) for k in args.k_values],
        "n_train_pairs_used": int(len(train_items)),
        "n_scored_rows": int(len(scored_df)),
        "n_pairs": int(n_pairs),
        "n_failed_pairs": n_failed_pairs,
        "scored_path": str(scored_path),
        "decomp_path": str(decomp_path),
        "pooled_path": str(pooled_path),
        "explainer_ckpt_path": str(explainer_ckpt_path),
        "train_log_path": str(train_log_path),
        "runtime": {
            "train_seconds": float(train_seconds),
            "score_inference_seconds": float(score_inference_seconds),
            "decomposition_seconds": float(decomposition_seconds),
            "inference_seconds": float(inference_seconds),
            "method_total_seconds": float(method_total_seconds),
            "script_total_seconds": float(total_seconds),
            "train_seconds_per_pair_per_epoch": float(
                train_seconds / max(1, len(train_items) * args.train_epochs)
            ),
            "score_inference_seconds_per_pair": float(
                score_inference_seconds / max(1, n_pairs)
            ),
            "inference_seconds_per_pair": float(
                inference_seconds / max(1, n_pairs)
            ),
            "n_train_pairs": int(len(train_items)),
            "n_explain_pairs": int(n_pairs),
            "n_scored_rows": int(len(scored_df)),
            "device": str(device),
            "gpu_name": gpu_name,
        },
    }

    summary_path = out_dir / "summary.json"
    save_json(summary, summary_path)

    print(f"[OK] saved summary: {summary_path}")
    print(f"[INFO] failed pairs: {n_failed_pairs}")
    print(f"[INFO] total_seconds: {total_seconds:.2f}")
    print("[DONE] PGExplainer baseline finished.")


if __name__ == "__main__":
    main()