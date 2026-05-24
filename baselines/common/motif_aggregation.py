from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch

from baselines.common.utils import parse_atom_indices


AggType = Literal["mean", "sum", "max"]


def _aggregate(values: np.ndarray, agg: AggType = "mean") -> float:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return float("nan")

    if agg == "mean":
        return float(np.mean(values))
    if agg == "sum":
        return float(np.sum(values))
    if agg == "max":
        return float(np.max(values))

    raise ValueError(f"Unknown aggregation: {agg}")


def node_scores_to_motif_scores(
    motif_df: pd.DataFrame,
    node_scores: np.ndarray | torch.Tensor,
    atom_indices_col: str = "atom_indices",
    agg: AggType = "mean",
    fill_empty: float = 0.0,
) -> np.ndarray:
    """
    Convert node-level importance scores into motif-level scores.

    Args:
        motif_df:
            Rows for one molecule-assay pair.
            Must contain atom_indices_col.
        node_scores:
            Shape [num_nodes].
        atom_indices_col:
            Column containing motif atom indices.
        agg:
            mean/sum/max over atoms in each motif.
        fill_empty:
            Score for empty or invalid motif.

    Returns:
        motif_scores: np.ndarray, shape [num_motifs].
    """
    if isinstance(node_scores, torch.Tensor):
        node_scores = node_scores.detach().cpu().numpy()

    node_scores = np.asarray(node_scores, dtype=float).reshape(-1)

    scores = []
    for _, row in motif_df.iterrows():
        atom_indices = parse_atom_indices(row[atom_indices_col])
        atom_indices = [i for i in atom_indices if 0 <= i < len(node_scores)]

        if len(atom_indices) == 0:
            scores.append(float(fill_empty))
            continue

        scores.append(_aggregate(node_scores[atom_indices], agg=agg))

    return np.asarray(scores, dtype=float)


def edge_scores_to_node_scores(
    edge_index: np.ndarray | torch.Tensor,
    edge_scores: np.ndarray | torch.Tensor,
    num_nodes: int,
    agg: AggType = "mean",
) -> np.ndarray:
    """
    Convert edge-level scores to node-level scores by incident edge aggregation.

    Args:
        edge_index:
            Shape [2, num_edges].
        edge_scores:
            Shape [num_edges].
        num_nodes:
            Number of nodes.
        agg:
            mean/sum/max over incident edges.

    Returns:
        node_scores: np.ndarray, shape [num_nodes].
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    if isinstance(edge_scores, torch.Tensor):
        edge_scores = edge_scores.detach().cpu().numpy()

    edge_index = np.asarray(edge_index, dtype=int)
    edge_scores = np.asarray(edge_scores, dtype=float).reshape(-1)

    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    if edge_index.shape[1] != len(edge_scores):
        raise ValueError(
            f"edge_index E={edge_index.shape[1]} and edge_scores len={len(edge_scores)} mismatch"
        )

    buckets = [[] for _ in range(num_nodes)]

    for e, score in enumerate(edge_scores):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])

        if 0 <= src < num_nodes:
            buckets[src].append(float(score))
        if 0 <= dst < num_nodes:
            buckets[dst].append(float(score))

    node_scores = np.zeros(num_nodes, dtype=float)

    for i, vals in enumerate(buckets):
        if len(vals) == 0:
            node_scores[i] = 0.0
        else:
            node_scores[i] = _aggregate(np.asarray(vals, dtype=float), agg=agg)

    return node_scores


def edge_scores_to_motif_scores(
    motif_df: pd.DataFrame,
    edge_index: np.ndarray | torch.Tensor,
    edge_scores: np.ndarray | torch.Tensor,
    num_nodes: int,
    atom_indices_col: str = "atom_indices",
    edge_to_node_agg: AggType = "mean",
    node_to_motif_agg: AggType = "mean",
) -> np.ndarray:
    """
    Edge score -> node score -> motif score.
    Useful for GNNExplainer / PGExplainer / GraphMask.
    """
    node_scores = edge_scores_to_node_scores(
        edge_index=edge_index,
        edge_scores=edge_scores,
        num_nodes=num_nodes,
        agg=edge_to_node_agg,
    )

    return node_scores_to_motif_scores(
        motif_df=motif_df,
        node_scores=node_scores,
        atom_indices_col=atom_indices_col,
        agg=node_to_motif_agg,
    )


def binary_node_set_to_motif_scores(
    motif_df: pd.DataFrame,
    selected_nodes: set[int] | list[int] | np.ndarray,
    atom_indices_col: str = "atom_indices",
    score_mode: Literal["overlap_ratio", "binary_any", "count"] = "overlap_ratio",
) -> np.ndarray:
    """
    Convert selected node set into motif scores.
    Useful for SubgraphX output.

    score_mode:
        overlap_ratio: |motif atoms ∩ selected| / |motif atoms|
        binary_any: 1 if overlap exists else 0
        count: number of overlapping atoms
    """
    selected = set(int(x) for x in selected_nodes)

    scores = []
    for _, row in motif_df.iterrows():
        atom_indices = parse_atom_indices(row[atom_indices_col])
        atom_set = set(atom_indices)

        if len(atom_set) == 0:
            scores.append(0.0)
            continue

        overlap = len(atom_set & selected)

        if score_mode == "overlap_ratio":
            scores.append(overlap / len(atom_set))
        elif score_mode == "binary_any":
            scores.append(float(overlap > 0))
        elif score_mode == "count":
            scores.append(float(overlap))
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    return np.asarray(scores, dtype=float)