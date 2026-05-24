from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_batch(data):
    """
    Ensure PyG Data object has batch vector.
    Single graph case: all nodes belong to batch 0.
    """
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(
            data.x.size(0),
            dtype=torch.long,
            device=data.x.device,
        )
    return data


def forward_logits(model, data) -> torch.Tensor:
    """
    Generic forward for graph-level multi-task GNN.

    Returns:
        logits: shape [num_tasks] for a single graph,
                or [batch_size, num_tasks] for batched graphs.
    """
    data = ensure_batch(data)
    out = model(data)

    if isinstance(out, tuple):
        out = out[0]

    return out


def get_assay_logit(
    logits: torch.Tensor,
    assay_idx: int,
    graph_idx: int | None = None,
) -> torch.Tensor:
    """
    Extract target assay logit.

    Supports:
    - logits shape [num_tasks]
    - logits shape [1, num_tasks]
    - logits shape [batch_size, num_tasks]
    """
    assay_idx = int(assay_idx)

    if logits.ndim == 1:
        return logits[assay_idx]

    if logits.ndim == 2:
        if graph_idx is None:
            graph_idx = 0
        return logits[int(graph_idx), assay_idx]

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def get_target_logit(model, data, assay_idx: int) -> torch.Tensor:
    data = ensure_batch(data)
    logits = forward_logits(model, data)
    return get_assay_logit(logits, assay_idx=assay_idx)


def normalize_torch(
    x: torch.Tensor,
    mode: Literal["none", "minmax", "zscore"] = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    if mode == "none":
        return x

    if x.numel() == 0:
        return x

    if mode == "minmax":
        return (x - x.min()) / (x.max() - x.min() + eps)

    if mode == "zscore":
        return (x - x.mean()) / (x.std(unbiased=False) + eps)

    raise ValueError(f"Unknown normalize mode: {mode}")


def normalize_numpy(
    x: np.ndarray,
    mode: Literal["none", "minmax", "zscore"] = "none",
    eps: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if mode == "none":
        return x

    if len(x) == 0:
        return x

    if np.all(np.isnan(x)):
        return np.zeros_like(x, dtype=float)

    if mode == "minmax":
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        return (x - x_min) / (x_max - x_min + eps)

    if mode == "zscore":
        mean = np.nanmean(x)
        std = np.nanstd(x)
        return (x - mean) / (std + eps)

    raise ValueError(f"Unknown normalize mode: {mode}")


def get_node_embeddings_and_logits(model, data) -> torch.Tensor:
    """
    For GNN_graph style model:
        node_emb = model.gnn_node(data)
        graph_emb = model.pool(node_emb, data.batch)
        logits = model.graph_pred_linear(graph_emb)

    This helper is used by SA / GradCAM when we need gradient
    w.r.t. final node embeddings.

    If your model uses a different field name, adjust here once.

    Returns:
        node_emb:  [num_nodes, emb_dim]
        graph_emb: [batch_size, emb_dim]
        logits:   [batch_size, num_tasks]
    """
    data = ensure_batch(data)

    if not hasattr(model, "gnn_node"):
        raise AttributeError("model must have attribute 'gnn_node' for node-embedding explanations")

    if not hasattr(model, "pool"):
        raise AttributeError("model must have attribute 'pool' for graph pooling")

    if not hasattr(model, "graph_pred_linear"):
        raise AttributeError("model must have attribute 'graph_pred_linear'")

    node_emb = model.gnn_node(data)
    graph_emb = model.pool(node_emb, data.batch)
    logits = model.graph_pred_linear(graph_emb)

    return node_emb, graph_emb, logits


def zero_model_grads(model) -> None:
    model.zero_grad(set_to_none=True)


def detach_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def safe_abs_score(
    original: torch.Tensor | float,
    masked: torch.Tensor | float,
) -> float:
    if isinstance(original, torch.Tensor):
        original = float(original.detach().cpu())
    if isinstance(masked, torch.Tensor):
        masked = float(masked.detach().cpu())
    return abs(original - masked)


def signed_drop_score(
    original: torch.Tensor | float,
    masked: torch.Tensor | float,
) -> float:
    if isinstance(original, torch.Tensor):
        original = float(original.detach().cpu())
    if isinstance(masked, torch.Tensor):
        masked = float(masked.detach().cpu())
    return original - masked