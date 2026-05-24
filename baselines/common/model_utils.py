from __future__ import annotations

from pathlib import Path

import torch

from datasets.toxcast_graph_dataset import ToxCastSharedDataset
from models.gnn.gnn import GNN_graph


def build_toxcast_dataset(
    toxcast_all_csv: str,
    assay_table_csv: str,
) -> ToxCastSharedDataset:
    return ToxCastSharedDataset(
        toxcast_all_csv=toxcast_all_csv,
        assay_table_csv=assay_table_csv,
    )


def build_assay_to_task_idx(dataset: ToxCastSharedDataset) -> dict[str, int]:
    return {assay: i for i, assay in enumerate(dataset.task_names)}


def build_smiles_to_dataset_idx(dataset: ToxCastSharedDataset) -> dict[str, int]:
    out = {}
    for i in range(len(dataset)):
        data = dataset.get(i)
        out[data.smiles] = i
    return out


def load_backbone_model(
    ckpt_dir: str,
    seed: int,
    num_tasks: int,
    device: torch.device,
    backbone_model: str = "gin",
    num_layer: int = 5,
    emb_dim: int = 300,
    drop_ratio: float = 0.5,
    graph_pooling: str = "mean",
) -> GNN_graph:
    model = GNN_graph(
        num_tasks=num_tasks,
        num_layer=num_layer,
        emb_dim=emb_dim,
        gnn_type=backbone_model,
        drop_ratio=drop_ratio,
        graph_pooling=graph_pooling,
    ).to(device)

    ckpt_path = (
        Path(ckpt_dir)
        / f"toxcast_shared_{backbone_model}_best_seed{seed}.pt"
    )

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model