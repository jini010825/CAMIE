#%%
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from datasets.toxcast_graph_dataset import ToxCastSharedDataset
from models.gnn.gnn import GNN_graph
#%%
def get_args():
    p = argparse.ArgumentParser("Extract graph/node embeddings from shared ToxCast GNN")

    p.add_argument("--csv", type=str, default="datasets/raw_data/chem_dataset/toxcast_all.csv")
    p.add_argument("--assay_table_csv", type=str, default="datasets/processed/toxcast/hierarchical/assay_table.csv")
    p.add_argument("--ckpt_path", type=str, default="assets/toxcast_gnn/ckpt/toxcast_shared_gin_best_seed0.pt")
    p.add_argument("--out_dir", type=str, default="assets/toxcast_gnn/gnn_emb")

    p.add_argument("--model", type=str, default="gin", choices=["gin", "gcn"])
    p.add_argument("--num_layer", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=300)
    p.add_argument("--drop_ratio", type=float, default=0.5)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--split_mode", type=str, default="all", choices=["train", "valid", "test", "all"])

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()
#%%
@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()

    all_idx = []
    all_smiles = []
    all_split = []
    all_graph_emb = []

    node_emb_list = []
    node_batch_list = []
    node_ptr = [0]

    for batch in loader:
        batch = batch.to(device)

        # Requires support in gnn.py
        logits, graph_emb, node_emb = model(batch, return_emb=True, return_node_emb=True)

        all_idx.append(batch.idx.view(-1).cpu().numpy())
        all_graph_emb.append(graph_emb.cpu().numpy())

        # smiles / split are stored as a python list
        all_smiles.extend(batch.smiles)
        all_split.extend(batch.split)

        node_emb_cpu = node_emb.cpu().numpy()
        batch_cpu = batch.batch.cpu().numpy()

        node_emb_list.append(node_emb_cpu)
        node_batch_list.append(batch_cpu)

        # Record the node segment for each graph
        # batch.ptr shape = [num_graphs + 1]
        ptr = batch.ptr.cpu().numpy()
        base = node_ptr[-1]
        for k in range(1, len(ptr)):
            node_ptr.append(base + ptr[k])

    idx_all = np.concatenate(all_idx, axis=0)
    graph_emb_all = np.concatenate(all_graph_emb, axis=0)

    node_emb_all = np.concatenate(node_emb_list, axis=0)
    node_batch_all = np.concatenate(node_batch_list, axis=0)
    node_ptr = np.array(node_ptr, dtype=np.int64)

    meta_df = pd.DataFrame({
        "idx": idx_all,
        "smiles": all_smiles,
        "split": all_split,
    })

    return meta_df, graph_emb_all, node_emb_all, node_batch_all, node_ptr
#%%
def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ToxCastSharedDataset(
        toxcast_all_csv=args.csv,
        assay_table_csv=args.assay_table_csv,
    )

    train_idx, valid_idx, test_idx = dataset.get_split_indices()

    if args.split_mode == "train":
        subset_idx = train_idx.tolist()
    elif args.split_mode == "valid":
        subset_idx = valid_idx.tolist()
    elif args.split_mode == "test":
        subset_idx = test_idx.tolist()
    elif args.split_mode == "all":
        subset_idx = np.arange(len(dataset)).tolist()
    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")

    subset = torch.utils.data.Subset(dataset, subset_idx)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    model = GNN_graph(
        num_tasks=len(dataset.task_names),
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        gnn_type=args.model,
    ).to(device)

    ckpt_path = Path(args.ckpt_dir) / f"toxcast_shared_{args.model}_best_seed{args.seed}.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    meta_df, graph_emb_all, node_emb_all, node_batch_all, node_ptr = extract_embeddings(model, loader, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = out_dir / f"seed{args.seed}"
    graph_emb_dir = out_dir / f"seed{args.seed}"
    node_emb_dir = out_dir / f"seed{args.seed}"
    node_batch_dir = out_dir / f"seed{args.seed}"
    node_ptr_dir = out_dir / f"seed{args.seed}"

    meta_dir.mkdir(parents=True, exist_ok=True)
    graph_emb_dir.mkdir(parents=True, exist_ok=True)
    node_emb_dir.mkdir(parents=True, exist_ok=True)
    node_batch_dir.mkdir(parents=True, exist_ok=True)
    node_ptr_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / f"{args.split_mode}_graph_meta.csv"
    graph_emb_path = graph_emb_dir / f"{args.split_mode}_graph_emb.npy"
    node_emb_path = node_emb_dir / f"{args.split_mode}_node_emb.npy"
    node_batch_path = node_batch_dir / f"{args.split_mode}_node_batch.npy"
    node_ptr_path = node_ptr_dir / f"{args.split_mode}_node_ptr.npy"


    meta_df.to_csv(meta_path, index=False)
    np.save(graph_emb_path, graph_emb_all)
    np.save(node_emb_path, node_emb_all)
    np.save(node_batch_path, node_batch_all)
    np.save(node_ptr_path, node_ptr)

    with open(out_dir / "task_names.json", "w") as f:
        json.dump(dataset.task_names, f, indent=2)

    print("[INFO] split      :", args.split_mode)
    print("[INFO] graphs     :", len(meta_df))
    print("[INFO] graph emb  :", graph_emb_all.shape)
    print("[INFO] node emb   :", node_emb_all.shape)
    print("[INFO] saved meta :", meta_path)
    print("[INFO] saved gemb :", graph_emb_path)
    print("[INFO] saved nemb :", node_emb_path)
#%%
if __name__ == "__main__":
    main()