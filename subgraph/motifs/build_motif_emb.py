#%%
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
#%%
def get_args():
    p = argparse.ArgumentParser("Build motif embeddings from node embeddings and motif mapping")

    p.add_argument("--graph_emb_dir", type=str, default="assets/toxcast_gnn/gnn_emb")

    p.add_argument("--motif_mapping_pkl", type=str, default="assets/motif_split/smiles_to_hierarchical_mapping.pkl")
    p.add_argument("--motif_table_csv", type=str, default="datasets/processed/toxcast/hierarchical/motif_table.csv")

    p.add_argument("--out_dir", type=str, default="assets/toxcast_gnn/motif_emb")
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "sum"])

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_mode", type=str, default="all", choices=["train", "valid", "test", "all"])

    return p.parse_args()
#%%
def pool_nodes(x: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return x.mean(axis=0)
    elif mode == "sum":
        return x.sum(axis=0)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")
#%%
def main():
    args = get_args()

    graph_emb_dir = Path(args.graph_emb_dir)
    graph_meta_path = graph_emb_dir / f"seed{args.seed}" / f"{args.split_mode}_graph_meta.csv"
    node_emb_path = graph_emb_dir / f"seed{args.seed}" / f"{args.split_mode}_node_emb.npy"
    node_ptr_path = graph_emb_dir / f"seed{args.seed}" / f"{args.split_mode}_node_ptr.npy"

    graph_meta = pd.read_csv(graph_meta_path)
    node_emb = np.load(node_emb_path)
    node_ptr = np.load(node_ptr_path)

    motif_table = pd.read_csv(args.motif_table_csv)

    with open(args.motif_mapping_pkl, "rb") as f:
        motif_mapping = pickle.load(f)

    # graph index -> smiles
    smiles_list = graph_meta["smiles"].tolist()

    # smiles -> graph row index
    smiles_to_graph_idx = {s: i for i, s in enumerate(smiles_list)}

    motif_emb_rows = []
    meta_rows = []

    for _, row in motif_table.iterrows():
        motif_id = row["motif_id"]
        molecule_id = row["molecule_id"]
        smiles = row["smiles"]
        motif_local_id = row["motif_local_id"]
        motif_smiles = row["motif_smiles"]

        if smiles not in smiles_to_graph_idx:
            continue
        if smiles not in motif_mapping:
            continue

        graph_idx = smiles_to_graph_idx[smiles]
        start = int(node_ptr[graph_idx])
        end = int(node_ptr[graph_idx + 1])

        graph_node_emb = node_emb[start:end]  # [num_nodes_in_graph, d]

        info = motif_mapping[smiles]
        motif_list = info.get("motifs", [])

        # Find the corresponding motif using motif_local_id
        target_motif = None
        for m in motif_list:
            if int(m["motif_id"]) == int(motif_local_id):
                target_motif = m
                break

        if target_motif is None:
            continue

        atom_indices = target_motif.get("atoms", [])
        if len(atom_indices) == 0:
            continue

        atom_indices = np.array(atom_indices, dtype=int)

        if atom_indices.max() >= len(graph_node_emb):
            continue

        motif_node_emb = graph_node_emb[atom_indices]
        motif_emb = pool_nodes(motif_node_emb, mode=args.pool)

        motif_emb_rows.append(motif_emb)
        meta_rows.append({
            "motif_id": motif_id,
            "molecule_id": molecule_id,
            "smiles": smiles,
            "motif_local_id": motif_local_id,
            "motif_smiles": motif_smiles,
            "num_atoms_in_motif": len(atom_indices),
        })

    motif_emb_all = np.stack(motif_emb_rows, axis=0)
    motif_meta_df = pd.DataFrame(meta_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = out_dir / f"seed{args.seed}"
    meta_dir = out_dir / f"seed{args.seed}"

    emb_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    emb_path = emb_dir / "motif_emb.npy"
    meta_path = meta_dir / "motif_emb_meta.csv"

    np.save(emb_path, motif_emb_all)
    motif_meta_df.to_csv(meta_path, index=False)

    print("[INFO] motif emb shape:", motif_emb_all.shape)
    print("[INFO] saved emb     :", emb_path)
    print("[INFO] saved meta    :", meta_path)
#%%
if __name__ == "__main__":
    main()