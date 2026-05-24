#%%
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from datasets.toxcast_graph_dataset import ToxCastSharedDataset
from models.gnn.gnn import GNN_graph
from subgraph.scoring.utils import zero_mask_motif_nodes
#%%
def get_args():
    p = argparse.ArgumentParser("Build context-conditioned motif scoring dataset with pseudo targets")

    p.add_argument("--toxcast_all_csv", type=str, default="datasets/raw_data/chem_dataset/toxcast_all.csv")
    p.add_argument("--assay_table_csv", type=str, default="datasets/processed/toxcast/hierarchical/assay_table.csv")
    p.add_argument("--motif_assay_table_csv", type=str, default="datasets/processed/toxcast/hierarchical/motif_assay_table.csv")
    p.add_argument("--motif_mapping_pkl", type=str, default="assets/motif_split/smiles_to_hierarchical_mapping.pkl")
    p.add_argument("--graph_emb_dir", type=str, default="assets/toxcast_gnn/gnn_emb")
    p.add_argument("--motif_emb_dir", type=str, default="assets/toxcast_gnn/motif_emb")
    p.add_argument("--assay_emb_meta_csv", type=str, default="assets/assay_emb/biobert_emb/hierarchy_core_meta.csv")
    p.add_argument("--ckpt_dir", type=str, default="assets/toxcast_gnn/ckpt")
    
    p.add_argument("--model", type=str, default="gin", choices=["gin", "gcn"])
    p.add_argument("--num_layer", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=300)
    p.add_argument("--drop_ratio", type=float, default=0.5)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="assets/scoring_dataset")

    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()
#%%
@torch.no_grad()
def compute_pseudo_target_for_one(
    model,
    data,
    assay_idx: int,
    atom_indices,
    device,
):
    model.eval()

    data = data.to(device)

    # original
    logit_full = model(data).view(-1)   # [T]
    prob_full = torch.sigmoid(logit_full)

    # masked
    masked_data = zero_mask_motif_nodes(data, atom_indices).to(device)
    logit_masked = model(masked_data).view(-1)
    prob_masked = torch.sigmoid(logit_masked)

    logit_diff = torch.abs(logit_full[assay_idx] - logit_masked[assay_idx]).item()
    prob_diff = torch.abs(prob_full[assay_idx] - prob_masked[assay_idx]).item()

    return logit_diff, prob_diff
#%%
def main(): 

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = ToxCastSharedDataset(
        toxcast_all_csv=args.toxcast_all_csv,
        assay_table_csv=args.assay_table_csv,
    )

    # model
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

    # tables
    motif_assay = pd.read_csv(args.motif_assay_table_csv)

    graph_meta_csv = Path(args.graph_emb_dir) / f"seed{args.seed}" / "all_graph_meta.csv"
    graph_meta = pd.read_csv(graph_meta_csv)
    
    motif_emb_meta_csv = Path(args.motif_emb_dir) / f"seed{args.seed}" / "motif_emb_meta.csv"
    motif_emb_meta = pd.read_csv(motif_emb_meta_csv)
    assay_emb_meta = pd.read_csv(args.assay_emb_meta_csv)

    with open(args.motif_mapping_pkl, "rb") as f:
        motif_mapping = pickle.load(f)

    # idx maps
    smiles_to_dataset_idx = {dataset.get(i).smiles: i for i in range(len(dataset))}
    smiles_to_graph_emb_idx = {row["smiles"]: row["idx"] for _, row in graph_meta.iterrows()}
    motif_id_to_emb_idx = {row["motif_id"]: i for i, row in motif_emb_meta.iterrows()}
    assay_to_emb_idx = {row["assay"]: i for i, row in assay_emb_meta.iterrows()}
    assay_to_task_idx = {a: i for i, a in enumerate(dataset.task_names)}

    rows = []

    for _, row in motif_assay.iterrows():
        motif_id = row["motif_id"]
        molecule_id = row["molecule_id"]
        smiles = row["smiles"]
        assay = row["assay"]
        split = row["split"]
        label = row["label"]
        motif_local_id = row["motif_local_id"]

        if smiles not in smiles_to_dataset_idx:
            continue
        if smiles not in motif_mapping:
            continue
        if assay not in assay_to_task_idx:
            continue
        if assay not in assay_to_emb_idx:
            continue
        if motif_id not in motif_id_to_emb_idx:
            continue
        if smiles not in smiles_to_graph_emb_idx:
            continue

        motif_list = motif_mapping[smiles].get("motifs", [])
        target_motif = None
        for m in motif_list:
            if int(m["motif_id"]) == int(motif_local_id):
                target_motif = m
                break
        if target_motif is None:
            continue

        atom_indices = target_motif.get("atoms", [])
        dataset_idx = smiles_to_dataset_idx[smiles]
        assay_idx = assay_to_task_idx[assay]

        data = dataset.get(dataset_idx)

        logit_diff, prob_diff = compute_pseudo_target_for_one(
            model=model,
            data=data,
            assay_idx=assay_idx,
            atom_indices=atom_indices,
            device=device,
        )

        rows.append({
            "motif_assay_id": row["motif_assay_id"],
            "motif_id": motif_id,
            "motif_local_id": row["motif_local_id"],
            "molecule_id": molecule_id,
            "motif_smiles": row["motif_smiles"],
            "smiles": smiles,
            "assay": assay,
            "split": split,
            "label": label,
            "graph_emb_idx": int(smiles_to_graph_emb_idx[smiles]),
            "motif_emb_idx": int(motif_id_to_emb_idx[motif_id]),
            "assay_emb_idx": int(assay_to_emb_idx[assay]),
            "pseudo_target_logit_diff": float(logit_diff),
            "pseudo_target_prob_diff": float(prob_diff),
            "num_atoms_in_motif": int(len(atom_indices)),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"motif_context_scoring_table_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[OK] saved: {out_csv}")
    print(f"[INFO] rows: {len(rows)}")
#%%
if __name__ == "__main__":
    main()