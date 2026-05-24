#%%
import os
import pickle
import argparse
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
#%%
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all_csv", type=str, default="datasets/raw_data/chem_dataset/toxcast_all.csv")
    p.add_argument("--assay_meta_csv", type=str, default="assets/assay_emb/biobert_emb/hierarchy_selected_meta.csv")
    p.add_argument("--assay_embedding_npy", type=str, default="assets/assay_emb/biobert_emb/hierarchy_selected_mean.npy")
    p.add_argument("--motif_mapping_pkl", type=str, default="assets/motif_split/smiles_to_hierarchical_mapping.pkl")
    p.add_argument("--split_csv", type=str, default=None)
    p.add_argument("--decomposition_method", type=str, default="brics")
    p.add_argument("--out_dir", type=str, default="datasets/processed/toxcast/hierarchical")
    return p.parse_args()
#%%
def build_assay_table(
    assay_meta_csv: str,
    assay_embedding_npy: Optional[str] = None,
) -> pd.DataFrame:
    assay_df = pd.read_csv(assay_meta_csv).copy()

    if "assay" not in assay_df.columns:
        raise ValueError("[assay_table] 'assay' column is required.")

    assay_df = assay_df.drop_duplicates(subset=["assay"]).reset_index(drop=True)
    assay_df["assay_id"] = [f"assay_{i:05d}" for i in range(len(assay_df))]
    assay_df["emb_idx"] = assay_df.index.astype(int)

    if "direction" not in assay_df.columns:
        assay_df["direction"] = pd.NA

    keep_cols = [
        "assay_id",
        "assay",
        "domain",
        "tier",
        "moa",
        "target_gene",
        "direction",
        "embedding_text",
        "semantic_summary",
        "context_role",
        "emb_idx",
    ]
    keep_cols = [c for c in keep_cols if c in assay_df.columns]

    assay_table = assay_df[keep_cols].copy()

    if assay_embedding_npy is not None and os.path.exists(assay_embedding_npy):
        emb = np.load(assay_embedding_npy)
        if len(emb) != len(assay_table):
            print(
                f"[Warning] assay embedding rows ({len(emb)}) != assay_table rows ({len(assay_table)})"
            )

    return assay_table
#%%
def build_molecule_assay_table(
    all_csv: str,
    assay_table: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.read_csv(all_csv).copy()

    needed = ["smiles", "assay", "label", "split"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"[molecule_assay_table] required column missing: {c}")

    # Generate molecule_id (based on unique smiles)
    smiles_df = df[["smiles"]].drop_duplicates().reset_index(drop=True)
    smiles_df["molecule_id"] = [f"mol_{i:06d}" for i in range(len(smiles_df))]

    df = df.merge(smiles_df, on="smiles", how="left")

    # Link assay_id
    assay_key = assay_table[["assay_id", "assay"]].drop_duplicates()
    df = df.merge(assay_key, on="assay", how="left")

    # Keep only assays present in assay_table
    df = df[df["assay_id"].notna()].copy()
    df = df.reset_index(drop=True)
    df["molecule_assay_id"] = [f"ma_{i:07d}" for i in range(len(df))]

    keep_cols = [
        "molecule_assay_id",
        "molecule_id",
        "smiles",
        "assay_id",
        "assay",
        "label",
        "split",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].copy()
#%%
def load_motif_mapping(mapping_pkl: str) -> Dict[str, Any]:
    if not os.path.exists(mapping_pkl):
        raise FileNotFoundError(f"motif mapping file not found: {mapping_pkl}")

    with open(mapping_pkl, "rb") as f:
        mapping = pickle.load(f)

    if not isinstance(mapping, dict):
        raise ValueError("motif mapping pickle must be a dict keyed by smiles.")

    return mapping
#%%
def build_motif_table(
    molecule_assay_table: pd.DataFrame,
    motif_mapping: Dict[str, Any],
    decomposition_method: str = "brics",
) -> pd.DataFrame:
    mol_df = molecule_assay_table[["molecule_id", "smiles"]].drop_duplicates().reset_index(drop=True)

    rows = []
    global_motif_counter = 0

    for _, row in mol_df.iterrows():
        molecule_id = row["molecule_id"]
        smiles = row["smiles"]

        info = motif_mapping.get(smiles, None)

        # Fallback if mapping does not exist
        if info is None or "motifs" not in info or len(info["motifs"]) == 0:
            rows.append({
                "motif_id": f"motif_{global_motif_counter:08d}",
                "molecule_id": molecule_id,
                "smiles": smiles,
                "motif_local_id": 0,
                "motif_smiles": smiles,
                "atom_indices": "[]",
                "num_atoms_in_molecule": pd.NA,
                "motif_edges": "[]",
                "decomposition_method": decomposition_method,
                "n_motifs_in_molecule": 1,
            })
            global_motif_counter += 1
            continue

        motif_list = info.get("motifs", [])
        motif_edges = info.get("motif_edges", [])
        num_atoms = info.get("num_atoms", pd.NA)
        n_motifs = len(motif_list)

        for motif in motif_list:
            motif_local_id = motif.get("motif_id", None)
            motif_smiles = motif.get("smiles", None)
            atom_indices = motif.get("atoms", [])

            rows.append({
                "motif_id": f"motif_{global_motif_counter:08d}",
                "molecule_id": molecule_id,
                "smiles": smiles,
                "motif_local_id": motif_local_id,
                "motif_smiles": motif_smiles,
                "atom_indices": str(atom_indices),
                "num_atoms_in_molecule": num_atoms,
                "motif_edges": str(motif_edges),
                "decomposition_method": decomposition_method,
                "n_motifs_in_molecule": n_motifs,
            })
            global_motif_counter += 1

    motif_table = pd.DataFrame(rows)
    return motif_table
#%%
# =========================================================
# 5. motif_assay_table
# =========================================================
def build_motif_assay_table(
    molecule_assay_table: pd.DataFrame,
    motif_table: pd.DataFrame,
) -> pd.DataFrame:
    pair_df = molecule_assay_table.merge(
        motif_table[
            [
                "motif_id",
                "molecule_id",
                "motif_smiles",
                "motif_local_id",
                "atom_indices",
                "decomposition_method",
                "n_motifs_in_molecule",
            ]
        ],
        on="molecule_id",
        how="left",
    ).copy()

    pair_df = pair_df.reset_index(drop=True)
    pair_df["motif_assay_id"] = [f"msa_{i:08d}" for i in range(len(pair_df))]

    keep_cols = [
        "motif_assay_id",
        "motif_id",
        "molecule_id",
        "smiles",
        "motif_smiles",
        "motif_local_id",
        "atom_indices",
        "decomposition_method",
        "n_motifs_in_molecule",
        "assay_id",
        "assay",
        "label",
        "split",
    ]
    keep_cols = [c for c in keep_cols if c in pair_df.columns]

    return pair_df[keep_cols].copy()
#%%
def check_assay_embedding_alignment(
    assay_table: pd.DataFrame,
    assay_embedding_npy: Optional[str],
):
    if assay_embedding_npy is None or not os.path.exists(assay_embedding_npy):
        print("[Info] assay embedding npy not provided or not found.")
        return

    emb = np.load(assay_embedding_npy)
    print(f"[Info] assay embedding shape: {emb.shape}")
    print(f"[Info] assay_table rows: {len(assay_table)}")

    if len(emb) != len(assay_table):
        print("[Warning] assay embedding row count and assay_table row count do not match.")
#%%
def print_summary(
    assay_table: pd.DataFrame,
    molecule_assay_table: pd.DataFrame,
    motif_table: pd.DataFrame,
    motif_assay_table: pd.DataFrame,
):
    print("\n===== Summary =====")
    print(f"assays               : {len(assay_table)}")
    print(f"molecule-assay rows  : {len(molecule_assay_table)}")
    print(f"unique molecules     : {molecule_assay_table['molecule_id'].nunique()}")
    print(f"unique assays        : {molecule_assay_table['assay_id'].nunique()}")
    print(f"motifs               : {len(motif_table)}")
    print(f"motif-assay rows     : {len(motif_assay_table)}")

    if "split" in molecule_assay_table.columns:
        print("\n[molecule_assay_table split counts]")
        print(molecule_assay_table["split"].value_counts(dropna=False))

    if "split" in motif_assay_table.columns:
        print("\n[motif_assay_table split counts]")
        print(motif_assay_table["split"].value_counts(dropna=False))
#%%
def save_tables(
    out_dir: str,
    assay_table: pd.DataFrame,
    molecule_assay_table: pd.DataFrame,
    motif_table: pd.DataFrame,
    motif_assay_table: pd.DataFrame,
):
    os.makedirs(out_dir, exist_ok=True)

    assay_path = os.path.join(out_dir, "assay_table.csv")
    molecule_assay_path = os.path.join(out_dir, "molecule_assay_table.csv")
    motif_path = os.path.join(out_dir, "motif_table.csv")
    motif_assay_path = os.path.join(out_dir, "motif_assay_table.csv")

    assay_table.to_csv(assay_path, index=False)
    molecule_assay_table.to_csv(molecule_assay_path, index=False)
    motif_table.to_csv(motif_path, index=False)
    motif_assay_table.to_csv(motif_assay_path, index=False)

    print(f"[Saved] {assay_path}")
    print(f"[Saved] {molecule_assay_path}")
    print(f"[Saved] {motif_path}")
    print(f"[Saved] {motif_assay_path}")
#%%
# =========================================================
# 8. main
# =========================================================
def main():
    args = get_args()

    print("\n[1] Build assay_table")
    assay_table = build_assay_table(
        assay_meta_csv=args.assay_meta_csv,
        assay_embedding_npy=args.assay_embedding_npy,
    )
    print(assay_table.head())

    print("\n[2] Check assay embedding alignment")
    check_assay_embedding_alignment(
        assay_table=assay_table,
        assay_embedding_npy=args.assay_embedding_npy,
    )

    print("\n[3] Build molecule_assay_table")
    molecule_assay_table = build_molecule_assay_table(
        all_csv=args.all_csv,
        assay_table=assay_table,
    )
    print(molecule_assay_table.head())

    print("\n[4] Load motif mapping")
    motif_mapping = load_motif_mapping(args.motif_mapping_pkl)
    print(f"[Info] motif mapping loaded: {len(motif_mapping)} smiles")

    print("\n[5] Build motif_table")
    motif_table = build_motif_table(
        molecule_assay_table=molecule_assay_table,
        motif_mapping=motif_mapping,
        decomposition_method=args.decomposition_method,
    )
    print(motif_table.head())

    print("\n[6] Build motif_assay_table")
    motif_assay_table = build_motif_assay_table(
        molecule_assay_table=molecule_assay_table,
        motif_table=motif_table,
    )
    print(motif_assay_table.head())

    print_summary(
        assay_table=assay_table,
        molecule_assay_table=molecule_assay_table,
        motif_table=motif_table,
        motif_assay_table=motif_assay_table,
    )

    print("\n[7] Save tables")
    save_tables(
        out_dir=args.out_dir,
        assay_table=assay_table,
        molecule_assay_table=molecule_assay_table,
        motif_table=motif_table,
        motif_assay_table=motif_assay_table,
    )

    print("\nDone.")
#%%
if __name__ == "__main__":
    main()