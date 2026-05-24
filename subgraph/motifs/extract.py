#%%
from __future__ import annotations

import os
import pickle
import argparse
from typing import Dict, Any, List

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from utils.chemutils import brics_decomp, tree_decomp
#%%
def get_args():
    p = argparse.ArgumentParser("Extract hierarchical motifs for target ToxCast assay subset")

    p.add_argument(
        "--toxcast_all_csv",
        type=str,
        default="datasets/raw_data/chem_dataset/toxcast_all.csv",
    )
    p.add_argument(
        "--assay_table_csv",
        type=str,
        default="datasets/processed/toxcast/hierarchical/assay_table.csv",
    )
    p.add_argument(
        "--method",
        type=str,
        default="brics",
        choices=["brics", "tree"],
    )
    p.add_argument(
        "--output_path",
        type=str,
        default="assets/motif_split/smiles_to_hierarchical_mapping.pkl",
    )
    return p.parse_args()
#%%
def extract_hierarchical_info(smiles: str, method: str = "brics") -> Dict[str, Any] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if method == "brics":
        cliques, edges = brics_decomp(mol)
    elif method == "tree":
        cliques, edges = tree_decomp(mol)
    else:
        raise ValueError(f"Unknown decomposition method: {method}")

    motif_list = []
    for i, atom_indices in enumerate(cliques):
        motif_smiles = Chem.MolFragmentToSmiles(mol, atom_indices)
        motif_list.append({
            "motif_id": i,
            "smiles": motif_smiles,
            "atoms": atom_indices,
        })

    return {
        "motifs": motif_list,
        "motif_edges": edges,
        "num_atoms": mol.GetNumAtoms(),
    }
#%%
def load_target_smiles(
    toxcast_all_csv: str,
    assay_table_csv: str,
) -> List[str]:
    """
    Extract unique smiles corresponding to target assays (30 assays)
    based on toxcast_all.csv + assay_table.csv
    """
    df = pd.read_csv(toxcast_all_csv).copy()
    assay_table = pd.read_csv(assay_table_csv).copy()

    if "assay" not in df.columns:
        raise ValueError(f"'assay' column not found in {toxcast_all_csv}")
    if "smiles" not in df.columns:
        raise ValueError(f"'smiles' column not found in {toxcast_all_csv}")
    if "assay" not in assay_table.columns:
        raise ValueError(f"'assay' column not found in {assay_table_csv}")

    df["assay"] = df["assay"].astype(str).str.strip()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    assay_table["assay"] = assay_table["assay"].astype(str).str.strip()

    target_assays = assay_table["assay"].drop_duplicates().tolist()
    df = df[df["assay"].isin(target_assays)].copy()

    unique_smiles = df["smiles"].dropna().drop_duplicates().tolist()

    print(f"[INFO] target assays        : {len(target_assays)}")
    print(f"[INFO] filtered rows        : {len(df)}")
    print(f"[INFO] unique target smiles : {len(unique_smiles)}")

    return unique_smiles
#%%
def main():
    args = get_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    unique_smiles = load_target_smiles(
        toxcast_all_csv=args.toxcast_all_csv,
        assay_table_csv=args.assay_table_csv,
    )

    mapping_results = {}
    failed_smiles = []

    print(f"[INFO] Extracting hierarchical motifs using '{args.method}' ...")

    for smiles in tqdm(unique_smiles, total=len(unique_smiles)):
        info = extract_hierarchical_info(smiles, method=args.method)
        if info is not None:
            mapping_results[smiles] = info
        else:
            failed_smiles.append(smiles)

    with open(args.output_path, "wb") as f:
        pickle.dump(mapping_results, f)

    print(f"[OK] saved mapping: {args.output_path}")
    print(f"[INFO] success count : {len(mapping_results)}")
    print(f"[INFO] failed count  : {len(failed_smiles)}")

    if len(failed_smiles) > 0:
        failed_path = os.path.splitext(args.output_path)[0] + "_failed_smiles.txt"
        with open(failed_path, "w") as f:
            for s in failed_smiles:
                f.write(str(s) + "\n")
        print(f"[INFO] failed smiles saved to: {failed_path}")
#%%
if __name__ == "__main__":
    main()
# %%