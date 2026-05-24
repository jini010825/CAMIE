import numpy as np
import torch

def zero_mask_motif_nodes(data, atom_indices):
    data = data.clone()

    if len(atom_indices) == 0:
        return data

    atom_indices = torch.tensor(atom_indices, dtype=torch.long, device=data.x.device)
    atom_indices = atom_indices[atom_indices < data.x.size(0)]

    if len(atom_indices) == 0:
        return data

    data.x[atom_indices] = 0
    return data

def zero_mask_atom_indices(data, atom_indices):
    """
    data.x shape: [num_nodes, feat_dim]
    atom_indices: list[int] or np.ndarray
    """
    data = data.clone()

    if atom_indices is None:
        return data

    if len(atom_indices) == 0:
        return data

    atom_indices = np.array(sorted(set(atom_indices)), dtype=np.int64)
    atom_indices = atom_indices[atom_indices < data.x.size(0)]

    if len(atom_indices) == 0:
        return data

    atom_indices_t = torch.tensor(atom_indices, dtype=torch.long, device=data.x.device)
    data.x[atom_indices_t] = 0
    return data


def collect_atom_indices_for_set(group_df, motif_mapping_for_smiles, set_name: str):
    """
    group_df: decomposition rows for one (molecule, assay, rule, k)
    motif_mapping_for_smiles: motif_mapping[smiles]
    set_name: "S1" or "S2"
    """
    motif_list = motif_mapping_for_smiles.get("motifs", [])
    motif_id_to_atoms = {int(m["motif_id"]): m.get("atoms", []) for m in motif_list}

    selected = group_df[group_df["set_assignment"] == set_name].copy()
    local_ids = selected["motif_local_id"].astype(int).tolist() if "motif_local_id" in selected.columns else []

    atom_union = []
    for mid in local_ids:
        atom_union.extend(motif_id_to_atoms.get(int(mid), []))

    atom_union = sorted(set(atom_union))
    return atom_union