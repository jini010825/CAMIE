#%%
from __future__ import annotations

import torch
from typing import Optional

from torch_geometric.data import Data
from rdkit import Chem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
#%%
def smiles_to_data(smiles: str, y: int, idx: int) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.long)

    edges = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_to_feature_vector(bond)

        edges.append((i, j))
        edge_attrs.append(bf)
        edges.append((j, i))
        edge_attrs.append(bf)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y)], dtype=torch.float),
    )
    data.idx = int(idx)
    return data
#%%
def smiles_to_data_GrphOnly(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = torch.tensor(atom_features_list, dtype=torch.long)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_to_feature_vector(bond)

        edge_indices.append((i, j))
        edge_indices.append((j, i))
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data