#%%
from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from datasets.utils import smiles_to_data_GrphOnly

class ToxCastSharedDataset(InMemoryDataset):
    """
    One graph per molecule.
    y:    [1, T]
    mask: [1, T]
    split stored as python attribute
    """
    def __init__(
        self,
        toxcast_all_csv: str,
        assay_table_csv: str,
        transform=None,
        pre_transform=None,
    ):
        self.toxcast_all_csv = toxcast_all_csv
        self.assay_table_csv = assay_table_csv

        df = pd.read_csv(toxcast_all_csv)
        assay_table = pd.read_csv(assay_table_csv)

        self.task_names = assay_table["assay"].drop_duplicates().tolist()
        df = df[df["assay"].isin(self.task_names)].copy()

        # Check consistency of split assignments per molecule
        split_check = df.groupby("smiles")["split"].nunique()
        bad = split_check[split_check > 1]
        if len(bad) > 0:
            raise ValueError(
                f"Found molecules assigned to multiple splits. Example: {bad.index.tolist()[:10]}"
            )

        # pivot to multitask table
        base = df[["smiles", "split"]].drop_duplicates().reset_index(drop=True)

        y_pivot = df.pivot_table(
            index="smiles",
            columns="assay",
            values="label",
            aggfunc="first",
        )
        y_pivot = y_pivot.reindex(columns=self.task_names)

        m_pivot = y_pivot.notna().astype(int)

        self.multitask_table = base.merge(y_pivot.reset_index(), on="smiles", how="left")
        self.mask_table = base.merge(m_pivot.reset_index(), on="smiles", how="left")

        self.multitask_table["molecule_id"] = [
            f"mol_{i:06d}" for i in range(len(self.multitask_table))
        ]

        super().__init__(None, transform, pre_transform)

        data_list = []
        for i in range(len(self.multitask_table)):
            row_y = self.multitask_table.iloc[i]
            row_m = self.mask_table.iloc[i]

            smiles = row_y["smiles"]
            split = row_y["split"]
            molecule_id = row_y["molecule_id"]

            try:
                data = smiles_to_data_GrphOnly(smiles)
            except Exception:
                continue

            y = []
            mask = []
            for t in self.task_names:
                y_val = row_y[t]
                m_val = row_m[t]

                if pd.isna(y_val):
                    y.append(0.0)
                else:
                    y.append(float(y_val))

                mask.append(float(m_val))

            data.y = torch.tensor(y, dtype=torch.float).view(1, -1)
            data.mask = torch.tensor(mask, dtype=torch.float).view(1, -1)
            data.idx = torch.tensor([i], dtype=torch.long)

            data.smiles = smiles
            data.split = split
            data.molecule_id = molecule_id

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

    def get_split_indices(self):
        train_idx, valid_idx, test_idx = [], [], []

        for i in range(len(self)):
            d = self.get(i)
            if d.split == "train":
                train_idx.append(i)
            elif d.split == "valid":
                valid_idx.append(i)
            elif d.split == "test":
                test_idx.append(i)

        return (
            np.array(train_idx, dtype=int),
            np.array(valid_idx, dtype=int),
            np.array(test_idx, dtype=int),
        )