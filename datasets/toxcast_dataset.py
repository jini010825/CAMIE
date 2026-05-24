#%%
import os
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import deepchem as dc
#%%
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split_mode", type=str, default="scaffold", choices=["random", "scaffold", "stratified", "index"])
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="datasets/raw_data/chem_dataset")
    return p.parse_args()
#%%
def build_toxcast_long_df(
    dataset,
    tasks,
    split_name: str,
):
    """
    DeepChem dataset -> long-format DataFrame

    output columns:
    - smiles
    - assay
    - label
    - split
    """
    ids = dataset.ids
    y = dataset.y
    w = dataset.w if hasattr(dataset, "w") else None

    rows = []

    n_samples = len(ids)
    n_tasks = len(tasks)

    for i in range(n_samples):
        smiles = ids[i]

        for j in range(n_tasks):
            assay = tasks[j]
            label = y[i, j]

            weight = None
            if w is not None:
                weight = w[i, j]

            # Determine labeled mask
            is_missing = False

            if label is None:
                is_missing = True
            elif isinstance(label, float) and np.isnan(label):
                is_missing = True
            elif weight is not None and weight <= 0:
                is_missing = True

            # Discard unlabeled
            if is_missing:
                continue

            rows.append({
                "smiles": smiles,
                "assay": assay,
                "label": label,
                "split": split_name,
            })

    df = pd.DataFrame(rows)
    return df
#%%
def load_toxcast_unsplit(featurizer: str = "ECFP"):
    tasks, datasets, transformers = dc.molnet.load_toxcast(
        featurizer=featurizer,
        splitter=None,
        reload=True,
    )

    if isinstance(datasets, (list, tuple)):
        dataset = datasets[0]
    else:
        dataset = datasets

    return tasks, dataset, transformers
#%%
def split_toxcast_dataset(
    dataset,
    split_mode: str = "random",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    split_mode:
    - random
    - scaffold
    - stratified
    - index
    """
    split_mode = split_mode.lower()

    if split_mode == "random":
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    elif split_mode == "scaffold":
        splitter = dc.splits.ScaffoldSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    elif split_mode == "stratified":
        # Use the random stratified splitter provided by DeepChem for multitask datasets
        splitter = dc.splits.RandomStratifiedSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    elif split_mode == "index":
        splitter = dc.splits.IndexSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    else:
        raise ValueError(
            f"Unsupported split_mode: {split_mode}. "
            f"Choose from ['random', 'scaffold', 'stratified', 'index']"
        )

    return train_dataset, valid_dataset, test_dataset
#%%
def save_split_csvs(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "toxcast_train.csv")
    valid_path = os.path.join(out_dir, "toxcast_valid.csv")
    test_path = os.path.join(out_dir, "toxcast_test.csv")
    all_path = os.path.join(out_dir, "toxcast_all.csv")

    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    df_train.to_csv(train_path, index=False)
    df_valid.to_csv(valid_path, index=False)
    df_test.to_csv(test_path, index=False)
    df_all.to_csv(all_path, index=False)

    print(f"[Saved] {train_path}")
    print(f"[Saved] {valid_path}")
    print(f"[Saved] {test_path}")
    print(f"[Saved] {all_path}")

    return train_path, valid_path, test_path, all_path
#%%
def print_summary(df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame):
    print("\n===== Split Summary =====")
    print(f"train rows: {len(df_train)}")
    print(f"valid rows: {len(df_valid)}")
    print(f"test rows : {len(df_test)}")

    print("\n===== Unique molecules =====")
    print(f"train smiles: {df_train['smiles'].nunique()}")
    print(f"valid smiles: {df_valid['smiles'].nunique()}")
    print(f"test smiles : {df_test['smiles'].nunique()}")

    print("\n===== Unique assays =====")
    print(f"train assays: {df_train['assay'].nunique()}")
    print(f"valid assays: {df_valid['assay'].nunique()}")
    print(f"test assays : {df_test['assay'].nunique()}")

    print("\n===== Example =====")
    print(df_train.head())
#%%
def main():
    args = get_args()

    print("\n[1] Loading unsplit ToxCast dataset ...")
    tasks, dataset, transformers = load_toxcast_unsplit()

    print(f"[Info] number of tasks: {len(tasks)}")
    print(f"[Info] total molecules: {len(dataset)}")

    print(f"\n[2] Splitting dataset with mode='{args.split_mode}' ...")
    train_dataset, valid_dataset, test_dataset = split_toxcast_dataset(
        dataset=dataset,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n[3] Building long-format DataFrames ...")
    df_train = build_toxcast_long_df(train_dataset, tasks, split_name="train")
    df_valid = build_toxcast_long_df(valid_dataset, tasks, split_name="valid")
    df_test = build_toxcast_long_df(test_dataset, tasks, split_name="test")

    print_summary(df_train, df_valid, df_test)

    print("\n[4] Saving CSV files ...")
    save_split_csvs(df_train, df_valid, df_test, args.out_dir)

    print("\nDone.")

#%%
if __name__ == "__main__":
    main()
# %%