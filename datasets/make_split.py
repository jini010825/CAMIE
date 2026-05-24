#%%
from __future__ import annotations

import os
from pathlib import Path
import argparse

import pandas as pd

from datasets.split import split_key, make_split, save_split_npz
#%%
def get_args():
    p = argparse.ArgumentParser("Make train/val/test split from processed CSV")
    p.add_argument(
        "--csv",
        type=str,
        default="datasets/processed/processed_scar.csv",
        help="Path to processed csv containing SMILES and label columns",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="datasets/splits/scar",
        help="Directory to save split npz",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.2)
    return p.parse_args()
#%%
def main():
    args = get_args()

    df = pd.read_csv(args.csv)

    if "label" not in df.columns:
        raise ValueError("CSV must contain column: 'label'")

    y = df["label"].astype(int).values

    train_idx, val_idx, test_idx = make_split(
        n=len(df),
        y=y,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    key = split_key(args.seed, args.train_ratio, args.val_ratio, args.test_ratio)
    csv_stem = Path(args.csv).stem
    out_path = os.path.join(args.out_dir, f"{csv_stem}_split_{key}.npz")

    save_split_npz(out_path, train_idx, val_idx, test_idx)

    print("[OK] saved:", out_path)
    print("split sizes:", len(train_idx), len(val_idx), len(test_idx))
    print(
        "pos rate:",
        float(y.mean()),
        "train/val/test pos:",
        float(y[train_idx].mean()),
        float(y[val_idx].mean()),
        float(y[test_idx].mean()),
    )
#%%
if __name__ == "__main__":
    main()