from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Sequence, Iterable

import numpy as np
import pandas as pd
import torch


# -------------------------------------------------
# common keys
# -------------------------------------------------
GROUP_COLS = ["molecule_id", "smiles", "assay", "split", "label"]
RULE_GROUP_COLS = GROUP_COLS + ["rule", "k"]

REQUIRED_MOTIF_COLUMNS = [
    "motif_assay_id",
    "motif_id",
    "motif_local_id",
    "molecule_id",
    "motif_smiles",
    "smiles",
    "assay",
    "split",
    "label",
]


# -------------------------------------------------
# path / io
# -------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


# -------------------------------------------------
# reproducibility
# -------------------------------------------------
def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# parsing helpers
# -------------------------------------------------
def parse_atom_indices(x) -> list[int]:
    """
    Parse atom_indices column.

    Supports:
    - list[int]
    - tuple[int]
    - np.ndarray
    - string like "[1, 2, 3]"
    - string like "1,2,3"
    """
    if x is None:
        return []

    if isinstance(x, float) and np.isnan(x):
        return []

    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]

    if isinstance(x, (list, tuple, set)):
        return [int(v) for v in x]

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set, np.ndarray)):
                return [int(v) for v in parsed]
            if isinstance(parsed, int):
                return [int(parsed)]
        except Exception:
            pass

        # fallback: "1,2,3"
        s = s.replace("[", "").replace("]", "")
        if s.strip() == "":
            return []
        return [int(v.strip()) for v in s.split(",") if v.strip() != ""]

    raise TypeError(f"Unsupported atom_indices type: {type(x)}")


def validate_columns(df: pd.DataFrame, required_cols: Sequence[str], name: str = "df") -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


# -------------------------------------------------
# selection utils
# -------------------------------------------------
def topk_idx(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape={scores.shape}")

    if len(scores) == 0:
        return np.array([], dtype=int)

    # Send NaNs to the bottom
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)

    k = max(1, min(int(k), len(scores)))

    # stable tie-breaking by original index
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:k]


def assign_s1_s2_by_topk(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    selected = set(topk_idx(scores, k).tolist())
    return np.array(
        ["S2" if i in selected else "S1" for i in range(len(scores))],
        dtype=object,
    )


def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    if np.isnan(x_min) or np.isnan(x_max):
        return np.zeros_like(x, dtype=float)

    return (x - x_min) / (x_max - x_min + eps)


def zscore_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x

    mean = np.nanmean(x)
    std = np.nanstd(x)

    if np.isnan(mean) or np.isnan(std):
        return np.zeros_like(x, dtype=float)

    return (x - mean) / (std + eps)


# -------------------------------------------------
# optional filtering helpers
# -------------------------------------------------
def filter_by_k_and_rule(
    df: pd.DataFrame,
    k: int | None = None,
    rules: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if k is not None and "k" in out.columns:
        out = out[out["k"] == int(k)].copy()

    if rules is not None and "rule" in out.columns:
        out = out[out["rule"].isin(rules)].copy()

    return out


def filter_pairs_by_min_num_motifs(
    pair_df: pd.DataFrame,
    min_num_motifs: int = 1,
) -> pd.DataFrame:
    if min_num_motifs <= 1:
        return pair_df.copy()

    if "num_motifs" not in pair_df.columns:
        raise ValueError("pair_df must have 'num_motifs' column")

    return pair_df[pair_df["num_motifs"] >= int(min_num_motifs)].copy()


# -------------------------------------------------
# pooling helpers
# -------------------------------------------------
def mean_pool_or_zero(x: np.ndarray, emb_dim: int) -> tuple[np.ndarray, int]:
    if x.shape[0] == 0:
        return np.zeros((emb_dim,), dtype=np.float32), 1
    return x.mean(axis=0).astype(np.float32), 0


# -------------------------------------------------
# internal quality metrics
# -------------------------------------------------
def compute_gap(values: np.ndarray, selected_idx: np.ndarray) -> float:
    """
    mean(S2) - mean(S1)
    """
    values = np.asarray(values, dtype=float)
    selected_idx = np.asarray(selected_idx, dtype=int)

    mask = np.zeros(len(values), dtype=bool)
    mask[selected_idx] = True

    s2 = values[mask]
    s1 = values[~mask]

    if len(s2) == 0 or len(s1) == 0:
        return np.nan

    return float(np.mean(s2) - np.mean(s1))


def compute_oracle_sets(
    values: np.ndarray,
    k: int,
    tie_break_ids: Iterable,
) -> set[int]:
    """
    Oracle top-k by values desc, then tie_break_ids asc.
    Returns a set of row indices.
    """
    tmp = pd.DataFrame({
        "value": np.asarray(values, dtype=float),
        "tie_id": list(tie_break_ids),
        "idx": np.arange(len(values)),
    }).sort_values(
        by=["value", "tie_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    oracle_k = min(max(1, int(k)), len(tmp))
    return set(tmp.loc[:oracle_k - 1, "idx"].tolist())


def compute_oracle_overlap(
    values: np.ndarray,
    selected_idx: np.ndarray,
    k: int,
    tie_break_ids: Iterable,
) -> float:
    oracle_set = compute_oracle_sets(values, k, tie_break_ids)
    pred_set = set(np.asarray(selected_idx, dtype=int).tolist())

    inter = len(pred_set & oracle_set)
    denom = max(min(int(k), len(values)), 1)
    return inter / denom


def compute_oracle_jaccard(
    values: np.ndarray,
    selected_idx: np.ndarray,
    k: int,
    tie_break_ids: Iterable,
) -> float:
    oracle_set = compute_oracle_sets(values, k, tie_break_ids)
    pred_set = set(np.asarray(selected_idx, dtype=int).tolist())

    inter = len(pred_set & oracle_set)
    union = len(pred_set | oracle_set)
    return inter / max(union, 1)


def compute_exact_match(
    values: np.ndarray,
    selected_idx: np.ndarray,
    k: int,
    tie_break_ids: Iterable,
) -> float:
    oracle_set = compute_oracle_sets(values, k, tie_break_ids)
    pred_set = set(np.asarray(selected_idx, dtype=int).tolist())
    return float(pred_set == oracle_set)