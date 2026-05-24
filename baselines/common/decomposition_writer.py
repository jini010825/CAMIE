from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from baselines.common.utils import (
    GROUP_COLS,
    ensure_dir,
    parse_atom_indices,
    topk_idx,
    validate_columns,
)


def _safe_get(row: pd.Series, col: str, default=np.nan):
    return row[col] if col in row.index else default


def build_decomposition_rows_for_group(
    group_df: pd.DataFrame,
    score_col: str,
    rule: str,
    k: int,
    higher_is_better: bool = True,
) -> list[dict]:
    """
    Build decomposition rows for one molecule-assay pair.

    Required output columns are intentionally compatible with existing
    fidelity evaluation code.
    """
    g = group_df.reset_index(drop=True).copy()

    if score_col not in g.columns:
        raise ValueError(f"score_col='{score_col}' not found in group_df")

    scores = g[score_col].astype(float).to_numpy()

    if not higher_is_better:
        scores_for_rank = -scores
    else:
        scores_for_rank = scores

    selected = set(topk_idx(scores_for_rank, k).tolist())

    # stable ranking: score desc, motif_local_id asc
    tie_col = "motif_local_id" if "motif_local_id" in g.columns else "motif_id"
    rank_df = pd.DataFrame({
        "idx": np.arange(len(g)),
        "score": scores_for_rank,
        "tie": g[tie_col].tolist(),
    }).sort_values(["score", "tie"], ascending=[False, True])

    rank_map = {
        int(idx): rank + 1
        for rank, idx in enumerate(rank_df["idx"].tolist())
    }

    rows = []
    for i, row in g.iterrows():
        set_assignment = "S2" if i in selected else "S1"

        rows.append({
            "motif_assay_id": _safe_get(row, "motif_assay_id"),
            "motif_id": _safe_get(row, "motif_id"),
            "motif_local_id": _safe_get(row, "motif_local_id"),
            "molecule_id": _safe_get(row, "molecule_id"),
            "motif_smiles": _safe_get(row, "motif_smiles"),
            "smiles": _safe_get(row, "smiles"),
            "assay": _safe_get(row, "assay"),
            "split": _safe_get(row, "split"),
            "label": _safe_get(row, "label"),
            "motif_emb_idx": _safe_get(row, "motif_emb_idx"),
            "rule": rule,
            "k": int(k),
            "rank_in_group": int(rank_map[i]),
            "score": float(scores[i]),
            "set_assignment": set_assignment,
            "pseudo_target_logit_diff": _safe_get(row, "pseudo_target_logit_diff"),
            "pseudo_target_prob_diff": _safe_get(row, "pseudo_target_prob_diff"),
            "num_atoms_in_motif": _safe_get(row, "num_atoms_in_motif"),
            "atom_indices": _safe_get(row, "atom_indices"),
        })

    return rows


def build_pooled_meta_rows_from_decomp(
    decomp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build pooled_representation_meta.csv from decomposition table.
    """
    required = GROUP_COLS + ["rule", "k", "set_assignment"]
    validate_columns(decomp_df, required, name="decomp_df")

    rows = []
    for keys, g in decomp_df.groupby(GROUP_COLS + ["rule", "k"], sort=False):
        molecule_id, smiles, assay, split, label, rule, k = keys

        num_motifs = int(len(g))
        s2_size = int((g["set_assignment"] == "S2").sum())
        s1_size = int((g["set_assignment"] == "S1").sum())

        pooled_id = f"{molecule_id}||{assay}||{rule}||k{k}"

        rows.append({
            "pooled_id": pooled_id,
            "molecule_id": molecule_id,
            "smiles": smiles,
            "assay": assay,
            "split": split,
            "label": label,
            "rule": rule,
            "k": int(k),
            "num_motifs": num_motifs,
            "s2_size": s2_size,
            "s1_size": s1_size,
            "s1_empty": int(s1_size == 0),
            "s2_empty": int(s2_size == 0),
        })

    return pd.DataFrame(rows)


def build_decomposition_table(
    scored_df: pd.DataFrame,
    score_col: str,
    rule: str,
    k_values: Iterable[int],
    group_cols: list[str] | None = None,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Convert motif-level scores into decomposition table for multiple k values.

    Args:
        scored_df:
            Motif-level dataframe containing one row per motif-assay pair.
        score_col:
            Column containing baseline score.
        rule:
            Baseline name. e.g., "sa", "gradcam", "gnnexplainer".
        k_values:
            Top-k motif values.
        group_cols:
            Default GROUP_COLS.
        higher_is_better:
            If False, smaller score is selected first.

    Returns:
        decomp_df
    """
    group_cols = group_cols or GROUP_COLS
    validate_columns(scored_df, group_cols + [score_col], name="scored_df")

    all_rows = []

    for k in k_values:
        for _, g in scored_df.groupby(group_cols, sort=False):
            rows = build_decomposition_rows_for_group(
                group_df=g,
                score_col=score_col,
                rule=rule,
                k=int(k),
                higher_is_better=higher_is_better,
            )
            all_rows.extend(rows)

    decomp_df = pd.DataFrame(all_rows)

    # stable column order
    preferred_cols = [
        "motif_assay_id",
        "motif_id",
        "motif_local_id",
        "molecule_id",
        "motif_smiles",
        "smiles",
        "assay",
        "split",
        "label",
        "motif_emb_idx",  
        "rule",
        "k",
        "rank_in_group",
        "score",
        "set_assignment",
        "pseudo_target_logit_diff",
        "pseudo_target_prob_diff",
        "num_atoms_in_motif",
        "atom_indices",
    ]

    cols = [c for c in preferred_cols if c in decomp_df.columns]
    extra_cols = [c for c in decomp_df.columns if c not in cols]
    return decomp_df[cols + extra_cols]


def save_decomposition_outputs(
    scored_df: pd.DataFrame,
    out_dir: str | Path,
    score_col: str,
    rule: str,
    k_values: Iterable[int],
    decomp_name: str = "motif_decomposition_table.csv",
    pooled_name: str = "pooled_representation_meta.csv",
) -> tuple[Path, Path]:
    """
    Build and save:
    - motif_decomposition_table.csv
    - pooled_representation_meta.csv
    """
    out_dir = ensure_dir(out_dir)

    decomp_df = build_decomposition_table(
        scored_df=scored_df,
        score_col=score_col,
        rule=rule,
        k_values=k_values,
    )
    pooled_df = build_pooled_meta_rows_from_decomp(decomp_df)

    decomp_path = out_dir / decomp_name
    pooled_path = out_dir / pooled_name

    decomp_df.to_csv(decomp_path, index=False)
    pooled_df.to_csv(pooled_path, index=False)

    return decomp_path, pooled_path


def build_decomposition_by_node_coverage(
    scored_df: pd.DataFrame,
    score_col: str,
    rule: str,
    coverage: float = 0.30,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Optional eXEL-style node coverage selection.
    Not recommended as main metric yet, but useful for robustness analysis.
    """
    group_cols = group_cols or GROUP_COLS
    rows = []

    for _, g in scored_df.groupby(group_cols, sort=False):
        g = g.reset_index(drop=True).copy()
        g = g.sort_values(score_col, ascending=False).reset_index(drop=True)

        # prefer num_atoms_in_molecule if available
        if "num_atoms_in_molecule" in g.columns:
            total_atoms = int(g["num_atoms_in_molecule"].iloc[0])
        else:
            all_atoms = set()
            for atoms in g["atom_indices"].tolist():
                all_atoms.update(parse_atom_indices(atoms))
            total_atoms = len(all_atoms)

        target_atoms = max(1, int(round(total_atoms * float(coverage))))

        selected_atoms = set()
        assignments = []

        for _, row in g.iterrows():
            atoms = set(parse_atom_indices(row["atom_indices"]))
            if len(selected_atoms) < target_atoms:
                assignments.append("S2")
                selected_atoms.update(atoms)
            else:
                assignments.append("S1")

        for i, row in g.iterrows():
            rows.append({
                **row.to_dict(),
                "rule": rule,
                "k": f"cov{coverage}",
                "rank_in_group": i + 1,
                "score": float(row[score_col]),
                "set_assignment": assignments[i],
                "coverage_target": float(coverage),
                "selected_atom_coverage": len(selected_atoms) / max(total_atoms, 1),
            })

    return pd.DataFrame(rows)