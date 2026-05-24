#%%
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import wandb

from utils.utils import set_seed
from baselines.common.utils import save_json

#%%
def get_args():
    p = argparse.ArgumentParser("Train joint MLP scorer for context-conditioned motif importance")

    p.add_argument("--scoring_table_dir", type=str, default="assets/scoring/scoring_dataset")
    p.add_argument("--graph_emb_dir", type=str, default="assets/toxcast_gnn/gnn_emb")
    p.add_argument("--motif_emb_dir", type=str, default="assets/toxcast_gnn/motif_emb")
    p.add_argument("--assay_emb_npy", type=str, default="assets/assay_emb/biobert_emb/hierarchy_core_mean.npy")

    p.add_argument(
        "--target_col",
        type=str,
        default="pseudo_target_logit_diff",
        choices=["pseudo_target_logit_diff", "pseudo_target_prob_diff"],
    )

    p.add_argument("--out_dir", type=str, default="assets/scoring/joint_mlp")

    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_mode", type=str, default="all", choices=["train", "valid", "test", "all"])
    p.add_argument("--num_workers", type=int, default=4)



    # wandb (minimal)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="joint-mlp-scorer")
    p.add_argument("--wandb_name", type=str, default=None)

    return p.parse_args()

#%%
class ScoringDataset(Dataset):
    def __init__(
        self,
        table_df: pd.DataFrame,
        graph_emb: np.ndarray,
        motif_emb: np.ndarray,
        assay_emb: np.ndarray,
        target_col: str,
    ):
        self.df = table_df.reset_index(drop=True)
        self.graph_emb = graph_emb
        self.motif_emb = motif_emb
        self.assay_emb = assay_emb
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        g_idx = int(row["graph_emb_idx"])
        m_idx = int(row["motif_emb_idx"])
        a_idx = int(row["assay_emb_idx"])

        z_g = self.graph_emb[g_idx].astype(np.float32)
        z_m = self.motif_emb[m_idx].astype(np.float32)
        z_a = self.assay_emb[a_idx].astype(np.float32)

        y = np.float32(row[self.target_col])

        return {
            "z_m": torch.tensor(z_m, dtype=torch.float32),
            "z_g": torch.tensor(z_g, dtype=torch.float32),
            "z_a": torch.tensor(z_a, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "motif_assay_id": row["motif_assay_id"],
        }

#%%
class JointMLPScorer(nn.Module):
    def __init__(self, motif_dim=300, graph_dim=300, assay_dim=768,
                 hidden_dim=256, num_layers=3, dropout=0.2):
        super().__init__()

        self.assay_proj = nn.Linear(assay_dim, motif_dim)

        input_dim = motif_dim + graph_dim + motif_dim + motif_dim + motif_dim
        # zm, zg, za_proj, zm*zg, zm*za_proj

        self.input_norm = nn.LayerNorm(input_dim)

        layers = []
        dim_in = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim_in = hidden_dim

        layers.append(nn.Linear(dim_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_m, z_g, z_a):
        z_a_proj = self.assay_proj(z_a)

        z_mg = z_m * z_g
        z_ma = z_m * z_a_proj

        x = torch.cat([z_m, z_g, z_a_proj, z_mg, z_ma], dim=1)
        x = self.input_norm(x)
        return self.net(x).view(-1)

#%%
def maybe_init_wandb(args):
    if not args.use_wandb:
        return

    run_name = args.wandb_name
    if run_name is None:
        run_name = f"mse_{args.target_col}_seed{args.seed}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "target_col": args.target_col,
            "seed": args.seed,
            "split_mode": args.split_mode,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    )

#%%
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    ys, preds = [], []

    for batch in loader:
        z_m = batch["z_m"].to(device)
        z_g = batch["z_g"].to(device)
        z_a = batch["z_a"].to(device)
        y = batch["y"].to(device)

        pred = model(z_m, z_g, z_a)

        ys.append(y.cpu().numpy())
        preds.append(pred.cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    pred_all = np.concatenate(preds, axis=0)

    mse = float(np.mean((pred_all - y_all) ** 2))
    mae = float(np.mean(np.abs(pred_all - y_all)))

    if len(y_all) > 1 and np.std(y_all) > 0 and np.std(pred_all) > 0:
        corr = float(np.corrcoef(y_all, pred_all)[0, 1])
    else:
        corr = float("nan")

    return {
        "mse": mse,
        "mae": mae,
        "corr": corr,
        "y_all": y_all,
        "pred_all": pred_all,
    }

#%%
def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in loader:
        z_m = batch["z_m"].to(device)
        z_g = batch["z_g"].to(device)
        z_a = batch["z_a"].to(device)
        y = batch["y"].to(device)

        pred = model(z_m, z_g, z_a)
        loss = nn.functional.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += float(loss.item()) * batch_size
        total_n += batch_size

    return total_loss / max(total_n, 1)

#%%
def main():
    t_start = time.perf_counter()

    args = get_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb init
    maybe_init_wandb(args)

    # load files
    scoring_table_csv = Path(args.scoring_table_dir)
    #scoring_table_csv = Path(args.scoring_table_dir) / f"motif_context_scoring_table_seed{args.seed}.csv"
    df = pd.read_csv(scoring_table_csv)

    graph_emb_path = Path(args.graph_emb_dir) / f"seed{args.seed}" / f"{args.split_mode}_graph_emb.npy"
    graph_emb = np.load(graph_emb_path)

    motif_emb_path = Path(args.motif_emb_dir) / f"seed{args.seed}" / f"motif_emb.npy"
    motif_emb = np.load(motif_emb_path)

    assay_emb = np.load(args.assay_emb_npy)

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"[INFO] train rows: {len(train_df)}")
    print(f"[INFO] valid rows: {len(valid_df)}")
    print(f"[INFO] test rows : {len(test_df)}")

    train_set = ScoringDataset(train_df, graph_emb, motif_emb, assay_emb, args.target_col)
    valid_set = ScoringDataset(valid_df, graph_emb, motif_emb, assay_emb, args.target_col)
    test_set = ScoringDataset(test_df, graph_emb, motif_emb, assay_emb, args.target_col)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    motif_dim = motif_emb.shape[1]
    graph_dim = graph_emb.shape[1]
    assay_dim = assay_emb.shape[1]

    model = JointMLPScorer(
        motif_dim=motif_dim,
        graph_dim=graph_dim,
        assay_dim=assay_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mse = float("inf")
    best_state = None
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        valid_result = evaluate(model, valid_loader, device)
        test_result = evaluate(model, test_loader, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_mse={train_loss:.6f} "
            f"val_mse={valid_result['mse']:.6f} "
            f"val_mae={valid_result['mae']:.6f} "
            f"val_corr={valid_result['corr']:.4f} "
            f"test_mse={test_result['mse']:.6f} "
            f"test_corr={test_result['corr']:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": valid_result["mse"],
            "val_mae": valid_result["mae"],
            "val_corr": valid_result["corr"],
            "test_mse": test_result["mse"],
            "test_mae": test_result["mae"],
            "test_corr": test_result["corr"],
        })

        if valid_result["mse"] < best_val_mse:
            best_val_mse = valid_result["mse"]
            best_state = model.state_dict()
            torch.save(best_state, out_dir / f"best_joint_mlp_scorer_seed{args.seed}.pt")
            print(f"[OK] saved best model: {out_dir / f'best_joint_mlp_scorer_seed{args.seed}.pt'}")

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/mse": train_loss,
                    "valid/mse": valid_result["mse"],
                    "valid/mae": valid_result["mae"],
                    "valid/corr": valid_result["corr"],
                    "best/val_mse_so_far": best_val_mse,
                },
                step=epoch,
            )

    log_csv = out_dir / f"train_log_seed{args.seed}.csv"
    pd.DataFrame(log_rows).to_csv(log_csv, index=False)

    # best model eval
    model.load_state_dict(best_state)

    valid_result = evaluate(model, valid_loader, device)
    test_result = evaluate(model, test_loader, device)
    
    train_seconds = time.perf_counter() - t_start

    summary = {
        "best_val_mse": valid_result["mse"],
        "best_val_mae": valid_result["mae"],
        "best_val_corr": valid_result["corr"],
        "test_mse": test_result["mse"],
        "test_mae": test_result["mae"],
        "test_corr": test_result["corr"],
    }

    runtime = {
        "stage": "train",
        "method": "mse",
        "train_seconds": float(train_seconds),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
    }

    save_json(runtime, out_dir / f"runtime_train_seed{args.seed}.json")
    save_json(summary, out_dir / f"summary_seed{args.seed}.json")
    print("[OK] saved summary:", out_dir / f"summary_seed{args.seed}.json")

    if args.use_wandb:
        wandb.summary["best_val_mse"] = valid_result["mse"]
        wandb.summary["best_val_corr"] = valid_result["corr"]
        wandb.summary["test_mse"] = test_result["mse"]
        wandb.summary["test_corr"] = test_result["corr"]
        wandb.finish()

#%%
if __name__ == "__main__":
    main()