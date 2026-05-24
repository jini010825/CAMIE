#%%
from __future__ import annotations

import argparse
import os
from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from datasets.toxcast_graph_dataset import ToxCastSharedDataset
from models.gnn.gnn import GNN_graph
from utils.utils import set_seed
#%%
def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--csv", type=str, default="datasets/raw_data/chem_dataset/toxcast_all.csv")
    p.add_argument("--assay_table_csv", type=str, default="datasets/processed/toxcast/hierarchical/assay_table.csv")

    p.add_argument("--out_dir", type=str, default="assets/toxcast_gnn/gnn_preds")
    p.add_argument("--ckpt_dir", type=str, default="assets/toxcast_gnn/ckpt")

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--model", type=str, default="gin", choices=["gin", "gcn"])
    p.add_argument("--num_layer", type=int, default=5)
    p.add_argument("--emb_dim", type=int, default=300)
    p.add_argument("--drop_ratio", type=float, default=0.5)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--save_every", type=int, default=1)

    return p.parse_args()
#%%
def masked_multitask_bce_loss(logit, y, mask):
    """
    logit: [B, T]
    y:     [B, T]
    mask:  [B, T]
    # Compute BCE only for observed samples per task -> average over tasks
    """
    loss_mat = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, y, reduction="none"
    )  # [B, T]

    loss_mat = loss_mat * mask
    task_count = mask.sum(dim=0)  # [T]

    valid_task = task_count > 0
    if valid_task.sum() == 0:
        return logit.sum() * 0.0

    task_loss = torch.zeros_like(task_count, dtype=torch.float)
    task_loss[valid_task] = loss_mat[:, valid_task].sum(dim=0) / task_count[valid_task]

    return task_loss[valid_task].mean()
#%%
@torch.no_grad()
def eval_model(model, loader, device, task_names):
    model.eval()

    ys, ps, ms, idxs = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        logit = model(batch)  # [B, T]
        prob = torch.sigmoid(logit).cpu().numpy()

        y = batch.y.view(logit.size(0), -1).cpu().numpy()
        m = batch.mask.view(logit.size(0), -1).cpu().numpy()

        ys.append(y)
        ps.append(prob)
        ms.append(m)
        idxs.append(batch.idx.view(-1).cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(ps, axis=0)
    m_all = np.concatenate(ms, axis=0)
    idx_all = np.concatenate(idxs, axis=0)

    per_task_auc = []
    per_task_ap = []

    for t in range(len(task_names)):
        valid = m_all[:, t] > 0
        if valid.sum() == 0:
            per_task_auc.append(float("nan"))
            per_task_ap.append(float("nan"))
            continue

        y_t = y_all[valid, t]
        p_t = p_all[valid, t]

        if len(np.unique(y_t)) < 2:
            per_task_auc.append(float("nan"))
            per_task_ap.append(float("nan"))
            continue

        auc = roc_auc_score(y_t, p_t)
        ap = average_precision_score(y_t, p_t)

        per_task_auc.append(auc)
        per_task_ap.append(ap)

    macro_auc = float(np.nanmean(per_task_auc))
    macro_ap = float(np.nanmean(per_task_ap))

    return {
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "per_task_auc": per_task_auc,
        "per_task_ap": per_task_ap,
        "idx_all": idx_all,
        "y_all": y_all,
        "p_all": p_all,
        "m_all": m_all,
    }
#%%
def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ToxCastSharedDataset(
        toxcast_all_csv=args.csv,
        assay_table_csv=args.assay_table_csv,
    )
    task_names = dataset.task_names
    num_tasks = len(task_names)

    train_idx, val_idx, test_idx = dataset.get_split_indices()

    train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_set = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_set = torch.utils.data.Subset(dataset, test_idx.tolist())

    g = torch.Generator()
    g.manual_seed(int(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = GNN_graph(
        num_tasks=num_tasks,
        num_layer=args.num_layer,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        gnn_type=args.model,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = -1
    best_state = None

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    log_rows = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            logit = model(batch)  # [B, T]
            y = batch.y.view(logit.size(0), -1).float()
            mask = batch.mask.view(logit.size(0), -1).float()

            loss = masked_multitask_bce_loss(logit, y, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            n_train_batches += 1

        val_result = eval_model(model, val_loader, device, task_names)
        val_auc = val_result["macro_auc"]
        val_ap = val_result["macro_ap"]

        if val_auc > best_val:
            best_val = val_auc
            best_state = copy.deepcopy(model.state_dict())

        mean_train_loss = train_loss_sum / max(n_train_batches, 1)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={mean_train_loss:.4f} "
            f"val_macro_auc={val_auc:.4f} "
            f"val_macro_ap={val_ap:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_macro_auc": val_auc,
            "val_macro_ap": val_ap,
        })

        if (epoch % args.save_every) == 0 or (epoch == args.epochs - 1):
            ckpt_path = Path(args.ckpt_dir) / f"toxcast_shared_{args.model}_seed{args.seed}_epoch{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print("[CKPT] saved:", ckpt_path)

    model.load_state_dict(best_state)

    val_result = eval_model(model, val_loader, device, task_names)
    test_result = eval_model(model, test_loader, device, task_names)

    print(
        f"[BEST MODEL] "
        f"VAL AUC={val_result['macro_auc']:.4f} AP={val_result['macro_ap']:.4f} | "
        f"TEST AUC={test_result['macro_auc']:.4f} AP={test_result['macro_ap']:.4f}"
    )

    # save best checkpoint
    best_ckpt_path = Path(args.ckpt_dir) / f"toxcast_shared_{args.model}_best_seed{args.seed}.pt"
    torch.save(best_state, best_ckpt_path)
    print("[OK] saved best ckpt:", best_ckpt_path)

    # save log
    pd.DataFrame(log_rows).to_csv(Path(args.out_dir) / f"trainlog_toxcast_shared_{args.model}_seed{args.seed}.csv", index=False)

    # save summary metrics
    metrics_path = Path(args.out_dir) / f"metrics_toxcast_shared_{args.model}_seed{args.seed}.csv"
    pd.DataFrame([{
        "model": args.model,
        "seed": int(args.seed),
        "val_macro_auc": float(val_result["macro_auc"]),
        "val_macro_ap": float(val_result["macro_ap"]),
        "test_macro_auc": float(test_result["macro_auc"]),
        "test_macro_ap": float(test_result["macro_ap"]),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "num_tasks": int(num_tasks),
    }]).to_csv(metrics_path, index=False)
    print("[OK] saved:", metrics_path)

    # save task-wise metrics
    per_task_path = Path(args.out_dir) / f"per_task_metrics_toxcast_shared_{args.model}_seed{args.seed}.csv"
    pd.DataFrame({
        "assay": task_names,
        "val_auc": val_result["per_task_auc"],
        "val_ap": val_result["per_task_ap"],
        "test_auc": test_result["per_task_auc"],
        "test_ap": test_result["per_task_ap"],
    }).to_csv(per_task_path, index=False)
    print("[OK] saved:", per_task_path)

    # save task names
    with open(Path(args.out_dir) / "task_names.json", "w") as f:
        json.dump(task_names, f, indent=2)

#%%
if __name__ == "__main__":
    main()