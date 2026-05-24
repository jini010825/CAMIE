#%%
import os
import ast
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig, BertModel
#%%
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--text_col", type=str, default="embedding_text")
    p.add_argument("--out_dir", type=str, default="assets/assay_emb/biobert_emb")
    p.add_argument("--prefix", type=str, default=None)
    p.add_argument("--model_name", type=str, default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    return p.parse_args()
#%%
def ensure_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()
#%%
def mean_pooling(last_hidden_state, attention_mask):
    """
    last_hidden_state: [B, T, H]
    attention_mask: [B, T]
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
#%%
@torch.no_grad()
def encode_texts(
    texts,
    tokenizer,
    model,
    device="cuda",
    batch_size=16,
    max_length=128,
):
    cls_embeddings = []
    mean_embeddings = []

    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state  # [B, T, H]

        cls_emb = last_hidden[:, 0, :]  # [CLS]
        mean_emb = mean_pooling(last_hidden, encoded["attention_mask"])

        cls_embeddings.append(cls_emb.cpu().numpy())
        mean_embeddings.append(mean_emb.cpu().numpy())

    cls_embeddings = np.concatenate(cls_embeddings, axis=0)
    mean_embeddings = np.concatenate(mean_embeddings, axis=0)

    return cls_embeddings, mean_embeddings
#%%
def save_outputs(df, cls_emb, mean_emb, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    cls_path = os.path.join(out_dir, f"{prefix}_cls.npy")
    mean_path = os.path.join(out_dir, f"{prefix}_mean.npy")
    meta_path = os.path.join(out_dir, f"{prefix}_meta.csv")

    np.save(cls_path, cls_emb)
    np.save(mean_path, mean_emb)

    meta_cols = [c for c in df.columns if c in [
        "mode", "assay", "domain", "tier", "moa", "target_gene",
        "hierarchy_path", "embedding_text", "semantic_summary", "context_role"
    ]]
    df[meta_cols].to_csv(meta_path, index=False)

    print(f"Saved CLS embeddings:  {cls_path}")
    print(f"Saved Mean embeddings: {mean_path}")
    print(f"Saved metadata:        {meta_path}")
#%%
def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.input_csv}")

    texts = df[args.text_col].fillna("").astype(str).tolist()

    if args.prefix is None:
        prefix = os.path.splitext(os.path.basename(args.input_csv))[0]
    else:
        prefix = args.prefix

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
    config = BertConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1", config=config)
    model.to(device)

    cls_emb, mean_emb = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("CLS shape :", cls_emb.shape)
    print("Mean shape:", mean_emb.shape)

    save_outputs(df, cls_emb, mean_emb, args.out_dir, prefix)
#%%
if __name__ == "__main__":
    main()