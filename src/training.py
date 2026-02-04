# src/training.py

from __future__ import annotations
import json
from pathlib import Path
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer


MODEL_NAME_DEFAULT = "vinai/bertweet-base"

USER_RE = re.compile(r"@\w+")
URL_RE  = re.compile(r"http\S+|www\.\S+")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_tweet(t: str) -> str:
    t = USER_RE.sub("<user>", t)
    t = URL_RE.sub("<url>", t)
    return t


class LabeledTextDataset(Dataset):
    """
    Stores raw/processed text + multi-label targets. Tokenization happens in collate.
    """
    def __init__(self, df: pd.DataFrame, text_col: str, label_cols: List[str]):
        self.texts  = df[text_col].astype(str).tolist()
        self.labels = df[label_cols].astype("float32").values

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor | str]:
        return {
            "text": normalize_tweet(self.texts[i]),
            "labels": torch.from_numpy(self.labels[i]),
        }


@dataclass
class CollateBatch:
    """
    Top-level collator (picklable) so num_workers > 0 works.
    """
    tokenizer: AutoTokenizer
    max_len: int = 128

    def __call__(self, batch):
        texts  = [b["text"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch], dim=0)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc


class BertweetClassifier(nn.Module):
    """
    BERTweet encoder + mean pooling + linear head for multi-label logits.
    """
    def __init__(self, model_name: str, num_labels: int, unfreeze_top_k: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze all parameters first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze top-k transformer layers (partial fine-tuning)
        total_layers = self.encoder.config.num_hidden_layers
        start = max(0, total_layers - unfreeze_top_k)
        for idx in range(start, total_layers):
            for p in self.encoder.encoder.layer[idx].parameters():
                p.requires_grad = True

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pool over valid tokens
        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        x = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        x = self.dropout(x)
        return self.classifier(x)  # logits [B, C]


def compute_pos_weight(df: pd.DataFrame, label_cols: List[str], device: str) -> torch.Tensor:
    """
    pos_weight for BCEWithLogitsLoss: neg_count/pos_count per label.
    """
    Y = df[label_cols].astype("float32").values
    Y_t = torch.tensor(Y, dtype=torch.float32)
    N = Y_t.shape[0]
    pos = Y_t.sum(dim=0)
    neg = N - pos
    pos_weight = (neg / (pos + 1e-8)).to(device).float()
    return pos_weight


def load_processed_dataframe(
    data_path: str | Path,
) -> pd.DataFrame:
    """
    Loads a processed dataset from CSV/Parquet.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)
    if data_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(data_path)

    raise ValueError(f"Unsupported processed data format: {data_path.suffix} (use .csv or .parquet)")


def train_model(
    processed_df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_cols: Optional[List[str]] = None,
    model_name: str = MODEL_NAME_DEFAULT,
    max_len: int = 128,
    batch_size: int = 16,
    val_frac: float = 0.2,
    epochs: int = 25,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    patience: int = 5,
    unfreeze_top_k: int = 3,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[nn.Module, AutoTokenizer, Dict[str, List[float]], List[str]]:
    """
    Trains classifier from processed_df. Returns (model, tokenizer, history, label_cols).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    # Infer label columns if not provided
    if label_cols is None:
        label_cols = processed_df.drop(columns=[text_col]).columns.tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    dataset = LabeledTextDataset(processed_df, text_col=text_col, label_cols=label_cols)

    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    collate = CollateBatch(tokenizer=tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )

    model = BertweetClassifier(
        model_name=model_name,
        num_labels=len(label_cols),
        unfreeze_top_k=unfreeze_top_k,
    ).to(device)

    pos_weight = compute_pos_weight(processed_df, label_cols, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss-based scheduler (correct for val_loss)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)

    best_val = float("inf")
    best_state = None
    bad = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        running = 0.0

        for batch in train_loader:
            yb = batch["labels"].to(device)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }

            opt.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))

        # ---- Validate ----
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                yb = batch["labels"].to(device)
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                logits = model(**inputs)
                vloss += criterion(logits, yb).item()

        val_loss = vloss / max(1, len(val_loader))

        scheduler.step(val_loss)

        cur_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience and best_state is not None:
                model.load_state_dict(best_state)
                break

        print(f"Epoch {epoch+1}/{epochs} - train: {train_loss:.4f} - val: {val_loss:.4f} - lr: {cur_lr:.2e}")

    return model, tokenizer, history, label_cols


# ============================================================
# CLI: training takes input from artifacts/data (CSV/Parquet)
# ============================================================
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Train BERTweet multilabel model from processed dataset.")
    p.add_argument(
        "--data",
        default=os.environ.get("PROCESSED_DATA", ""),
        help="Path to processed dataset (.csv/.parquet). If empty, tries to find one in DATA_DIR.",
    )
    p.add_argument(
        "--data_dir",
        default=os.environ.get("DATA_DIR", "artifacts/data"),
        help="Directory where processed files are stored (artifacts/data).",
    )
    p.add_argument("--text_col", default="cleaned_text")
    p.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--unfreeze_top_k", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _pick_latest_processed_file(data_dir: Path) -> Path:
    """
    Picks the most recently modified processed_*.parquet/csv in artifacts/data.
    """
    candidates = list(data_dir.glob("processed_*.parquet")) + list(data_dir.glob("processed_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No processed_*.parquet/csv found in {data_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    args = _parse_args()

    data_dir = Path(args.data_dir)
    data_path = Path(args.data) if args.data else _pick_latest_processed_file(data_dir)

    df = load_processed_dataframe(data_path)

    model, tokenizer, history, label_cols = train_model(
        processed_df=df,
        text_col=args.text_col,
        label_cols=None,  # infer from df
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        unfreeze_top_k=args.unfreeze_top_k,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # NOTE: We only train here. Exporting weights/tokenizer/config should be done by scripts/export_hf_dir.py
    # In Colab, you would run:
    #   python src/training.py
    #   python scripts/export_hf_dir.py
    #   python scripts/build_joblib_bundle.py

    print("\n✅ Training complete.")
    print("Loaded processed data from:", data_path.resolve())
    print("Num labels:", len(label_cols))


    # --- inside main(), AFTER training finishes ---
    ckpt_dir = Path(os.environ.get("CKPT_DIR", "artifacts/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pt"

    payload = {
        "state_dict": model.state_dict(),
        "label_cols": label_cols,
        "model_name": args.model_name,
        "max_len": args.max_len,
    }

    torch.save(payload, ckpt_path)
    print("✅ Saved checkpoint to:", ckpt_path.resolve())



if __name__ == "__main__":
    main()
