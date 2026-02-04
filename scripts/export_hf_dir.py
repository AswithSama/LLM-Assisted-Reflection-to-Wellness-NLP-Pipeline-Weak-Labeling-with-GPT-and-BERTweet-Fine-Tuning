# scripts/export_hf_dir.py
# Trains/loads your checkpoint, exports HF-style folder, AND builds a portable .joblib bundle.

from __future__ import annotations

import os
import json
import joblib
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoTokenizer

from src.training import BertweetClassifier  # uses your same model class


# ---- Files to ignore (large weights, temporary files, etc.) ----
EXCLUDE_NAMES = {"pytorch_model.bin", "pytorch_model.safetensors", ".DS_Store"}
EXCLUDE_SUFFIXES = {".pt", ".ckpt"}


def read_all_tokenizer_files(dirpath: Path) -> Dict[str, bytes]:
    """
    Collects tokenizer/config files (bpe.codes, vocab.txt, config.json, etc.).
    Skips heavy model weights and irrelevant files.
    """
    blobs: Dict[str, bytes] = {}
    for p in dirpath.iterdir():
        if p.is_dir():
            continue
        name = p.name
        if name in EXCLUDE_NAMES or p.suffix in EXCLUDE_SUFFIXES:
            continue
        if any(name.endswith(ext) for ext in (".json", ".txt", ".model", ".codes")):
            blobs[name] = p.read_bytes()
    return blobs


def pack_model(export_dir: Path, model_bin: str = "pytorch_model.bin") -> Dict:
    """
    Creates a portable joblib bundle containing:
      - Model config (label info, thresholds, etc.)
      - Tokenizer files
      - State dict
    """
    cfg = json.loads((export_dir / "config.json").read_text())
    bundle = {
        "cfg": {
            "model_name": cfg.get("model_name", "vinai/bertweet-base"),
            "num_labels": int(cfg["num_labels"]),
            "label_cols": list(cfg["label_cols"]),
            "thresholds": cfg.get("thresholds", [0.5] * int(cfg["num_labels"])),
            "max_len": int(cfg.get("max_len", 128)),
        },
        "tokenizer_files": read_all_tokenizer_files(export_dir),
        "state_dict": {k: v.cpu() for k, v in torch.load(export_dir / model_bin, map_location="cpu").items()},
    }
    return bundle


def export_hf_dir_and_joblib(
    ckpt_path: Path,
    export_dir: Path,
    joblib_out: Path,
    thresholds: list[float] | None = None,
) -> None:
    """
    1) Load your training checkpoint (.pt) which includes state_dict + label_cols + model_name + max_len
    2) Write HF-style export folder:
       - pytorch_model.bin
       - tokenizer files
       - config.json (metadata)
    3) Pack into reflection_model.joblib using your technique.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name: str = ckpt["model_name"]
    label_cols = list(ckpt["label_cols"])
    max_len: int = int(ckpt.get("max_len", 128))

    # thresholds default
    if thresholds is None:
        thresholds = [0.5] * len(label_cols)

    export_dir.mkdir(parents=True, exist_ok=True)
    joblib_out.parent.mkdir(parents=True, exist_ok=True)

    # Recreate model and load weights
    model = BertweetClassifier(model_name=model_name, num_labels=len(label_cols), unfreeze_top_k=0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1) weights
    torch.save(model.state_dict(), export_dir / "pytorch_model.bin")

    # 2) tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.save_pretrained(export_dir)

    # 3) metadata for inference
    meta = {
        "model_name": model_name,
        "num_labels": len(label_cols),
        "label_cols": label_cols,
        "thresholds": thresholds,
        "max_len": max_len,
    }
    (export_dir / "config.json").write_text(json.dumps(meta, indent=2))

    # 4) build joblib bundle (your technique)
    bundle = pack_model(export_dir, model_bin="pytorch_model.bin")
    joblib.dump(bundle, joblib_out, compress=3)

    print("✅ Exported HF dir to:", export_dir.resolve())
    print("✅ Saved joblib to:", joblib_out.resolve())
    print("Contains bpe.codes:", "bpe.codes" in bundle["tokenizer_files"])


def main():
    # In Colab you can set these via:
    # %env CKPT_PATH=artifacts/checkpoints/best_model.pt
    # %env EXPORT_DIR=artifacts/bertweet_export
    # %env JOBLIB_OUT=artifacts/job_files/reflection_model.joblib
    ckpt_path = Path(os.environ.get("CKPT_PATH", "artifacts/checkpoints/best_model.pt"))
    export_dir = Path(os.environ.get("EXPORT_DIR", "artifacts/bertweet_export"))
    joblib_out = Path(os.environ.get("JOBLIB_OUT", "artifacts/job_files/reflection_model.joblib"))

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Make sure training saved artifacts/checkpoints/best_model.pt (or set CKPT_PATH)."
        )

    export_hf_dir_and_joblib(ckpt_path=ckpt_path, export_dir=export_dir, joblib_out=joblib_out)


if __name__ == "__main__":
    main()
