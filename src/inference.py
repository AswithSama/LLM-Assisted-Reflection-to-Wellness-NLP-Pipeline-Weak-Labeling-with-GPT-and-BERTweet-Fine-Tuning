# src/inference.py
# Loads the packaged reflection_model.joblib (cfg + tokenizer files + state_dict)
# and runs multilabel inference.

from __future__ import annotations

import io
import os
import re
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import joblib
import torch
from transformers import AutoTokenizer

from src.training import BertweetClassifier  # reuse the exact same architecture


USER_RE = re.compile(r"@\w+")
URL_RE  = re.compile(r"http\S+|www\.\S+")


def get_device(prefer: Optional[str] = None) -> str:
    if prefer is not None:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_tweet(t: str) -> str:
    t = USER_RE.sub("<user>", t)
    t = URL_RE.sub("<url>", t)
    return t


def _load_tokenizer_from_joblib(tokenizer_files: Dict[str, bytes]) -> AutoTokenizer:
    """
    Reconstruct HF tokenizer from bytes by writing them into a temporary directory
    and calling AutoTokenizer.from_pretrained on that directory.
    """
    tmpdir = tempfile.TemporaryDirectory()
    d = Path(tmpdir.name)

    for name, blob in tokenizer_files.items():
        (d / name).write_bytes(blob)

    # prefer local-only to avoid accidental network calls in some environments
    tokenizer = AutoTokenizer.from_pretrained(str(d), use_fast=False, local_files_only=True)

    # attach so temp dir isn't GC'd too early
    tokenizer._tmpdir = tmpdir  # type: ignore[attr-defined]
    return tokenizer


def load_joblib_bundle(joblib_path: str | Path) -> Tuple[torch.nn.Module, AutoTokenizer, Dict[str, Any]]:
    """
    Returns: (model, tokenizer, cfg)
    cfg includes: model_name, num_labels, label_cols, thresholds, max_len
    """
    joblib_path = Path(joblib_path)
    if not joblib_path.exists():
        raise FileNotFoundError(f"Joblib bundle not found: {joblib_path}")

    bundle = joblib.load(joblib_path)

    cfg = bundle["cfg"]
    state_dict = bundle["state_dict"]
    tok_files = bundle["tokenizer_files"]

    model = BertweetClassifier(
        model_name=cfg["model_name"],
        num_labels=int(cfg["num_labels"]),
        unfreeze_top_k=0,  # inference
    )
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = _load_tokenizer_from_joblib(tok_files)

    return model, tokenizer, cfg


def predict(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    label_cols: List[str],
    max_len: int = 128,
    threshold: float = 0.5,
    device: Optional[str] = None,
    return_probabilities: bool = True,
    only_above_threshold: bool = True,
) -> List[Dict[str, Any]]:
    device = get_device(device)

    model.eval()
    model.to(device)

    norm_texts = [normalize_tweet(str(t)) for t in texts]

    with torch.no_grad():
        enc = tokenizer(
            norm_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits)

    results: List[Dict[str, Any]] = []
    for original_text, row_probs in zip(texts, probs):
        row_probs_list = row_probs.detach().cpu().tolist()

        if only_above_threshold:
            pred_labels = [label_cols[i] for i, p in enumerate(row_probs_list) if p > threshold]
        else:
            pred_labels = [label_cols[i] for i in range(len(label_cols))]

        out: Dict[str, Any] = {"text": original_text, "predicted_labels": pred_labels}

        if return_probabilities:
            out["probabilities"] = {label_cols[i]: float(row_probs_list[i]) for i in range(len(label_cols))}

        results.append(out)

    return results


def pretty_print_predictions(
    predictions: List[Dict[str, Any]],
    threshold: float = 0.5,
    show_only_above_threshold: bool = True,
) -> None:
    for pred in predictions:
        print(f"Text: {pred['text']}")
        print(f"Predicted labels: {pred['predicted_labels']}")

        probs = pred.get("probabilities", {})
        if probs:
            print("Probabilities:")
            for lbl, score in probs.items():
                if (not show_only_above_threshold) or (score > threshold):
                    print(f"  {lbl}: {score:.3f}")
        print()


# ============================================================
# CLI: run inference from a .joblib bundle
# ============================================================
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Run inference from reflection_model.joblib")
    p.add_argument(
        "--bundle",
        default=os.environ.get("MODEL_BUNDLE", "artifacts/job_files/reflection_model.joblib"),
        help="Path to reflection_model.joblib",
    )
    p.add_argument("--text", default=None, help="Single text input")
    p.add_argument("--text_file", default=None, help="Optional: file with one text per line")
    p.add_argument("--threshold", type=float, default=None, help="Override threshold (default uses cfg.thresholds[0] or 0.5)")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default auto)")
    return p.parse_args()


def main():
    args = _parse_args()

    model, tokenizer, cfg = load_joblib_bundle(args.bundle)

    label_cols = list(cfg["label_cols"])
    max_len = int(cfg.get("max_len", 128))

    # threshold logic: if user passes threshold, use it.
    # else if cfg has thresholds list, use 0.5-style first value.
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        ths = cfg.get("thresholds", None)
        threshold = float(ths[0]) if isinstance(ths, list) and len(ths) else 0.5

    # collect texts
    texts: List[str] = []
    if args.text is not None:
        texts = [args.text]
    elif args.text_file is not None:
        lines = Path(args.text_file).read_text(encoding="utf-8").splitlines()
        texts = [ln for ln in lines if ln.strip()]
    else:
        raise ValueError("Provide --text or --text_file")

    preds = predict(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        label_cols=label_cols,
        max_len=max_len,
        threshold=threshold,
        device=args.device,
        return_probabilities=True,
        only_above_threshold=True,
    )

    pretty_print_predictions(preds, threshold=threshold, show_only_above_threshold=True)


if __name__ == "__main__":
    main()
