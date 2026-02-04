# app.py
# app.py
from __future__ import annotations

import os, tempfile, joblib, re
from pathlib import Path
from typing import Dict, Any, Iterable, Set, List

from flask import Flask, request, jsonify, render_template_string

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

import spacy
import emoji
from spacy.lang.en.stop_words import STOP_WORDS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# INLINE PREPROCESSING (copied from your preprocessing.py)
# -------------------------
EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002700-\U000027BF"  # dingbats
    u"\U0001F900-\U0001F9FF"  # supplemental symbols
    u"\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE
)

def build_nlp(model: str = "en_core_web_sm"):
    return spacy.load(model, disable=["ner", "parser", "textcat"])

def build_stopwords() -> Set[str]:
    negations = {"no", "not", "nor", "n't"}
    context_words = {
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
        "this", "that", "these", "those",
        "who", "whom", "whose", "which", "what",
        "own",
        "here", "there", "where", "when", "why", "how",
    }
    return set(STOP_WORDS) - negations - context_words

def emoji_to_text(e: str) -> str:
    return emoji.demojize(e)

def is_emoji(token: str) -> bool:
    return bool(EMOJI_PATTERN.fullmatch(token))

def in_bertweet(token: str, tokenizer: AutoTokenizer) -> bool:
    return tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id

def preprocess_doc(doc, tokenizer: AutoTokenizer, stop_words: Set[str]) -> List[str]:
    tokens: List[str] = []
    last_punct = None
    last_emoji = None

    for tok in doc:
        t = tok.text
        lower = t.lower()

        # EMOJIS (keep one per run)
        if is_emoji(t):
            alias = emoji_to_text(t)
            if alias != last_emoji and in_bertweet(alias, tokenizer):
                tokens.append(alias)
                last_emoji = alias
            last_punct = None
            continue

        # PUNCTUATION (keep ! ? ...)
        if tok.is_punct:
            if t in {"!", "?", "..."}:
                if t != last_punct:
                    tokens.append(t)
                    last_punct = t
            else:
                last_punct = None
            last_emoji = None
            continue

        # NEGATION handling: didn't -> not
        if lower in {"n't", "'nt"}:
            tokens.append("not")
            last_punct = None
            last_emoji = None
            continue

        # NUMBERS (keep)
        if tok.like_num:
            tokens.append(lower)
            last_punct = None
            last_emoji = None
            continue

        # WORDS
        if tok.is_alpha:
            lemma = tok.lemma_.lower()
            if lemma == "-pron-":  # spaCy quirk
                lemma = lower

            if lemma not in stop_words and in_bertweet(lemma, tokenizer):
                tokens.append(lemma)

            last_punct = None
            last_emoji = None
            continue

        last_punct = None
        last_emoji = None

    return tokens

def preprocess_texts_inline(
    texts: Iterable[str],
    nlp,
    tokenizer: AutoTokenizer,
    stop_words: Set[str],
    batch_size: int = 512,
) -> List[str]:
    outputs: List[str] = []
    for doc in nlp.pipe((str(t) for t in texts), batch_size=batch_size):
        toks = preprocess_doc(doc, tokenizer=tokenizer, stop_words=stop_words)
        outputs.append(" ".join(toks))
    return outputs


# -------------------------
# MODEL
# -------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1)
        x = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.classifier(self.dropout(x))


# -------------------------
# BUNDLE LOADER
# -------------------------
def load_bundle_joblib(path: Path):
    """
    Expects a dict with:
      - "cfg": {model_name, num_labels, label_cols, thresholds, max_len}
      - "state_dict": torch state dict (CPU tensors)
      - "tokenizer_files": {filename: bytes}
    Rebuilds tokenizer from a temp dir and loads a model instance.
    """
    b = joblib.load(path)
    cfg = b["cfg"]

    tmpdir = tempfile.TemporaryDirectory()
    for name, blob in b["tokenizer_files"].items():
        (Path(tmpdir.name) / name).write_bytes(blob)

    tok = AutoTokenizer.from_pretrained(tmpdir.name)

    model = MeanPoolClassifier(cfg["model_name"], int(cfg["num_labels"]))
    model.load_state_dict(b["state_dict"], strict=True)
    model.to(DEVICE).eval()

    thr = torch.tensor(cfg.get("thresholds", [0.5] * int(cfg["num_labels"])), device=DEVICE)

    return {
        "tmp": tmpdir,  # keep referenced so files persist
        "tok": tok,
        "model": model,
        "labels": list(cfg["label_cols"]),
        "thr": thr,
        "max_len": int(cfg.get("max_len", 128)),
    }


@torch.no_grad()
def predict_probs(texts: List[str], bundle) -> List[Dict[str, Any]]:
    tok, mdl, labels, thr, max_len = (
        bundle["tok"], bundle["model"], bundle["labels"], bundle["thr"], bundle["max_len"]
    )

    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    logits = mdl(enc["input_ids"], enc["attention_mask"])
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).int()

    out: List[Dict[str, Any]] = []
    for i, txt in enumerate(texts):
        p = probs[i]
        topk = torch.topk(p, k=min(5, len(labels))).indices.tolist()
        out.append({
            "text": txt,
            "probs": {labels[j]: float(p[j]) for j in range(len(labels))},
            "labels_on": [labels[j] for j in range(len(labels)) if preds[i, j].item() == 1],
            "top5": [(labels[j], float(p[j])) for j in topk],
        })
    return out


# -------------------------
# LOAD BUNDLE + PREPROC ARTIFACTS (ONCE)
# -------------------------
REF_PATH = Path(os.environ.get(
    "REF_JOBLIB",
    "/Users/aswithsama/Desktop/github_uplifty/bundles1/job_files/reflection_model.joblib"
))
REF = load_bundle_joblib(REF_PATH)

# Preprocessing objects loaded once (fast in requests)
NLP = build_nlp("en_core_web_sm")
STOP_WORDS_SET = build_stopwords()

# Use the SAME tokenizer object that came with the bundle for vocab checks.
# This keeps in_bertweet(...) consistent and avoids loading another tokenizer.
PREPROC_TOKENIZER = REF["tok"]


# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

FORM_HTML = """
<!doctype html>
<title>Reflection Label Probabilities</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;max-width:900px}
  textarea{width:100%;height:8rem}
  .row{margin-bottom:1rem}
  button{padding:.6rem 1rem;border-radius:8px;border:1px solid #ccc;cursor:pointer}
  .blk{margin:1rem 0;padding:.75rem 1rem;border:1px solid #eee;border-radius:8px;background:#fafafa}
  code{white-space:pre-wrap}
</style>
<h2>Reflection Label Probabilities</h2>
<form method="POST" action="/score">
  <div class="row"><textarea name="text" placeholder="Type or paste your reflection..."></textarea></div>
  <button type="submit">Get Probabilities</button>
</form>

{% if simple %}
  <div class="blk"><b>Predicted Probabilities</b><br><code>{{ simple.reflection }}</code></div>
  <div class="blk"><b>Cleaned Text (debug)</b><br><code>{{ simple.cleaned }}</code></div>
{% elif result %}
  <h3>Full JSON (debug)</h3>
  <pre>{{ result | tojson(indent=2) }}</pre>
{% endif %}

<p style="margin-top:1rem">JSON API:
<code>POST /api/score?compact=1</code> with <code>{"text":"..."}</code></p>
"""

@app.get("/")
def index():
    return render_template_string(FORM_HTML, result=None)


@app.post("/score")
def score_form():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template_string(FORM_HTML, result=None, simple=None), 400

    full = score_text(text)
    simple = make_simple_lines(full)
    return render_template_string(FORM_HTML, simple=simple, result=None)


@app.post("/api/score")
def api_score():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    full = score_text(text)
    if request.args.get("compact"):
        return jsonify(make_simple_lines(full))
    return jsonify(full)


# -------------------------
# SCORE WITH PREPROCESSING
# -------------------------
@torch.no_grad()
def score_text(text: str) -> Dict[str, Any]:
    """
    1) preprocess raw text using your custom pipeline
    2) run model on cleaned text
    """
    cleaned = preprocess_texts_inline(
        [text],
        nlp=NLP,
        tokenizer=PREPROC_TOKENIZER,
        stop_words=STOP_WORDS_SET,
        batch_size=64,
    )[0]

    result = predict_probs([cleaned], REF)[0]
    return {
        "text": text,
        "cleaned_text": cleaned,
        "probs": result["probs"],
        "labels_on": result["labels_on"],
        "top5": result["top5"],
    }


def make_simple_lines(payload: Dict[str, Any]) -> Dict[str, str]:
    probs_sorted = sorted(payload["probs"].items(), key=lambda kv: kv[1], reverse=True)
    probs_str = ", ".join(f"{k}: {v:.3f}" for k, v in probs_sorted)
    return {"reflection": probs_str, "cleaned": payload.get("cleaned_text", "")}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

