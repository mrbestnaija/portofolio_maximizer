#!/usr/bin/env python3
"""
Train a lightweight local instruction language model on PMX fine-tune JSONL data.

This is an internal fallback trainer so PMX_LLM_FINETUNE_COMMAND can execute
actual training without external SaaS dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on runtime env
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", str(text or ""))


def _load_records(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = (line or "").strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            instruction = str(row.get("instruction") or "").strip()
            output = str(row.get("output") or "").strip()
            if instruction and output:
                out.append({"instruction": instruction, "output": output})
    return out


def _build_vocab(records: list[dict[str, str]], max_vocab: int) -> tuple[dict[str, int], list[str]]:
    c = Counter()
    for row in records:
        text = f"{BOS} Instruction: {row['instruction']} Response: {row['output']} {EOS}"
        c.update(_tokenize(text))
    base = [PAD, UNK, BOS, EOS]
    keep = [tok for tok, _ in c.most_common(max(0, int(max_vocab) - len(base))) if tok not in set(base)]
    vocab = base + keep
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return stoi, vocab


def _encode(text: str, stoi: dict[str, int], max_seq_len: int) -> list[int]:
    toks = _tokenize(text)
    ids = [stoi.get(tok, stoi[UNK]) for tok in toks]
    ids = ids[: max(2, int(max_seq_len))]
    if len(ids) < 2:
        ids = ids + [stoi[EOS]]
    return ids


def _build_dataset(records: list[dict[str, str]], stoi: dict[str, int], max_seq_len: int) -> Any:
    if not TORCH_AVAILABLE or torch is None or TensorDataset is None:
        raise RuntimeError("Torch backend unavailable")
    seqs: list[list[int]] = []
    pad_id = stoi[PAD]
    for row in records:
        text = f"{BOS} Instruction: {row['instruction']} Response: {row['output']} {EOS}"
        ids = _encode(text, stoi, max_seq_len=max_seq_len)
        if len(ids) < max_seq_len:
            ids = ids + [pad_id] * (max_seq_len - len(ids))
        seqs.append(ids[:max_seq_len])
    x = torch.tensor(seqs, dtype=torch.long)  # type: ignore[union-attr]
    return TensorDataset(x)


if TORCH_AVAILABLE and nn is not None:

    class GRULM(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim)
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.proj = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x: Any) -> Any:
            e = self.emb(x)
            h, _ = self.gru(e)
            return self.proj(h)

else:

    class GRULM:  # type: ignore[no-redef]
        pass


@dataclass
class TrainMetrics:
    loss: float
    perplexity: float
    steps: int
    samples: int


def _choose_device(device: str) -> str:
    if not TORCH_AVAILABLE or torch is None:
        return "cpu"
    d = str(device or "auto").strip().lower()
    if d in {"cpu", "cuda", "mps"}:
        return d
    if torch.cuda.is_available():  # type: ignore[union-attr]
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[union-attr,attr-defined]
        return "mps"
    return "cpu"


def _train_torch(
    dataset: Any,
    *,
    vocab_size: int,
    pad_id: int,
    embed_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> tuple[Any, TrainMetrics]:
    if not TORCH_AVAILABLE or torch is None or nn is None or DataLoader is None:
        raise RuntimeError("Torch backend unavailable")
    model = GRULM(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = 0.0
    total_steps = 0
    model.train()
    for _ in range(max(1, int(epochs))):
        for (batch,) in loader:
            batch = batch.to(device)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += float(loss.item())
            total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    ppl = float(math.exp(min(20.0, avg_loss)))
    metrics = TrainMetrics(
        loss=avg_loss,
        perplexity=ppl,
        steps=total_steps,
        samples=len(dataset),
    )
    return model, metrics


def _train_fallback_ngram(records: list[dict[str, str]]) -> tuple[dict[str, Any], TrainMetrics]:
    """Pure-stdlib fallback trainer used when torch is unavailable."""
    sequences: list[list[str]] = []
    transitions = 0
    total_nll = 0.0

    next_counts: dict[str, Counter[str]] = {}
    vocab: set[str] = {PAD, UNK, BOS, EOS}
    for row in records:
        toks = _tokenize(f"{BOS} Instruction: {row['instruction']} Response: {row['output']} {EOS}")
        if len(toks) < 2:
            continue
        sequences.append(toks)
        vocab.update(toks)
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            next_counts.setdefault(a, Counter())[b] += 1

    vocab_size = max(1, len(vocab))
    for toks in sequences:
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i + 1]
            cnt = next_counts.get(a, Counter())
            numer = float(cnt.get(b, 0) + 1)  # Laplace smoothing
            denom = float(sum(cnt.values()) + vocab_size)
            p = max(1e-12, numer / denom)
            total_nll += -math.log(p)
            transitions += 1

    avg_loss = total_nll / max(1, transitions)
    ppl = float(math.exp(min(20.0, avg_loss)))
    model_payload = {
        "type": "ngram_bigram",
        "fallback_backend": "stdlib",
        "vocab_size": vocab_size,
        "next_token_counts": {tok: dict(c.most_common(64)) for tok, c in next_counts.items()},
    }
    metrics = TrainMetrics(
        loss=avg_loss,
        perplexity=ppl,
        steps=transitions,
        samples=len(records),
    )
    return model_payload, metrics


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input JSONL with instruction/output rows.")
    parser.add_argument("--output-dir", default="models/llm_finetune/latest", help="Output directory for checkpoint + metadata.")
    parser.add_argument("--max-vocab", type=int, default=12000, help="Max vocabulary size.")
    parser.add_argument("--max-seq-len", type=int, default=160, help="Max token length per sample.")
    parser.add_argument("--embed-dim", type=int, default=192, help="Embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="GRU hidden dimension.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--min-samples", type=int, default=8, help="Minimum sample count required to train.")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps.")
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="Fail if torch backend is unavailable (disables stdlib fallback backend).",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    records = _load_records(dataset_path)
    if len(records) < max(1, int(args.min_samples)):
        print(
            f"[llm_train] insufficient samples: {len(records)} < {int(args.min_samples)}; skipping training.",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.device)
    backend = "torch_gru" if TORCH_AVAILABLE else "fallback_ngram"

    if not TORCH_AVAILABLE and args.require_torch:
        print(f"[llm_train] torch backend unavailable: {TORCH_IMPORT_ERROR}", file=sys.stderr)
        return 1

    if TORCH_AVAILABLE and torch is not None:
        stoi, vocab = _build_vocab(records, max_vocab=int(args.max_vocab))
        ds = _build_dataset(records, stoi, max_seq_len=int(args.max_seq_len))
        model, metrics = _train_torch(
            ds,
            vocab_size=len(vocab),
            pad_id=stoi[PAD],
            embed_dim=int(args.embed_dim),
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=device,
        )
        ckpt = output_dir / "instruction_lm.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": {
                    "embed_dim": int(args.embed_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "max_seq_len": int(args.max_seq_len),
                },
            },
            ckpt,
        )
    else:
        fallback_model, metrics = _train_fallback_ngram(records)
        ckpt = output_dir / "instruction_lm_fallback.json"
        ckpt.write_text(json.dumps(fallback_model, ensure_ascii=True, indent=2), encoding="utf-8")
        device = "cpu"
        print(f"[llm_train] torch unavailable; used fallback backend ({TORCH_IMPORT_ERROR})")

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "device": device,
        "backend": backend,
        "samples": metrics.samples,
        "steps": metrics.steps,
        "loss": metrics.loss,
        "perplexity": metrics.perplexity,
        "checkpoint": str(ckpt),
        "torch_available": bool(TORCH_AVAILABLE),
        "torch_import_error": TORCH_IMPORT_ERROR if not TORCH_AVAILABLE else "",
    }
    meta_path = output_dir / "training_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[llm_train] checkpoint={ckpt}")
    print(f"[llm_train] summary={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
