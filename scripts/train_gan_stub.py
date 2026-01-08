#!/usr/bin/env python3
"""
Lightweight GAN trainer stub that respects PIPELINE_DEVICE and synthetic data.

If PyTorch is available, runs a tiny GAN loop over return sequences; otherwise
logs a warning and exits. Intended as a starting point for TimeGAN/diffusion
integration without pulling heavy dependencies by default.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def _detect_device(prefer_gpu: bool = True) -> str:
    if not prefer_gpu:
        return "cpu"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def _load_dataset(dataset_path: str, tickers: Optional[str]) -> pd.DataFrame:
    base = Path(dataset_path)
    if base.name == "latest.json":
        payload = {}
        try:
            payload = json.loads(base.read_text())
        except Exception as exc:
            logger.error("Failed to read latest pointer %s: %s", base, exc)
        base = Path(payload.get("dataset_path") or base.parent)

    if not base.exists():
        raise FileNotFoundError(f"Dataset path {base} not found")

    frames = []
    if base.is_dir():
        for pq in base.glob("*.parquet"):
            frames.append(pd.read_parquet(pq))
    elif base.suffix == ".parquet":
        frames.append(pd.read_parquet(base))

    if not frames:
        raise RuntimeError(f"No parquet files found under {base}")

    df = pd.concat(frames).sort_index()
    if tickers:
        subset = [t.strip() for t in tickers.split(",") if t.strip()]
        if "ticker" in df.columns and subset:
            df = df[df["ticker"].isin(subset)]
            logger.info("Subset to tickers: %s", subset)
    return df


def _build_sequences(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    if "Close" not in df.columns:
        raise ValueError("Dataset must include Close column")
    rets = df["Close"].astype(float).pct_change().fillna(0.0).values
    sequences = []
    for i in range(len(rets) - seq_len):
        sequences.append(rets[i : i + seq_len])
    return np.array(sequences, dtype=np.float32)


def _train_gan(sequences: np.ndarray, device: str, epochs: int, batch_size: int, latent_dim: int, hidden_dim: int, out_dir: Path) -> Tuple[float, float]:
    import torch
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self, latent_dim: int, hidden_dim: int, seq_len: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, seq_len),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self, seq_len: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(seq_len, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    device_t = torch.device(device)
    seq_len = sequences.shape[1]
    dataset = torch.utils.data.TensorDataset(torch.tensor(sequences))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G = Generator(latent_dim, hidden_dim, seq_len).to(device_t)
    D = Discriminator(seq_len, hidden_dim).to(device_t)
    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-3)
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for real_batch, in loader:
            real = real_batch.to(device_t)
            batch = real.size(0)
            # Train discriminator
            noise = torch.randn(batch, latent_dim, device=device_t)
            fake = G(noise).detach()
            d_opt.zero_grad()
            d_real = D(real)
            d_fake = D(fake)
            loss_d = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            d_opt.step()

            # Train generator
            noise = torch.randn(batch, latent_dim, device=device_t)
            gen = G(noise)
            g_opt.zero_grad()
            loss_g = criterion(D(gen), torch.ones_like(d_fake))
            loss_g.backward()
            g_opt.step()

        logger.info("Epoch %d/%d | loss_d=%.4f loss_g=%.4f", epoch + 1, epochs, loss_d.item(), loss_g.item())

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), out_dir / "generator.pt")
    torch.save(D.state_dict(), out_dir / "discriminator.pt")
    return float(loss_d.item()), float(loss_g.item())


@click.command()
@click.option("--dataset-path", default="data/synthetic/latest.json", show_default=True, help="Path to latest.json or dataset directory.")
@click.option("--prefer-gpu/--no-prefer-gpu", default=True, show_default=True, help="Prefer GPU when available; falls back to CPU.")
@click.option("--tickers", default=None, help="Optional comma-separated tickers to subset.")
@click.option("--seq-len", default=32, show_default=True, help="Sequence length for GAN training.")
@click.option("--latent-dim", default=16, show_default=True, help="Latent dimension for generator input.")
@click.option("--hidden-dim", default=64, show_default=True, help="Hidden dimension for generator/discriminator.")
@click.option("--batch-size", default=64, show_default=True, help="Training batch size.")
@click.option("--train-epochs", default=5, show_default=True, help="Training epochs (set 0 to skip training).")
@click.option("--output-dir", default="models/synthetic/gan_stub", show_default=True, help="Directory for checkpoints and metadata.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(dataset_path: str, prefer_gpu: bool, tickers: Optional[str], seq_len: int, latent_dim: int, hidden_dim: int,
         batch_size: int, train_epochs: int, output_dir: str, verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = _detect_device(prefer_gpu=prefer_gpu)
    os.environ["PIPELINE_DEVICE"] = device
    logger.info("GAN trainer device: %s (prefer_gpu=%s)", device, prefer_gpu)

    try:
        df = _load_dataset(dataset_path, tickers)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        return

    logger.info("Loaded dataset %s rows=%d columns=%d", dataset_path, len(df), len(df.columns))

    if train_epochs <= 0:
        logger.info("Training skipped (train_epochs<=0).")
        return

    try:
        sequences = _build_sequences(df, seq_len=seq_len)
    except Exception as exc:
        logger.error("Failed to build sequences: %s", exc)
        return

    if sequences.shape[0] < batch_size:
        logger.error("Not enough sequences for training: have %d, need >= batch_size (%d)", sequences.shape[0], batch_size)
        return

    try:
        loss_d, loss_g = _train_gan(
            sequences=sequences,
            device=device,
            epochs=train_epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dir=Path(output_dir),
        )
        meta = {
            "device": device,
            "epochs": train_epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "seq_len": seq_len,
            "loss_d": loss_d,
            "loss_g": loss_g,
            "dataset_path": dataset_path,
            "tickers": tickers,
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Training complete. Checkpoints written to %s", output_dir)
    except ImportError:
        logger.warning("PyTorch not installed; GAN training skipped. Install torch to enable training.")
    except Exception as exc:
        logger.error("GAN training failed: %s", exc)


if __name__ == "__main__":
    main()
