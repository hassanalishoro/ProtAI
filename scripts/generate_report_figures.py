"""Generate the figures for the FYP-1 report from the runs/ directory.

Reads each run's TensorBoard events file and the test split predictions,
and produces three PNG figures saved to a configurable output directory
(default: the report's ThesisFigs folder).

Figures produced:
  fig_training_curves.png    — val/pearson over epochs for all runs in one plot
  fig_loss_curves.png        — train/loss and val/loss over epochs for the headline run
  fig_predictions.png        — predicted-vs-actual scatter for the headline test set

Usage on the cloud pod (after training chain finishes):
    pip install tensorboard matplotlib    # if not already
    python3.11 scripts/generate_report_figures.py \
        --runs-dir /workspace/ProtAI/runs \
        --out-dir /workspace/ProtAI/report_figures

Then SCP the four PNGs back to your laptop and drop them into
the report's ThesisFigs/ folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protai.config import Config, resolve_path
from protai.data import PreShardedDataset
from protai.training.lit_module import ProtAILitModule


# Map each run name (in runs/) to a human-readable label for the legend.
# Names use the _random suffix because all final chain runs train on the
# random-split protocol; the cross-evaluation against the similarity test
# set is done as a separate step after training.
RUN_LABELS: Dict[str, str] = {
    "schnet_aff_random":      "SchNet, frame zero (arch)",
    "gnnmd_aff_random":       "GNN MD, frame zero (arch)",
    "frame_zero_random":      "SchNet, frame zero (frame)",
    "random_frame_random":    "SchNet, random frame",
    "headline_random_s42":    "SchNet headline, seed 42",
    "headline_random_s1337":  "SchNet headline, seed 1337",
}


def _read_scalar_series(events_dir: Path, tag: str) -> List[Tuple[int, float]]:
    """Pull (step, value) pairs for the given scalar tag from a run's TB log."""
    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(str(events_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, float(e.value)) for e in ea.Scalars(tag)]


def plot_training_curves(runs_dir: Path, out_path: Path) -> None:
    """One line per run, val/pearson on y vs epoch on x."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    for run_name, label in RUN_LABELS.items():
        run_path = runs_dir / run_name
        if not run_path.exists():
            print(f"  skip {run_name} (no run directory)")
            continue
        series = _read_scalar_series(run_path, "val/pearson")
        if not series:
            print(f"  skip {run_name} (no val/pearson logged)")
            continue
        # Convert step to approximate epoch by dividing by steps-per-epoch
        # (heuristic; adjust per-run if needed)
        steps = [s for s, _ in series]
        values = [v for _, v in series]
        epochs = list(range(1, len(values) + 1))
        ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.2, label=label)
        plotted_any = True

    if not plotted_any:
        print("No runs with val/pearson logged — skipping training-curves figure")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Pearson correlation")
    ax.set_title("Validation Pearson correlation across training")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_loss_curves(runs_dir: Path, out_path: Path,
                     headline_run: str = "headline_random_s42") -> None:
    """Train and val loss for the headline run on a single panel."""
    import matplotlib.pyplot as plt

    run_path = runs_dir / headline_run
    if not run_path.exists():
        print(f"No headline run found at {run_path} — skipping loss-curves figure")
        return

    train = _read_scalar_series(run_path, "train/loss")
    val = _read_scalar_series(run_path, "val/loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    if train:
        steps = [s for s, _ in train]
        vals = [v for _, v in train]
        ax.plot(steps, vals, label="Training loss", linewidth=1.0, alpha=0.7)
    if val:
        steps = [s for s, _ in val]
        vals = [v for _, v in val]
        ax.plot(steps, vals, label="Validation loss", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean squared error (normalized)")
    ax.set_title(f"Training and validation loss ({headline_run})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_predictions(runs_dir: Path, out_path: Path,
                     headline_run: str = "headline_random_s42") -> None:
    """Predicted-vs-actual scatter for the headline run on the test set."""
    import matplotlib.pyplot as plt

    ckpt_path = runs_dir / headline_run / "best.ckpt"
    if not ckpt_path.exists():
        print(f"No best.ckpt at {ckpt_path} — skipping scatter figure")
        return

    # Load the checkpoint and reconstruct the model
    print(f"  loading {ckpt_path} ...")
    module = ProtAILitModule.load_from_checkpoint(str(ckpt_path), map_location="cuda" if torch.cuda.is_available() else "cpu")
    module.eval()
    cfg = module.cfg

    # Test loader (frame_zero for determinism)
    processed = resolve_path(cfg.data.processed_dir)
    splits = resolve_path(cfg.data.splits_dir)
    test_ds = PreShardedDataset(
        processed_dir=processed,
        split_file=splits / cfg.data.test_split,
        target=cfg.model.target,
        frame_strategy="frame_zero",
        edge_cutoff=cfg.data.edge_cutoff,
        node_feature=("atomic_number" if cfg.model.name == "schnet" else "one_hot_element"),
    )
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    device = next(module.parameters()).device
    ys, yhats = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = module(batch)
            yhat = module._denormalize(pred)
            y = batch.y
            ys.append(y.cpu().numpy().flatten())
            yhats.append(yhat.cpu().numpy().flatten())
    y = np.concatenate(ys)
    yhat = np.concatenate(yhats)

    # Filter outliers for a readable plot (keep |err| < 100 kcal/mol)
    finite = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[finite], yhat[finite]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, yhat, s=8, alpha=0.4, edgecolor="none")
    lo = min(y.min(), yhat.min())
    hi = max(y.max(), yhat.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("True binding energy (kcal/mol)")
    ax.set_ylabel("Predicted binding energy (kcal/mol)")
    ax.set_title(f"Predicted vs actual ({headline_run} on test set)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"),
                   help="Directory containing per-run subdirectories")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "report_figures"),
                   help="Where to write PNG figures")
    p.add_argument("--headline-run", default="headline_random_s42",
                   help="Which run to use for loss curves and predictions scatter")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading runs from {runs_dir}")
    print(f"Writing figures to {out_dir}")

    plot_training_curves(runs_dir, out_dir / "fig_training_curves.png")
    plot_loss_curves(runs_dir, out_dir / "fig_loss_curves.png", args.headline_run)
    plot_predictions(runs_dir, out_dir / "fig_predictions.png", args.headline_run)

    print("\nDone. SCP the PNGs from", out_dir, "into your report's ThesisFigs/ folder.")


if __name__ == "__main__":
    main()
