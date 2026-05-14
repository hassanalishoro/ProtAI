"""Publication-quality figures for the ProtAI log K experiments.

Reads each run's actual outputs:
  * TensorBoard event files for per-epoch curves
  * test_metrics_similarity.json for cross-evaluation numbers
  * best.ckpt for an inference pass that produces the predicted-vs-actual scatter

No synthetic data, no extrapolation. Every number in every figure traces
back to a checkpoint or an event file in runs/.

Outputs (saved as both PNG @ 300 dpi and PDF for vector quality):

  fig_training_curves.png/pdf   validation Pearson over epochs, one line per run
  fig_loss_curves.png/pdf       train + val loss trajectory for the headline run
  fig_cross_eval.png/pdf        in-distribution vs cross-eval Pearson bar chart
  fig_predictions.png/pdf       predicted vs actual scatter on the headline test set
  fig_metric_panel.png/pdf      four-panel metric summary (Pearson, Spearman, RMSE, MAE)

Default headline run is `multitask_logk_energy` because that's the
configuration with the strongest combined in-dist + cross-eval result.
Override with --headline-run to point at a different checkpoint.

Usage:
    py -3.11 scripts/generate_report_figures.py
    py -3.11 scripts/generate_report_figures.py --runs-dir /workspace/ProtAI/runs
    py -3.11 scripts/generate_report_figures.py --out-dir docs/figures

Dependencies:
    pip install tensorboard matplotlib
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    name: str          # subdirectory under runs/
    label: str         # legend / axis label
    color: str         # consistent color across all figures
    marker: str = "o"


# Order matters — controls plot/legend ordering and bar chart positions.
# Colors picked from matplotlib's tableau-colorblind10 palette so the
# figures stay readable in greyscale prints and for color-blind readers.
RUNS: List[RunSpec] = [
    RunSpec(
        name="frame_zero_logk",
        label="SchNet, frame zero (static)",
        color="#0072B2",   # blue
        marker="s",
    ),
    RunSpec(
        name="headline_logk",
        label="SchNet, random frame (trajectory)",
        color="#D55E00",   # vermillion
        marker="o",
    ),
    RunSpec(
        name="multitask_logk_energy",
        label="SchNet, multitask (log K + energy aux)",
        color="#009E73",   # teal/green
        marker="^",
    ),
]

DEFAULT_HEADLINE_RUN = "multitask_logk_energy"

# Target axis label and unit, applied across figures.
TARGET_LABEL = r"Experimental affinity ($-\log_{10} K$)"
TARGET_UNIT = r"$-\log_{10} K$"


# ---------------------------------------------------------------------------
# Matplotlib styling — applied once at import time
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,

        "lines.linewidth": 1.6,
        "lines.markersize": 4,
    })


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_scalar(events_dir: Path, tag: str) -> List[Tuple[int, float]]:
    """Pull (step, value) pairs for the given scalar tag from a run's TB log.

    Multiple event files are aggregated (Lightning emits a new file each
    time training is resumed). Returns an empty list if the tag is absent.
    """
    from tensorboard.backend.event_processing import event_accumulator

    series: List[Tuple[int, float]] = []
    # event_accumulator handles multiple event files in the directory.
    ea = event_accumulator.EventAccumulator(str(events_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    for e in ea.Scalars(tag):
        series.append((int(e.step), float(e.value)))
    series.sort(key=lambda t: t[0])
    return series


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [warn] could not parse {path.name}: {e}")
        return None


def _resolve_runs(runs_dir: Path) -> List[RunSpec]:
    """Filter the master RUNS list down to those that actually exist on disk."""
    out: List[RunSpec] = []
    for spec in RUNS:
        if (runs_dir / spec.name).exists():
            out.append(spec)
        else:
            print(f"  [skip] no run dir at {runs_dir / spec.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 1 — training curves
# ---------------------------------------------------------------------------

def fig_training_curves(runs_dir: Path, runs: List[RunSpec], out_base: Path) -> None:
    """Validation Pearson over training epochs, one curve per run."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plotted = 0
    for spec in runs:
        series = _read_scalar(runs_dir / spec.name, "val/pearson")
        if not series:
            print(f"  [skip] no val/pearson logged for {spec.name}")
            continue
        # The TensorBoard step axis is the global training step counter.
        # We rebase to "validation epoch" (one validation pass per epoch).
        epochs = np.arange(1, len(series) + 1)
        values = np.array([v for _, v in series], dtype=float)
        ax.plot(epochs, values, color=spec.color, marker=spec.marker,
                label=spec.label, linewidth=1.5, markersize=4,
                markevery=max(1, len(epochs) // 30))
        plotted += 1

    if plotted == 0:
        print("  [skip] training curves — no data")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation Pearson $\rho$")
    ax.set_title("Validation Pearson during training")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_both(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — loss curves for the headline run
# ---------------------------------------------------------------------------

def fig_loss_curves(runs_dir: Path, headline_run: str, out_base: Path) -> None:
    """Train and validation loss for the headline run on a single panel."""
    import matplotlib.pyplot as plt

    run_path = runs_dir / headline_run
    if not run_path.exists():
        print(f"  [skip] no headline run at {run_path}")
        return

    train = _read_scalar(run_path, "train/loss")
    val = _read_scalar(run_path, "val/loss")
    if not train and not val:
        print(f"  [skip] no loss tags logged for {headline_run}")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if train:
        # Smooth training loss with a moving average — Lightning logs every
        # 10 steps and the per-step values are noisy.
        steps = np.array([s for s, _ in train], dtype=float)
        vals = np.array([v for _, v in train], dtype=float)
        if len(vals) > 25:
            window = max(5, len(vals) // 50)
            kernel = np.ones(window) / window
            smooth = np.convolve(vals, kernel, mode="valid")
            steps_smooth = steps[window - 1:]
            ax.plot(steps_smooth, smooth, color="#888", linewidth=1.0,
                    alpha=0.85, label="Training loss (smoothed)")
            ax.plot(steps, vals, color="#888", linewidth=0.4, alpha=0.25)
        else:
            ax.plot(steps, vals, color="#888", linewidth=1.0,
                    alpha=0.85, label="Training loss")
    if val:
        steps = np.array([s for s, _ in val], dtype=float)
        vals = np.array([v for _, v in val], dtype=float)
        ax.plot(steps, vals, color="#D55E00", linewidth=1.8,
                label="Validation loss", marker="o", markersize=4,
                markevery=max(1, len(steps) // 30))

    # Predict-the-mean baseline lives at MSE = 1.0 in z-score space.
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.99, 1.02, "Predict-the-mean baseline",
            ha="right", va="bottom", fontsize=8, color="black", alpha=0.7)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean squared error (normalized)")
    ax.set_title(f"Loss trajectory: {headline_run}")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_both(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — cross-evaluation bar chart
# ---------------------------------------------------------------------------

def fig_cross_eval(runs_dir: Path, runs: List[RunSpec], out_base: Path) -> None:
    """Grouped bar chart: in-distribution Pearson vs similarity-test Pearson."""
    import matplotlib.pyplot as plt

    rows = []
    for spec in runs:
        # In-distribution Pearson — read the run's final test metric from
        # the TensorBoard log under tag "test/pearson".
        in_series = _read_scalar(runs_dir / spec.name, "test/pearson")
        in_pearson = in_series[-1][1] if in_series else None

        # Cross-eval Pearson — from the JSON dump produced by eval.py.
        cross_json = _read_json(
            runs_dir / spec.name / "test_metrics_similarity.json"
        )
        cross_pearson = cross_json["pearson"] if cross_json else None

        rows.append((spec, in_pearson, cross_pearson))

    if all(in_p is None and cross_p is None for _, in_p, cross_p in rows):
        print("  [skip] cross-eval — no data")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    n = len(rows)
    bar_width = 0.36
    x = np.arange(n)

    in_vals = [in_p if in_p is not None else 0.0 for _, in_p, _ in rows]
    cross_vals = [cross_p if cross_p is not None else 0.0 for _, _, cross_p in rows]
    in_missing = [in_p is None for _, in_p, _ in rows]
    cross_missing = [cross_p is None for _, _, cross_p in rows]

    bars_in = ax.bar(
        x - bar_width / 2, in_vals, bar_width,
        color=[spec.color for spec, _, _ in rows],
        edgecolor="black", linewidth=0.6,
        label="In-distribution test (random_logk)",
    )
    bars_cross = ax.bar(
        x + bar_width / 2, cross_vals, bar_width,
        color=[spec.color for spec, _, _ in rows],
        edgecolor="black", linewidth=0.6, hatch="///",
        alpha=0.55,
        label="Cross-evaluation (similarity test, no retraining)",
    )

    # Annotate each bar with the numeric value.
    for bar, val, missing in list(zip(bars_in, in_vals, in_missing)) + \
                              list(zip(bars_cross, cross_vals, cross_missing)):
        if missing:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.005,
                    "n/a", ha="center", va="bottom", fontsize=8, color="grey")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([spec.label.replace("SchNet, ", "") for spec, _, _ in rows],
                       rotation=12, ha="right")
    ax.set_ylabel(r"Test Pearson $\rho$")
    ax.set_title("In-distribution vs cross-evaluation performance")
    ax.set_ylim(0, max(max(in_vals), max(cross_vals), 0.5) * 1.18)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save_both(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — predicted-vs-actual scatter (real inference pass)
# ---------------------------------------------------------------------------

def fig_predictions(runs_dir: Path, headline_run: str, out_base: Path) -> None:
    """Run the headline checkpoint on its in-distribution test set and scatter
    the predicted vs actual −log K values. Annotates with Pearson, RMSE."""
    import matplotlib.pyplot as plt
    import torch
    from torch_geometric.loader import DataLoader

    from protai.data import PreShardedDataset
    from protai.training.lit_module import ProtAILitModule
    from protai.config import resolve_path

    ckpt = runs_dir / headline_run / "best.ckpt"
    if not ckpt.exists():
        print(f"  [skip] no best.ckpt at {ckpt}")
        return

    print(f"  loading {ckpt.relative_to(REPO_ROOT) if REPO_ROOT in ckpt.parents else ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = ProtAILitModule.load_from_checkpoint(str(ckpt), map_location=device)
    module.eval()
    cfg = module.cfg

    processed = resolve_path(cfg.data.processed_dir)
    splits = resolve_path(cfg.data.splits_dir)
    test_ds = PreShardedDataset(
        processed_dir=processed,
        split_file=splits / cfg.data.test_split,
        target=cfg.model.target,
        frame_strategy="frame_zero",
        edge_cutoff=cfg.data.edge_cutoff,
        node_feature=("atomic_number" if cfg.model.name == "schnet"
                      else "one_hot_element"),
    )
    loader = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                        shuffle=False, num_workers=0)

    ys, yhats = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = module(batch)
            # Multitask returns dict; prefer the log-K head when present.
            if isinstance(pred, dict):
                head = "logk" if "logk" in pred else "energy"
                raw = pred[head]
                truth = (batch.y_logk if head == "logk" else batch.y_energy)
            else:
                raw = pred
                truth = batch.y
            yhat = module._denormalize(raw)
            ys.append(truth.cpu().numpy().flatten())
            yhats.append(yhat.cpu().numpy().flatten())

    y = np.concatenate(ys).astype(float)
    yhat = np.concatenate(yhats).astype(float)
    finite = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[finite], yhat[finite]

    pearson = float(np.corrcoef(y, yhat)[0, 1]) if y.std() > 0 and yhat.std() > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    n = len(y)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, yhat, s=14, alpha=0.45, color="#0072B2", edgecolor="none")
    lo = float(min(y.min(), yhat.min()))
    hi = float(max(y.max(), yhat.max()))
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="black", linestyle="--", linewidth=1.0, label=r"$\hat{y} = y$")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"True {TARGET_LABEL}")
    ax.set_ylabel(f"Predicted {TARGET_LABEL}")
    ax.set_title(f"{headline_run}: predicted vs actual on test set")
    ax.legend(loc="upper left")

    annotation = (
        f"$N$ = {n:,}\n"
        f"Pearson $\\rho$ = {pearson:.3f}\n"
        f"RMSE = {rmse:.2f}"
    )
    ax.text(0.97, 0.04, annotation, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="grey", linewidth=0.6, alpha=0.9))
    fig.tight_layout()
    _save_both(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — four-panel metric summary
# ---------------------------------------------------------------------------

def fig_metric_panel(runs_dir: Path, runs: List[RunSpec], out_base: Path) -> None:
    """Four-panel figure (Pearson, Spearman, RMSE, MAE) × in-dist + cross-eval.

    Compact one-glance summary suitable as a paper's headline figure.
    Higher is better for Pearson/Spearman; lower is better for RMSE/MAE.
    """
    import matplotlib.pyplot as plt

    metrics = [
        ("test/pearson", "pearson", "Pearson $\\rho$", True),
        ("test/spearman", "spearman", "Spearman $\\rho$", True),
        ("test/rmse", "rmse", f"RMSE ({TARGET_UNIT})", False),
        ("test/mae", "mae", f"MAE ({TARGET_UNIT})", False),
    ]

    rows = []
    for spec in runs:
        ind: Dict[str, Optional[float]] = {}
        for tb_tag, _, _, _ in metrics:
            series = _read_scalar(runs_dir / spec.name, tb_tag)
            ind[tb_tag] = series[-1][1] if series else None
        cross = _read_json(runs_dir / spec.name / "test_metrics_similarity.json") or {}
        rows.append((spec, ind, cross))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    bar_width = 0.36
    x = np.arange(len(rows))

    for ax, (tb_tag, json_key, ylabel, higher_better) in zip(axes, metrics):
        in_vals = [ind.get(tb_tag) for _, ind, _ in rows]
        cross_vals = [cross.get(json_key) for _, _, cross in rows]
        in_plot = [v if v is not None else 0.0 for v in in_vals]
        cross_plot = [v if v is not None else 0.0 for v in cross_vals]
        ax.bar(x - bar_width / 2, in_plot, bar_width,
               color=[spec.color for spec, _, _ in rows],
               edgecolor="black", linewidth=0.4, label="In-dist")
        ax.bar(x + bar_width / 2, cross_plot, bar_width,
               color=[spec.color for spec, _, _ in rows],
               hatch="///", alpha=0.55, edgecolor="black", linewidth=0.4,
               label="Cross-eval")

        for xi, v in zip(x - bar_width / 2, in_vals):
            if v is not None:
                ax.text(xi, v + 0.02 * max(in_plot + cross_plot, default=1),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        for xi, v in zip(x + bar_width / 2, cross_vals):
            if v is not None:
                ax.text(xi, v + 0.02 * max(in_plot + cross_plot, default=1),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([spec.name.replace("_logk", "")
                                   .replace("_energy", " (mt)")
                                   .replace("_", " ")
                            for spec, _, _ in rows],
                           rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        arrow = "↑" if higher_better else "↓"
        ax.set_title(f"{ylabel.split(' (')[0]} {arrow}")

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Per-metric comparison: in-distribution vs cross-evaluation", y=1.02)
    fig.tight_layout()
    _save_both(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save helper — writes both PNG and PDF
# ---------------------------------------------------------------------------

def _save_both(fig, out_base: Path) -> None:
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"  wrote {png.name}  +  {pdf.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"),
                        help="Directory containing per-run subdirectories.")
    parser.add_argument("--out-dir",
                        default=str(REPO_ROOT.parent / "Revised - Final Report Research - FYP-1" / "ThesisFigs"),
                        help="Where to write the figure files. Defaults to the report's ThesisFigs/.")
    parser.add_argument("--headline-run", default=DEFAULT_HEADLINE_RUN,
                        help=f"Run used for the loss + scatter figures (default: {DEFAULT_HEADLINE_RUN}).")
    parser.add_argument("--skip-scatter", action="store_true",
                        help="Skip the predicted-vs-actual scatter (avoids loading the model).")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading runs from {runs_dir}")
    print(f"Writing figures to {out_dir}\n")

    _setup_matplotlib()
    runs = _resolve_runs(runs_dir)
    if not runs:
        sys.exit(f"[fatal] no runs found under {runs_dir} matching the expected names: "
                 f"{[r.name for r in RUNS]}")

    print("\n[fig 1] training curves")
    fig_training_curves(runs_dir, runs, out_dir / "fig_training_curves")

    print("\n[fig 2] loss curves")
    fig_loss_curves(runs_dir, args.headline_run, out_dir / "fig_loss_curves")

    print("\n[fig 3] cross-evaluation comparison")
    fig_cross_eval(runs_dir, runs, out_dir / "fig_cross_eval")

    if not args.skip_scatter:
        print("\n[fig 4] predicted-vs-actual scatter")
        try:
            fig_predictions(runs_dir, args.headline_run, out_dir / "fig_predictions")
        except Exception as e:
            print(f"  [warn] scatter failed: {type(e).__name__}: {e}")
            print("  pass --skip-scatter to skip this step explicitly.")
    else:
        print("\n[fig 4] predicted-vs-actual scatter — skipped (--skip-scatter)")

    print("\n[fig 5] four-panel metric summary")
    fig_metric_panel(runs_dir, runs, out_dir / "fig_metric_panel")

    print("\nDone.")


if __name__ == "__main__":
    main()
