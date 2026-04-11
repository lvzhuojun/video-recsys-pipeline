"""Generate training curves and model comparison charts from saved JSON history.

Usage (from project root):
    python experiments/plot_results.py

Reads:
    experiments/results/*_history.json   — per-epoch metrics from each run
    experiments/results/ranking_results.json
    experiments/results/ablation_results.json

Outputs:
    experiments/results/figures/
        training_curves_ranking.png     — AUC per epoch for DeepFM / DIN / DIEN
        training_curves_retrieval.png   — Recall@10 per epoch for Two-Tower variants
        training_curves_mmoe.png        — Watch/Like AUC per epoch for MMoE
        model_comparison.png            — Bar chart: best AUC across all ranking models
        ablation_chart.png              — Bar chart: ablation study results
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")   # headless / Windows-safe backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    _MPL = False
    print("matplotlib not found — skipping plot generation.")

RESULTS_DIR = _ROOT / "experiments" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Consistent color palette across all charts
COLORS = {
    "deepfm":   "#2196F3",   # blue
    "din":      "#FF9800",   # orange
    "dien":     "#9C27B0",   # purple
    "mmoe":     "#4CAF50",   # green
    "two_tower_mean_pool": "#F44336",   # red
    "two_tower_sasrec":    "#00BCD4",   # cyan
}
LIGHTER = {k: v + "88" for k, v in COLORS.items()}   # 50% alpha for secondary metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_history(name: str) -> list[dict] | None:
    path = RESULTS_DIR / f"{name}_history.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _styled_ax(ax, title: str, xlabel: str = "Epoch", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Ranking training curves (DeepFM / DIN / DIEN)
# ---------------------------------------------------------------------------

def plot_ranking_curves() -> None:
    models = ["deepfm", "din", "dien"]
    histories = {m: _load_history(m) for m in models}
    available = {m: h for m, h in histories.items() if h}

    if not available:
        print("No ranking history JSON found — skipping ranking curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Ranking Model Training Curves (Synthetic KuaiRec-Schema Data)",
                 fontsize=14, fontweight="bold", y=1.01)

    # Left: AUC
    ax = axes[0]
    for name, hist in available.items():
        epochs = [d["epoch"] for d in hist]
        aucs   = [d.get("auc", float("nan")) for d in hist]
        ax.plot(epochs, aucs, color=COLORS.get(name, "#888"),
                linewidth=2, marker="o", markersize=3, label=name.upper())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _styled_ax(ax, "Validation AUC over Epochs", ylabel="AUC")

    # Right: LogLoss
    ax = axes[1]
    for name, hist in available.items():
        epochs    = [d["epoch"] for d in hist]
        logloss   = [d.get("logloss", float("nan")) for d in hist]
        ax.plot(epochs, logloss, color=COLORS.get(name, "#888"),
                linewidth=2, marker="s", markersize=3, label=name.upper())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _styled_ax(ax, "Validation LogLoss over Epochs", ylabel="LogLoss")

    plt.tight_layout()
    out = FIGURES_DIR / "training_curves_ranking.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 2. Retrieval training curves (Two-Tower variants)
# ---------------------------------------------------------------------------

def plot_retrieval_curves() -> None:
    variants = ["two_tower_mean_pool", "two_tower_sasrec"]
    histories = {v: _load_history(v) for v in variants}
    available = {v: h for v, h in histories.items() if h}

    if not available:
        print("No retrieval history JSON found — skipping retrieval curves.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, hist in available.items():
        epochs = [d["epoch"] for d in hist]
        # Try common retrieval metric keys
        for metric_key in ("recall@10", "recall_10", "Recall@10"):
            vals = [d.get(metric_key, None) for d in hist]
            if any(v is not None for v in vals):
                break
        vals = [v if v is not None else float("nan") for v in vals]
        label = "Two-Tower (MeanPool)" if "mean_pool" in name else "Two-Tower (SASRec)"
        ax.plot(epochs, vals, color=COLORS.get(name, "#888"),
                linewidth=2, marker="o", markersize=3, label=label)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _styled_ax(ax, "Recall@10 over Epochs — Two-Tower Variants",
               ylabel="Recall@10")
    ax.set_title(ax.get_title(), fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "training_curves_retrieval.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 3. MMoE multi-task training curves
# ---------------------------------------------------------------------------

def plot_mmoe_curves() -> None:
    hist = _load_history("mmoe")
    if not hist:
        print("No MMoE history JSON found — skipping MMoE curves.")
        return

    epochs     = [d["epoch"] for d in hist]
    watch_auc  = [d.get("watch_auc",  float("nan")) for d in hist]
    like_auc   = [d.get("like_auc",   float("nan")) for d in hist]
    train_loss = [d.get("train_loss", float("nan")) for d in hist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("MMoE Multi-Task Training Curves", fontsize=14,
                 fontweight="bold", y=1.01)

    ax = axes[0]
    ax.plot(epochs, watch_auc, color=COLORS["mmoe"], linewidth=2,
            marker="o", markersize=3, label="Watch AUC (primary)")
    ax.plot(epochs, like_auc,  color=LIGHTER["mmoe"], linewidth=2,
            marker="s", markersize=3, label="Like AUC", linestyle="--")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random baseline")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _styled_ax(ax, "Validation AUC — Both Tasks", ylabel="AUC")

    ax = axes[1]
    ax.plot(epochs, train_loss, color="#F44336", linewidth=2,
            marker="o", markersize=3, label="Train Loss (combined)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _styled_ax(ax, "Combined MTL Training Loss", ylabel="Loss")

    plt.tight_layout()
    out = FIGURES_DIR / "training_curves_mmoe.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 4. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison() -> None:
    ranking_path = RESULTS_DIR / "ranking_results.json"
    mtl_path     = RESULTS_DIR / "multitask_results.json"
    if not ranking_path.exists():
        print("ranking_results.json not found — skipping comparison chart.")
        return

    with open(ranking_path, encoding="utf-8") as f:
        ranking = json.load(f)

    models, aucs, gaucs = [], [], []
    for name, metrics in ranking.items():
        models.append(name.upper())
        aucs.append(metrics.get("auc", 0))
        gaucs.append(metrics.get("gauc", 0))

    if mtl_path.exists():
        with open(mtl_path, encoding="utf-8") as f:
            mtl = json.load(f)
        models.append("MMoE\n(watch)")
        aucs.append(mtl.get("watch_auc", 0))
        gaucs.append(mtl.get("watch_gauc", 0))

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models)*1.6), 5))
    bars1 = ax.bar([i - width/2 for i in x], aucs,  width, label="AUC",
                   color="#2196F3", alpha=0.85, edgecolor="white")
    bars2 = ax.bar([i + width/2 for i in x], gaucs, width, label="GAUC",
                   color="#FF9800", alpha=0.85, edgecolor="white")

    # Value labels on bars
    for bar in (*bars1, *bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random (0.5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0.45, min(1.0, max(aucs + gaucs) + 0.07))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ranking Model Comparison — AUC vs GAUC\n"
                 "(Synthetic KuaiRec-Schema Data with Feature-Label Correlation)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 5. Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation_chart() -> None:
    path = RESULTS_DIR / "ablation_results.json"
    if not path.exists():
        print("ablation_results.json not found — skipping ablation chart.")
        return

    with open(path, encoding="utf-8") as f:
        ablation = json.load(f)

    # Separate retrieval (Recall@10) and ranking (AUC) variants
    retrieval_names, retrieval_vals = [], []
    ranking_names,   ranking_vals   = [], []

    # ablation_results.json is a flat dict: {metric_name: float_value}
    # Group into retrieval (recall10) and ranking (auc) entries for plotting
    retrieval_map = {
        "In-batch\nNeg": ablation.get("retrieval_inbatch_recall10"),
        "Random\nNeg": ablation.get("retrieval_random_recall10"),
        "No Seq": ablation.get("retrieval_noseq_recall10"),
        "MeanPool": ablation.get("meanpool_recall10"),
        "SASRec": ablation.get("sasrec_recall10"),
    }
    ranking_map = {
        "DeepFM": ablation.get("deepfm_auc"),
        "DeepFM\n(no FM)": ablation.get("deepfm_nofm_auc"),
        "MLP": ablation.get("mlp_auc"),
        "DIN": ablation.get("din_auc"),
    }
    for k, v in retrieval_map.items():
        if v is not None:
            retrieval_names.append(k)
            retrieval_vals.append(v)
    for k, v in ranking_map.items():
        if v is not None:
            ranking_names.append(k)
            ranking_vals.append(v)

    n_panels = (1 if retrieval_names else 0) + (1 if ranking_names else 0)
    if n_panels == 0:
        print("No parseable ablation data — skipping ablation chart.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    if retrieval_names:
        ax = axes[panel]; panel += 1
        bars = ax.bar(retrieval_names, retrieval_vals, color="#2196F3",
                      alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_title("Retrieval Ablation Study", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    if ranking_names:
        ax = axes[panel]
        bars = ax.bar(ranking_names, ranking_vals, color="#FF9800",
                      alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2, label="Random")
        ax.set_ylabel("AUC", fontsize=11)
        ax.set_title("Ranking Ablation Study", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "ablation_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _MPL:
        print("Install matplotlib: pip install matplotlib")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating figures in {FIGURES_DIR} …")

    plot_ranking_curves()
    plot_retrieval_curves()
    plot_mmoe_curves()
    plot_model_comparison()
    plot_ablation_chart()

    print("Done. All available figures saved.")


if __name__ == "__main__":
    main()
