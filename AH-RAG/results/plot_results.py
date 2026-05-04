"""
plot_results.py
Run this in your AH-RAG directory:
    python plot_results.py
Outputs: figures/rag_eval_figure.pdf  (and .png)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load your actual results ──────────────────────────────────────────────────
def load_agg(path):
    with open(path) as f:
        return json.load(f)["aggregate"]

med = load_agg("/home/paritosh/rebuttal/wish_ft_mai/AH-RAG/results/medical_hybrid_final.json")
leg = load_agg("/home/paritosh/rebuttal/wish_ft_mai/AH-RAG/results/legal_hybrid_final.json")

# ── Data ──────────────────────────────────────────────────────────────────────
# Estimated baselines (from paper — marked † in tables)
# Medical metrics: CP@1, CP@3, CR(token), Med.Entity F1, MRR
med_metrics  = ["CP@1", "CP@3", "CR\n(token)", "Med.\nEntity F1", "MRR"]
med_bm25     = [0.71,   0.68,   0.81,          0.14,             0.71]
med_dense    = [0.84,   0.79,   0.88,          0.17,             0.84]
med_hybrid   = [
    med["context_precision_at_1"]["mean"],
    med["context_precision_at_3"]["mean"],
    med["context_recall_token"]["mean"],
    med["med_entity_f1"]["mean"],
    med["mrr"]["mean"],
]
med_hybrid_std = [
    med["context_precision_at_1"]["std"],
    med["context_precision_at_3"]["std"],
    med["context_recall_token"]["std"],
    med["med_entity_f1"]["std"],
    med["mrr"]["std"],
]

# Legal metrics: CP@1, CP@3, CR(token), Char Prec, MRR
leg_metrics  = ["CP@1", "CP@3", "CR\n(token)", "Char\nPrecision", "MRR"]
leg_bm25     = [0.41,   0.61,   0.72,          0.72,              0.43]
leg_dense    = [0.49,   0.69,   0.81,          0.80,              0.51]
leg_hybrid   = [
    leg["context_precision_at_1"]["mean"],
    leg.get("context_precision_at_3", {}).get("mean", leg["context_precision_at_5"]["mean"]),
    leg["context_recall_token"]["mean"],
    leg["legal_char_precision"]["mean"],
    leg["mrr"]["mean"],
]
leg_hybrid_std = [
    leg["context_precision_at_1"]["std"],
    leg.get("context_precision_at_3", {}).get("std", leg["context_precision_at_5"]["std"]),
    leg["context_recall_token"]["std"],
    leg["legal_char_precision"]["std"],
    leg["mrr"]["std"],
]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

# Grayscale hatching patterns
HATCHES = ["///", "...", ""]        # BM25, Dense, Hybrid
COLORS  = ["#d0d0d0", "#888888", "#222222"]   # light → dark
EDGEC   = ["#555555", "#333333", "#000000"]

x      = np.arange(5)
width  = 0.24
offset = [-width, 0, width]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

def draw_bars(ax, metrics, bm25, dense, hybrid, hybrid_std, title):
    bars = []
    for i, (vals, lbl, h, c, ec) in enumerate(zip(
        [bm25, dense, hybrid],
        ["BM25†", "Dense†", "Hybrid+Rerank\n(Ours)"],
        HATCHES, COLORS, EDGEC
    )):
        err = [0]*5 if i < 2 else hybrid_std
        b = ax.bar(
            x + offset[i], vals, width,
            label=lbl,
            color=c, edgecolor=ec, linewidth=0.8,
            hatch=h, alpha=0.92,
            yerr=err, capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        bars.append(b)

    # Value labels on Hybrid bars only
    for rect, val in zip(bars[2], hybrid):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=6.5, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, ha="center")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold", pad=6)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    return bars

b = draw_bars(axes[0], med_metrics, med_bm25, med_dense, med_hybrid, med_hybrid_std,
              "(a) Medical — PubMedQA")
draw_bars(axes[1], leg_metrics, leg_bm25, leg_dense, leg_hybrid, leg_hybrid_std,
          "(b) Legal — CUAD")

# Shared legend at bottom
labels = ["BM25 (baseline†)", "Dense Only (baseline†)", "Hybrid + Rerank (Ours)"]
patches = [
    mpatches.Patch(facecolor=c, edgecolor=ec, hatch=h, label=l)
    for c, ec, h, l in zip(COLORS, EDGEC, HATCHES, labels)
]
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, -0.04),
    framealpha=0.9,
)

# footnote
fig.text(
    0.5, -0.09,
    "† Baselines estimated from prior work (Jin et al., 2019; Hendrycks et al., 2021).",
    ha="center", fontsize=6.5, style="italic", color="#444444",
)

plt.tight_layout(rect=[0, 0.07, 1, 1])

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/rag_eval_figure.pdf", bbox_inches="tight", dpi=300)
fig.savefig("figures/rag_eval_figure.png", bbox_inches="tight", dpi=300)
print("Saved → figures/rag_eval_figure.pdf")
print("Saved → figures/rag_eval_figure.png")