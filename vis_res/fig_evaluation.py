"""
fig_evaluation.py
=================
Figures from the evaluation pipeline results.

Source: evaluace/results/{pipeline}_{generator}.jsonl
  - model field = EVALUATOR model (cross-validation: generator ≠ evaluator)
  - filename suffix = GENERATOR model provider (openai → gpt-5.4, anthropic → claude-sonnet-4-6)

Figures produced (saved to vis_res/figures/):
  eval_01_gate_pass_rate.pdf          — gate pass rate by pipeline × generator model
  eval_02_symptom_coverage.pdf        — symptom coverage (box plot)
  eval_03_quality_score.pdf           — quality score ratio (box plot)
  eval_04_diagnosis_accuracy.pdf      — diagnosis correct rate (1st attempt + overall)
  eval_05_summary_radar.pdf           — radar chart comparing all four pipelines
  eval_06_model_comparison.pdf        — side-by-side: GPT-5.4 vs Claude on all metrics
  eval_07_per_diagnosis_accuracy.pdf  — accuracy per ICD-10 code (horizontal bar)
  eval_08_heatmap_icd_pipeline.pdf    — heatmap: ICD × pipeline diagnosis accuracy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
EVAL    = ROOT / "evaluace" / "results"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

PIPELINES   = ["baseline", "tools_agent", "rag", "multiagent"]
GEN_MODELS  = ["openai", "anthropic"]

PIPELINE_LABELS = {
    "baseline":    "Baseline",
    "tools_agent": "Tools agent",
    "rag":         "RAG",
    "multiagent":  "Multi-agent",
}
GEN_MODEL_LABELS = {
    "openai":    "GPT-5.4",
    "anthropic": "Claude Sonnet 4.6",
}
PALETTE = {
    "baseline":    "#4C72B0",
    "tools_agent": "#DD8452",
    "rag":         "#55A868",
    "multiagent":  "#C44E52",
}
MODEL_COLORS = {
    "openai":    "#74ADD1",
    "anthropic": "#FF7043",
}
HATCH = {"openai": "", "anthropic": "///"}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_evaluations() -> pd.DataFrame:
    rows = []
    for pipeline in PIPELINES:
        for gen_model in GEN_MODELS:
            path = EVAL / f"{pipeline}_{gen_model}.jsonl"
            if not path.exists():
                print(f"  WARN: {path.name} not found")
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    diag = rec.get("diagnosis", {})
                    qual = rec.get("quality", {})
                    symp = rec.get("symptoms", {})
                    gate = rec.get("gate", {})

                    rows.append({
                        "pipeline":         pipeline,
                        "gen_model":        gen_model,
                        "eval_model":       rec.get("model", ""),
                        "icd_code":         rec.get("icd_code", ""),
                        "diagnosis_name":   rec.get("diagnosis_name", ""),
                        "gender":           rec.get("gender", ""),
                        "age":              rec.get("age"),
                        # Gate
                        "gate_passed":      bool(gate.get("passed", False)),
                        # Symptoms
                        "symptom_coverage": float(symp.get("symptom_coverage", 0.0)),
                        # Quality
                        "quality_score":    int(qual.get("score", 0)),
                        "quality_max":      int(qual.get("max_score", 14)),
                        "quality_ratio":    (qual.get("score", 0) / qual.get("max_score", 14)
                                             if qual.get("max_score") else 0.0),
                        # Diagnosis
                        "diag_correct":     bool(diag.get("correct", False)),
                        "diag_attempt":     diag.get("correct_attempt"),  # 1/2/3/None
                        "diag_correct_1":   (diag.get("correct_attempt") == 1),
                    })

    df = pd.DataFrame(rows)
    df["pipeline_label"] = df["pipeline"].map(PIPELINE_LABELS)
    df["gen_model_label"] = df["gen_model"].map(GEN_MODEL_LABELS)
    return df


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _grouped_bar_metric(ax, df, metric, ylabel, title,
                        ylim=(0, 110), fmt_pct=True, agg="mean"):
    """Grouped bar: pipeline × gen_model for a given metric (mean or sum)."""
    if agg == "mean":
        pivot = (
            df.groupby(["pipeline_label", "gen_model"])[metric]
            .mean()
            .mul(100 if fmt_pct else 1)
            .unstack("gen_model")
            .reindex([PIPELINE_LABELS[p] for p in PIPELINES])
        )
    else:
        pivot = (
            df.groupby(["pipeline_label", "gen_model"])[metric]
            .agg(agg)
            .unstack("gen_model")
            .reindex([PIPELINE_LABELS[p] for p in PIPELINES])
        )

    x     = np.arange(len(pivot))
    width = 0.35
    n_m   = len(pivot.columns)

    for i, model in enumerate(pivot.columns):
        offset = (i - (n_m - 1) / 2) * width
        for j, pl in enumerate(pivot.index):
            p_key = [k for k, v in PIPELINE_LABELS.items() if v == pl][0]
            val   = pivot[model].iloc[j]
            ax.bar(
                x[j] + offset, val,
                width=width * 0.9,
                color=PALETTE[p_key],
                hatch=HATCH.get(model, ""),
                edgecolor="white", linewidth=0.8,
                label=(f"{GEN_MODEL_LABELS.get(model, model)}"
                       if j == 0 else "_"),
            )
            top = ylim[1] * 0.012 if ylim else 0.5
            ax.text(
                x[j] + offset, val + top,
                f"{val:.0f}%" if fmt_pct else f"{val:.2f}",
                ha="center", va="bottom", fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if fmt_pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Generator model", ncol=2, framealpha=0.9, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


# ── Individual figures ─────────────────────────────────────────────────────────

def fig_gate_pass_rate(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar_metric(
        ax, df, "gate_passed",
        ylabel="Gate pass rate (%)",
        title="Medical Plausibility Gate — Pass Rate by Pipeline and Generator Model",
    )
    plt.tight_layout()
    out = OUT_DIR / "eval_01_gate_pass_rate.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_symptom_coverage(df: pd.DataFrame):
    """Box plot of symptom coverage by pipeline, coloured by generator model."""
    fig, ax = plt.subplots(figsize=(9, 5))

    order = [PIPELINE_LABELS[p] for p in PIPELINES]
    sns.boxplot(
        data=df,
        x="pipeline_label", y="symptom_coverage",
        hue="gen_model_label",
        order=order,
        palette=[MODEL_COLORS["openai"], MODEL_COLORS["anthropic"]],
        width=0.55,
        linewidth=1.2,
        fliersize=3,
        ax=ax,
    )

    # Target reference line (70%)
    ax.axhline(0.70, color="#C44E52", linestyle="--", linewidth=1.3,
               label="Target ≥ 70%")

    ax.set_xlabel("")
    ax.set_ylabel("Symptom coverage (fraction)")
    ax.set_title("Symptom Coverage by Pipeline and Generator Model")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend(title="Generator model", framealpha=0.9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "eval_02_symptom_coverage.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_quality_score(df: pd.DataFrame):
    """Box plot of quality score ratio by pipeline × generator model."""
    fig, ax = plt.subplots(figsize=(9, 5))

    order = [PIPELINE_LABELS[p] for p in PIPELINES]
    sns.boxplot(
        data=df,
        x="pipeline_label", y="quality_ratio",
        hue="gen_model_label",
        order=order,
        palette=[MODEL_COLORS["openai"], MODEL_COLORS["anthropic"]],
        width=0.55,
        linewidth=1.2,
        fliersize=3,
        ax=ax,
    )

    ax.axhline(0.77, color="#C44E52", linestyle="--", linewidth=1.3,
               label="Target ≥ 77%")

    ax.set_xlabel("")
    ax.set_ylabel(f"Quality score ratio  (score / {df['quality_max'].mode()[0]})")
    ax.set_title("Quality Score Ratio by Pipeline and Generator Model")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend(title="Generator model", framealpha=0.9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "eval_03_quality_score.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_diagnosis_accuracy(df: pd.DataFrame):
    """Grouped bar: diagnosis correct at attempt 1 and overall (any attempt)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (metric, label) in zip(
        axes,
        [("diag_correct_1", "Correct on 1st attempt (%)"),
         ("diag_correct",   "Correct on any attempt (%)")]
    ):
        _grouped_bar_metric(
            ax, df, metric,
            ylabel=label,
            title=label,
        )

    axes[0].set_title("Diagnosis Accuracy — First Attempt")
    axes[1].set_title("Diagnosis Accuracy — Any Attempt (up to 3)")
    fig.suptitle("Diagnosis Accuracy by Pipeline and Generator Model", fontsize=14, y=1.02)

    plt.tight_layout()
    out = OUT_DIR / "eval_04_diagnosis_accuracy.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_summary_radar(df: pd.DataFrame):
    """Radar chart: all four pipelines on 5 metrics (averaged over both models)."""
    metrics = ["gate_passed", "symptom_coverage", "quality_ratio", "diag_correct_1", "diag_correct"]
    labels  = ["Gate\npass rate", "Symptom\ncoverage", "Quality\nscore", "Diagnosis\n(1st attempt)", "Diagnosis\n(any attempt)"]
    N = len(metrics)

    # Aggregate per pipeline (mean over both gen models)
    agg = df.groupby("pipeline")[metrics].mean()

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for pipeline in PIPELINES:
        if pipeline not in agg.index:
            continue
        values = agg.loc[pipeline, metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=PALETTE[pipeline], label=PIPELINE_LABELS[pipeline])
        ax.fill(angles, values, alpha=0.10, color=PALETTE[pipeline])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_title("Pipeline Comparison — All Key Metrics\n(averaged over both generator models)",
                 pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / "eval_05_summary_radar.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_model_comparison(df: pd.DataFrame):
    """Side-by-side bar: GPT-5.4 vs Claude on four key metrics, one panel per pipeline."""
    metrics = ["gate_passed", "symptom_coverage", "quality_ratio", "diag_correct"]
    metric_labels = ["Gate pass rate", "Symptom coverage", "Quality ratio", "Diagnosis accuracy\n(any attempt)"]

    fig, axes = plt.subplots(1, len(PIPELINES), figsize=(14, 5), sharey=False)

    for ax, pipeline in zip(axes, PIPELINES):
        sub = df[df["pipeline"] == pipeline]
        vals_openai    = [sub[sub["gen_model"] == "openai"][m].mean()    for m in metrics]
        vals_anthropic = [sub[sub["gen_model"] == "anthropic"][m].mean() for m in metrics]

        x     = np.arange(len(metrics))
        width = 0.38
        ax.bar(x - width / 2, vals_openai,    width, label="GPT-5.4",
               color=MODEL_COLORS["openai"],    edgecolor="white")
        ax.bar(x + width / 2, vals_anthropic, width, label="Claude 4.6",
               color=MODEL_COLORS["anthropic"], edgecolor="white", hatch="///")

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(PIPELINE_LABELS[pipeline], fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("Score (fraction)")

    fig.suptitle("Generator Model Comparison (GPT-5.4 vs Claude Sonnet 4.6) per Pipeline",
                 fontsize=13, y=1.02)
    # single shared legend centred below all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Generator model", ncol=2, fontsize=9,
               framealpha=0.9, loc="upper center",
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out = OUT_DIR / "eval_06_model_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_per_diagnosis_accuracy(df: pd.DataFrame):
    """Horizontal bar: diagnosis accuracy per ICD-10 code across all pipelines."""
    acc = (
        df.groupby(["icd_code", "diagnosis_name"])["diag_correct"]
        .mean()
        .mul(100)
        .reset_index()
        .sort_values("diag_correct", ascending=True)
    )
    acc["label"] = acc["icd_code"] + "  " + acc["diagnosis_name"].str[:40]

    fig, ax = plt.subplots(figsize=(11, max(6, len(acc) * 0.38)))

    colors = [
        "#C44E52" if v < 40 else ("#DD8452" if v < 70 else "#55A868")
        for v in acc["diag_correct"]
    ]
    bars = ax.barh(acc["label"], acc["diag_correct"], color=colors, edgecolor="white",
                   height=0.7)

    for bar, val in zip(bars, acc["diag_correct"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8.5)

    ax.set_xlabel("Diagnosis accuracy — any attempt (%)")
    ax.set_title("Diagnosis Accuracy per ICD-10 Code\n(averaged across all pipelines and generator models)")
    ax.set_xlim(0, 112)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axvline(50, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # legend for colours
    patches = [
        mpatches.Patch(color="#C44E52", label="< 40% (hard)"),
        mpatches.Patch(color="#DD8452", label="40–70% (moderate)"),
        mpatches.Patch(color="#55A868", label="≥ 70% (easy)"),
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "eval_07_per_diagnosis_accuracy.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_heatmap_icd_pipeline(df: pd.DataFrame):
    """Heatmap: diagnosis accuracy per ICD-10 code × pipeline."""
    heat = (
        df.groupby(["icd_code", "pipeline_label"])["diag_correct"]
        .mean()
        .mul(100)
        .unstack("pipeline_label")
        .reindex(columns=[PIPELINE_LABELS[p] for p in PIPELINES])
    )

    # Add short diagnosis name to ICD code
    icd_names = df.groupby("icd_code")["diagnosis_name"].first().str[:35]
    heat.index = heat.index + "  " + heat.index.map(icd_names)

    fig_h = max(7, len(heat) * 0.45)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    sns.heatmap(
        heat,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0, vmax=100,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9},
        cbar_kws={"label": "Diagnosis accuracy (%)", "shrink": 0.6},
    )

    ax.set_title("Diagnosis Accuracy (%) per ICD-10 Code and Pipeline\n"
                 "(averaged over both generator models and all attempts)")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "eval_08_heatmap_icd_pipeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading evaluation data...")
    df = load_evaluations()
    print(f"  Loaded {len(df)} records")
    print(f"  Pipelines:       {df['pipeline'].value_counts().to_dict()}")
    print(f"  Generator models:{df['gen_model'].value_counts().to_dict()}")
    print(f"  ICD codes:       {df['icd_code'].nunique()}")
    print()

    print("Generating evaluation figures...")
    fig_gate_pass_rate(df)
    fig_symptom_coverage(df)
    fig_quality_score(df)
    fig_diagnosis_accuracy(df)
    fig_summary_radar(df)
    fig_model_comparison(df)
    fig_per_diagnosis_accuracy(df)
    fig_heatmap_icd_pipeline(df)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
