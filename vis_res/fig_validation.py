"""
fig_validation.py
=================
Figures from FHIR structural validation results.

Source: scenarios/{pipeline}/validation.jsonl
Fields: model (generator: openai/anthropic), patient_id, res_struct, all_errors, code_errors

Figures produced (saved to vis_res/figures/):
  val_01_structural_validity.pdf  — structural validity rate by pipeline × generator model
  val_02_code_errors_mean.pdf     — mean code errors per scenario
  val_03_zero_code_errors.pdf     — % of scenarios with zero code errors
  val_04_error_type_breakdown.pdf — top recurring error categories
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
SCEN    = ROOT / "scenarios"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

PIPELINES = ["baseline", "tools_agent", "rag", "multiagent"]
MODELS    = ["openai", "anthropic"]

PIPELINE_LABELS = {
    "baseline":    "Baseline",
    "tools_agent": "Tools agent",
    "rag":         "RAG",
    "multiagent":  "Multi-agent",
}
MODEL_LABELS = {
    "openai":    "GPT-5.4",
    "anthropic": "Claude Sonnet 4.6",
}

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE = {
    "baseline":    "#4C72B0",
    "tools_agent": "#DD8452",
    "rag":         "#55A868",
    "multiagent":  "#C44E52",
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

def load_validation() -> pd.DataFrame:
    rows = []
    for pipeline in PIPELINES:
        path = SCEN / pipeline / "validation.jsonl"
        if not path.exists():
            print(f"  WARN: {path} not found")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append({
                    "pipeline":     pipeline,
                    "model":        rec["model"],          # openai / anthropic (generator)
                    "patient_id":   rec["patient_id"],
                    "res_struct":   bool(rec["res_struct"]),
                    "code_errors":  int(rec["code_errors"]),
                    "n_errors":     len(rec.get("all_errors", [])),
                    "errors_raw":   rec.get("all_errors", []),
                })
    df = pd.DataFrame(rows)
    df["pipeline_label"] = df["pipeline"].map(PIPELINE_LABELS)
    df["model_label"]    = df["model"].map(MODEL_LABELS)
    # extract ICD from patient_id e.g. "J00_8_MatějKraus" → "J00"
    df["icd"] = df["patient_id"].str.extract(r"^([A-Z]\d+\.?\d*)")
    return df


def _classify_error(msg: str) -> str:
    """Map a raw error message to a short category label."""
    m = msg.lower()
    if "atc" in m or "medication" in m or "whocc" in m:
        return "ATC / medication code"
    if "snomed" in m or "sct" in m:
        return "SNOMED CT code"
    if "loinc" in m:
        return "LOINC code"
    if "icd" in m:
        return "ICD-10 code"
    if "empty" in m or "cannot be empty" in m or "@value" in m:
        return "Missing required value"
    if "constraint" in m or "ele-1" in m:
        return "FHIR constraint"
    return "Other"


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _grouped_bar(ax, df_pivot, ylabel, title, ylim=None, fmt_pct=False):
    """Draw a grouped bar chart on ax from a (pipeline × model) pivot."""
    n_pipelines = len(df_pivot)
    n_models    = len(df_pivot.columns)
    x           = np.arange(n_pipelines)
    width       = 0.35

    for i, model in enumerate(df_pivot.columns):
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            df_pivot[model],
            width=width * 0.9,
            label=MODEL_LABELS.get(model, model),
            hatch=HATCH.get(model, ""),
            edgecolor="white",
            linewidth=0.8,
        )
        # value labels
        for bar in bars:
            h = bar.get_height()
            if fmt_pct:
                label = f"{h:.0f}%"
            else:
                label = f"{h:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (ylim[1] * 0.01 if ylim else 0.5),
                label,
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot.index, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if fmt_pct:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Generator model", framealpha=0.9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


# ── Individual figures ─────────────────────────────────────────────────────────

def fig_structural_validity(df: pd.DataFrame):
    """Bar chart: % of structurally valid FHIR bundles (res_struct=True)."""
    pivot = (
        df.groupby(["pipeline_label", "model"])["res_struct"]
        .mean()
        .mul(100)
        .unstack("model")
        .reindex([PIPELINE_LABELS[p] for p in PIPELINES])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[p] for p in PIPELINES]

    n_pipelines = len(pivot)
    n_models    = len(pivot.columns)
    x           = np.arange(n_pipelines)
    width       = 0.35

    for i, model in enumerate(pivot.columns):
        offset = (i - (n_models - 1) / 2) * width
        for j, (pl, val) in enumerate(zip(pivot.index, pivot[model])):
            p_key = [k for k, v in PIPELINE_LABELS.items() if v == pl][0]
            ax.bar(
                x[j] + offset, val,
                width=width * 0.9,
                color=PALETTE[p_key],
                hatch=HATCH.get(model, ""),
                edgecolor="white", linewidth=0.8,
                label=f"{MODEL_LABELS.get(model, model)}" if j == 0 else "_",
            )
            ax.text(
                x[j] + offset, val + 1,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("Structurally valid scenarios (%)")
    ax.set_title("FHIR Bundle Structural Validity by Pipeline and Generator Model")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Generator model", ncol=2, framealpha=0.9, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # reference line at 100%
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    plt.tight_layout()
    out = OUT_DIR / "val_01_structural_validity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_code_errors_mean(df: pd.DataFrame):
    """Bar chart: mean number of code errors per scenario."""
    pivot = (
        df.groupby(["pipeline_label", "model"])["code_errors"]
        .mean()
        .unstack("model")
        .reindex([PIPELINE_LABELS[p] for p in PIPELINES])
    )
    std_pivot = (
        df.groupby(["pipeline_label", "model"])["code_errors"]
        .std()
        .unstack("model")
        .reindex([PIPELINE_LABELS[p] for p in PIPELINES])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    n_models = len(pivot.columns)
    x        = np.arange(len(pivot))
    width    = 0.35

    for i, model in enumerate(pivot.columns):
        offset = (i - (n_models - 1) / 2) * width
        for j, pl in enumerate(pivot.index):
            p_key = [k for k, v in PIPELINE_LABELS.items() if v == pl][0]
            ax.bar(
                x[j] + offset, pivot[model].iloc[j],
                width=width * 0.9,
                yerr=std_pivot[model].iloc[j],
                capsize=4,
                color=PALETTE[p_key],
                hatch=HATCH.get(model, ""),
                edgecolor="white", linewidth=0.8,
                error_kw={"elinewidth": 1.2, "ecolor": "#555"},
                label=f"{MODEL_LABELS.get(model, model)}" if j == 0 else "_",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("Mean code errors per scenario (±1 SD)")
    ax.set_title("FHIR Code Errors per Scenario by Pipeline and Generator Model")
    ax.legend(title="Generator model", ncol=2, framealpha=0.9, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "val_02_code_errors_mean.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_zero_code_errors(df: pd.DataFrame):
    """Bar chart: % of scenarios with zero code errors."""
    df2 = df.copy()
    df2["zero_errors"] = df2["code_errors"] == 0

    pivot = (
        df2.groupby(["pipeline_label", "model"])["zero_errors"]
        .mean()
        .mul(100)
        .unstack("model")
        .reindex([PIPELINE_LABELS[p] for p in PIPELINES])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x     = np.arange(len(pivot))
    width = 0.35

    for i, model in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2) * width
        for j, pl in enumerate(pivot.index):
            p_key = [k for k, v in PIPELINE_LABELS.items() if v == pl][0]
            ax.bar(
                x[j] + offset, pivot[model].iloc[j],
                width=width * 0.9,
                color=PALETTE[p_key],
                hatch=HATCH.get(model, ""),
                edgecolor="white", linewidth=0.8,
                label=f"{MODEL_LABELS.get(model, model)}" if j == 0 else "_",
            )
            ax.text(
                x[j] + offset, pivot[model].iloc[j] + 1,
                f"{pivot[model].iloc[j]:.0f}%",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("Scenarios with zero code errors (%)")
    ax.set_title("FHIR Code Correctness — Scenarios with No Code Errors")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Generator model", ncol=2, framealpha=0.9, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    plt.tight_layout()
    out = OUT_DIR / "val_03_zero_code_errors.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_error_type_breakdown(df: pd.DataFrame):
    """Stacked bar: breakdown of error categories by pipeline."""
    # Flatten all error messages
    rows = []
    for _, row in df.iterrows():
        for err in row["errors_raw"]:
            rows.append({
                "pipeline_label": row["pipeline_label"],
                "category": _classify_error(err),
            })

    if not rows:
        print("  SKIP val_04 — no errors found")
        return

    err_df   = pd.DataFrame(rows)
    counts   = (
        err_df.groupby(["pipeline_label", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex([PIPELINE_LABELS[p] for p in PIPELINES])
    )
    # normalise to % of total errors per pipeline
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)

    cat_colors = sns.color_palette("tab10", n_colors=len(pct.columns))

    fig, ax = plt.subplots(figsize=(9, 5))
    pct.plot(kind="bar", stacked=True, ax=ax,
             color=cat_colors, edgecolor="white", linewidth=0.5, width=0.6)

    ax.set_ylabel("Share of all errors (%)")
    ax.set_xlabel("")
    ax.set_title("FHIR Validation Error Categories by Pipeline")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Error category", bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.9, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # annotate total error count per pipeline
    for i, (pl, row) in enumerate(counts.iterrows()):
        total = row.sum()
        ax.text(i, 102, f"n={total}", ha="center", fontsize=9, color="#333")

    plt.tight_layout()
    out = OUT_DIR / "val_04_error_type_breakdown.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading validation data...")
    df = load_validation()
    print(f"  Loaded {len(df)} records from {df['pipeline'].nunique()} pipelines")
    print(f"  Pipelines: {df['pipeline'].value_counts().to_dict()}")
    print()

    print("Generating validation figures...")
    fig_structural_validity(df)
    fig_code_errors_mean(df)
    fig_zero_code_errors(df)
    fig_error_type_breakdown(df)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
