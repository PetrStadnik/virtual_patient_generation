"""
fig_questions.py
================
Figures analysing gate and quality question failure rates.

Source: evaluace/results/{pipeline}_{generator}.jsonl

Figures produced (saved to vis_res/figures/):
  q_01_gate_failure_rate.pdf    — which gate questions most often caused rejection
  q_02_quality_no_rate.pdf      — which quality questions were most often answered NO
  q_03_gate_by_pipeline.pdf     — gate question failure heatmap: question × pipeline
  q_04_quality_by_pipeline.pdf  — quality question NO-rate heatmap: question × pipeline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = Path(__file__).resolve().parent.parent
EVAL    = ROOT / "evaluace" / "results"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

PIPELINES = ["baseline", "tools_agent", "rag", "multiagent"]
PIPELINE_LABELS = {
    "baseline":    "Baseline",
    "tools_agent": "Tools agent",
    "rag":         "RAG",
    "multiagent":  "Multi-agent",
}

# Gate question texts and which answer triggers rejection
GATE_QUESTIONS = [
    ("REQ1", "Is this diagnosis plausible for this gender?",                                  "NO"),
    ("REQ2", "Is this diagnosis plausible for this age group?",                               "NO"),
    ("REQ3", "Are the key hallmark symptoms of the diagnosis present?",                       "NO"),
    ("REQ4", "Are the minimum required diagnostic findings present?",                         "NO"),
    ("RF1",  "Is there a finding that clearly contradicts this diagnosis?",                   "YES"),
    ("RF2",  "Is there another diagnosis that explains the symptoms better?",                 "YES"),
    ("RF3",  "Could symptoms be explained as medication side effects instead?",               "YES"),
    ("RF4",  "Does the patient possess required anatomical structures?",                      "NO"),
    ("RF5",  "Are all test results consistent with expected findings?",                       "NO"),
    ("RF6",  "Is there a test result that is impossible or highly atypical?",                 "YES"),
    ("RF7",  "Is any test result physiologically implausible?",                               "YES"),
    ("RF8",  "Are all numeric test results within physiologically possible ranges?",          "NO"),
    ("SAF1", "Are prescribed medications contraindicated given allergies?",                   "YES"),
    ("SAF2", "Are medication doses within clinically accepted ranges?",                       "NO"),
]
GATE_REJECT_ON = {qid: reject_on for qid, _, reject_on in GATE_QUESTIONS}
GATE_LABELS    = {qid: txt for qid, txt, _ in GATE_QUESTIONS}

# Quality question texts (SUP1-SUP14); NO = failure
QUALITY_QUESTIONS = [
    ("SUP1",  "Are the most common symptoms present?"),
    ("SUP2",  "Does the symptom pattern match the typical presentation?"),
    ("SUP3",  "Does the medical history increase the risk of this disease?"),
    ("SUP4",  "Has the patient experienced similar episodes previously?"),
    ("SUP5",  "Is there family history of the same disease?"),
    ("SUP6",  "Is there family history of related disorders?"),
    ("SUP7",  "Does the patient have major known risk factors?"),
    ("SUP8",  "Are there comorbidities that predispose to this disease?"),
    ("SUP9",  "Does lifestyle include behaviours that trigger this disease?"),
    ("SUP10", "Does occupation expose the patient to risk factors?"),
    ("SUP11", "Do hobbies expose the patient to triggers?"),
    ("SUP12", "Could current medications contribute to the disease?"),
    ("SUP13", "Is the onset and progression temporally consistent?"),
    ("SUP14", "Are all medications unrelated to treating the primary diagnosis?"),
]
QUALITY_LABELS = {qid: txt for qid, txt in QUALITY_QUESTIONS}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_evaluations() -> list[dict]:
    rows = []
    for pipeline in PIPELINES:
        for gen in ["openai", "anthropic"]:
            path = EVAL / f"{pipeline}_{gen}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rec["pipeline"]  = pipeline
                        rec["gen_model"] = gen
                        rows.append(rec)
    return rows


def build_gate_df(records: list[dict]) -> pd.DataFrame:
    """One row per (record × gate question) with a 'failed' flag."""
    rows = []
    for rec in records:
        for ans in rec.get("gate", {}).get("answers", []):
            qid  = ans["question_id"]
            ans_ = ans["answer"]
            reject_on = GATE_REJECT_ON.get(qid, "NO")
            failed = (ans_ == reject_on)           # gave the "wrong" answer
            rows.append({
                "pipeline":      rec["pipeline"],
                "gen_model":     rec["gen_model"],
                "icd_code":      rec.get("icd_code", ""),
                "question_id":   qid,
                "answer":        ans_,
                "failed":        failed,            # True = triggered or would trigger rejection
                "gate_passed":   rec.get("gate", {}).get("passed", True),
            })
    return pd.DataFrame(rows)


def build_quality_df(records: list[dict]) -> pd.DataFrame:
    """One row per (record × quality question) with a 'no' flag."""
    rows = []
    for rec in records:
        for ans in rec.get("quality", {}).get("answers", []):
            rows.append({
                "pipeline":    rec["pipeline"],
                "gen_model":   rec["gen_model"],
                "icd_code":    rec.get("icd_code", ""),
                "question_id": ans["question_id"],
                "answer":      ans["answer"],
                "no":          (ans["answer"] == "NO"),
            })
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_gate_failure_rate(gdf: pd.DataFrame):
    """Horizontal bar: % of evaluations where each gate question triggered a failure."""
    order = [q[0] for q in GATE_QUESTIONS]

    fail_rate = (
        gdf.groupby("question_id")["failed"]
        .mean()
        .mul(100)
        .reindex(order)
        .reset_index()
    )
    fail_rate["label"] = fail_rate["question_id"].map(
        lambda q: f"{q}  –  {GATE_LABELS.get(q, q)[:60]}"
    )
    fail_rate["reject_on"] = fail_rate["question_id"].map(GATE_REJECT_ON)
    fail_rate["color"] = fail_rate["reject_on"].map(
        {"YES": "#C44E52", "NO": "#4C72B0"}
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        fail_rate["label"], fail_rate["failed"],
        color=fail_rate["color"], edgecolor="white", height=0.7,
    )
    for bar, val in zip(bars, fail_rate["failed"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Failure rate (% of all evaluations where question triggered rejection)")
    ax.set_title("Gate Questions — Failure Rate Across All Pipelines and Models")
    ax.set_xlim(0, max(fail_rate["failed"].max() + 8, 25))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#C44E52", label="Rejected when answer = YES"),
        Patch(color="#4C72B0", label="Rejected when answer = NO"),
    ]
    ax.legend(handles=legend, loc="lower right", framealpha=0.9)

    ax.invert_yaxis()
    plt.tight_layout()
    out = OUT_DIR / "q_01_gate_failure_rate.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_quality_no_rate(qdf: pd.DataFrame):
    """Horizontal bar: % of evaluations where each quality question was answered NO."""
    order = [q[0] for q in QUALITY_QUESTIONS]

    no_rate = (
        qdf.groupby("question_id")["no"]
        .mean()
        .mul(100)
        .reindex(order)
        .reset_index()
    )
    no_rate["label"] = no_rate["question_id"].map(
        lambda q: f"{q}  –  {QUALITY_LABELS.get(q, q)[:65]}"
    )

    # colour by severity: >40% red, 20-40% orange, <20% green
    no_rate["color"] = no_rate["no"].map(
        lambda v: "#C44E52" if v > 40 else ("#DD8452" if v > 20 else "#55A868")
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        no_rate["label"], no_rate["no"],
        color=no_rate["color"], edgecolor="white", height=0.7,
    )
    for bar, val in zip(bars, no_rate["no"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("NO rate (% of evaluations where question was answered NO)")
    ax.set_title("Quality Questions — NO Answer Rate Across All Pipelines and Models")
    ax.set_xlim(0, max(no_rate["no"].max() + 8, 25))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#C44E52", label="> 40% NO rate (frequent gap)"),
        Patch(color="#DD8452", label="20–40% NO rate (occasional gap)"),
        Patch(color="#55A868", label="< 20% NO rate (usually satisfied)"),
    ]
    ax.legend(handles=legend, loc="lower right", framealpha=0.9)

    ax.invert_yaxis()
    plt.tight_layout()
    out = OUT_DIR / "q_02_quality_no_rate.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_gate_by_pipeline(gdf: pd.DataFrame):
    """Heatmap: gate question failure rate per question × pipeline."""
    order = [q[0] for q in GATE_QUESTIONS]
    pl_order = [PIPELINE_LABELS[p] for p in PIPELINES]

    gdf2 = gdf.copy()
    gdf2["pipeline_label"] = gdf2["pipeline"].map(PIPELINE_LABELS)

    heat = (
        gdf2.groupby(["question_id", "pipeline_label"])["failed"]
        .mean()
        .mul(100)
        .unstack("pipeline_label")
        .reindex(index=order, columns=pl_order)
    )
    heat.index = heat.index.map(lambda q: f"{q}: {GATE_LABELS.get(q,'')[:45]}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heat, ax=ax,
        annot=True, fmt=".0f",
        cmap="YlOrRd",
        vmin=0, vmax=heat.values.max(),
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9},
        cbar_kws={"label": "Failure rate (%)", "shrink": 0.6},
    )
    ax.set_title("Gate Question Failure Rate (%) by Question and Pipeline")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "q_03_gate_by_pipeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


def fig_quality_by_pipeline(qdf: pd.DataFrame):
    """Heatmap: quality question NO-rate per question × pipeline."""
    order    = [q[0] for q in QUALITY_QUESTIONS]
    pl_order = [PIPELINE_LABELS[p] for p in PIPELINES]

    qdf2 = qdf.copy()
    qdf2["pipeline_label"] = qdf2["pipeline"].map(PIPELINE_LABELS)

    heat = (
        qdf2.groupby(["question_id", "pipeline_label"])["no"]
        .mean()
        .mul(100)
        .unstack("pipeline_label")
        .reindex(index=order, columns=pl_order)
    )
    heat.index = heat.index.map(lambda q: f"{q}: {QUALITY_LABELS.get(q,'')[:55]}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heat, ax=ax,
        annot=True, fmt=".0f",
        cmap="YlOrRd",
        vmin=0, vmax=heat.values.max(),
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9},
        cbar_kws={"label": "NO rate (%)", "shrink": 0.6},
    )
    ax.set_title("Quality Question NO Rate (%) by Question and Pipeline")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "q_04_quality_by_pipeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"  Saved {out.name}")
    plt.close(fig)


# ── Exported stats for summary ─────────────────────────────────────────────────

def question_stats(records: list[dict]) -> dict:
    """Return gate and quality question stats for export_summary."""
    gdf = build_gate_df(records)
    qdf = build_quality_df(records)

    gate_order = [q[0] for q in GATE_QUESTIONS]
    qual_order = [q[0] for q in QUALITY_QUESTIONS]

    gate_fail = (
        gdf.groupby("question_id")["failed"]
        .mean().mul(100).reindex(gate_order)
    )
    qual_no = (
        qdf.groupby("question_id")["no"]
        .mean().mul(100).reindex(qual_order)
    )

    return {
        "gate_failure_rate_pct": {
            qid: {
                "text": GATE_LABELS.get(qid, ""),
                "reject_on": GATE_REJECT_ON.get(qid, ""),
                "failure_rate_pct": round(float(gate_fail.get(qid, 0)), 1),
            }
            for qid in gate_order
        },
        "quality_no_rate_pct": {
            qid: {
                "text": QUALITY_LABELS.get(qid, ""),
                "no_rate_pct": round(float(qual_no.get(qid, 0)), 1),
            }
            for qid in qual_order
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading evaluation data...")
    records = load_evaluations()
    print(f"  Loaded {len(records)} records")

    gdf = build_gate_df(records)
    qdf = build_quality_df(records)
    print(f"  Gate question answers:    {len(gdf)}")
    print(f"  Quality question answers: {len(qdf)}")
    print()

    print("Generating question figures...")
    fig_gate_failure_rate(gdf)
    fig_quality_no_rate(qdf)
    fig_gate_by_pipeline(gdf)
    fig_quality_by_pipeline(qdf)
    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
