"""
export_summary.py
=================
Exports all evaluation results as a structured JSON summary and a plain-text
narrative outline — ready for an LLM to describe or discuss.

Outputs (vis_res/):
  results_summary.json   — structured data (all metrics, all pipelines/models/ICD codes)
  results_narrative.txt  — plain-text numbered outline an LLM can narrate from
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL  = ROOT / "evaluace" / "results"
SCEN  = ROOT / "scenarios"
OUT   = Path(__file__).resolve().parent

PIPELINES = ["baseline", "tools_agent", "rag", "multiagent"]
PIPELINE_LABELS = {
    "baseline":    "Baseline",
    "tools_agent": "Tools agent",
    "rag":         "RAG",
    "multiagent":  "Multi-agent",
}
GEN_MODELS = ["openai", "anthropic"]
GEN_MODEL_LABELS = {
    "openai":    "GPT-5.4",
    "anthropic": "Claude Sonnet 4.6",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_validation():
    rows = []
    for pipeline in PIPELINES:
        path = SCEN / pipeline / "validation.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["pipeline"] = pipeline
                    rows.append(rec)
    return rows


def load_evaluations():
    rows = []
    for pipeline in PIPELINES:
        for gen in GEN_MODELS:
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


# ── Aggregation helpers ───────────────────────────────────────────────────────

def pct(values: list[bool | float]) -> float:
    if not values:
        return 0.0
    return round(mean(float(v) for v in values) * 100, 1)


def avg(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def sd(values: list[float]) -> float:
    return round(stdev(values), 4) if len(values) > 1 else 0.0


def _diag_correct(rec) -> bool:
    return bool(rec.get("diagnosis", {}).get("correct", False))


def _diag_correct_1(rec) -> bool:
    return rec.get("diagnosis", {}).get("correct_attempt") == 1


def _symptom_cov(rec) -> float:
    return float(rec.get("symptoms", {}).get("symptom_coverage", 0.0))


def _quality_ratio(rec) -> float:
    q = rec.get("quality", {})
    mx = q.get("max_score", 14) or 14
    return q.get("score", 0) / mx


def _gate_passed(rec) -> bool:
    return bool(rec.get("gate", {}).get("passed", False))


# ── Build summary dict ────────────────────────────────────────────────────────

def build_summary(val_rows, eval_rows) -> dict:
    summary: dict = {
        "meta": {
            "pipelines":     PIPELINES,
            "generator_models": {k: v for k, v in GEN_MODEL_LABELS.items()},
            "total_scenarios_generated": len(val_rows),
            "total_evaluation_records":  len(eval_rows),
        },
        "fhir_validation": {},
        "evaluation": {
            "by_pipeline":       {},
            "by_generator_model": {},
            "by_icd_code":       {},
            "gate_questions":    {},
            "quality_questions": {},
        },
    }

    # ── FHIR validation ───────────────────────────────────────────────────────
    for pipeline in PIPELINES:
        rows = [r for r in val_rows if r["pipeline"] == pipeline]
        for gen in GEN_MODELS:
            sub = [r for r in rows if r["model"] == gen]
            key = f"{PIPELINE_LABELS[pipeline]} / {GEN_MODEL_LABELS[gen]}"
            summary["fhir_validation"][key] = {
                "n":                   len(sub),
                "structural_validity_pct": pct([r["res_struct"] for r in sub]),
                "zero_code_errors_pct":    pct([r["code_errors"] == 0 for r in sub]),
                "mean_code_errors":        avg([r["code_errors"] for r in sub]),
                "sd_code_errors":          sd([r["code_errors"] for r in sub]),
                "total_code_errors":       sum(r["code_errors"] for r in sub),
            }

    # ── Evaluation by pipeline × generator ───────────────────────────────────
    for pipeline in PIPELINES:
        p_label = PIPELINE_LABELS[pipeline]
        summary["evaluation"]["by_pipeline"][p_label] = {}
        for gen in GEN_MODELS:
            g_label = GEN_MODEL_LABELS[gen]
            sub = [r for r in eval_rows
                   if r["pipeline"] == pipeline and r["gen_model"] == gen]
            if not sub:
                continue
            summary["evaluation"]["by_pipeline"][p_label][g_label] = {
                "n":                      len(sub),
                "evaluator_model":        sub[0].get("model", "?"),
                "gate_pass_rate_pct":     pct([_gate_passed(r) for r in sub]),
                "symptom_coverage_mean":  avg([_symptom_cov(r) for r in sub]),
                "symptom_coverage_sd":    sd([_symptom_cov(r) for r in sub]),
                "quality_ratio_mean":     avg([_quality_ratio(r) for r in sub]),
                "quality_ratio_sd":       sd([_quality_ratio(r) for r in sub]),
                "diagnosis_correct_1st_pct":  pct([_diag_correct_1(r) for r in sub]),
                "diagnosis_correct_any_pct":  pct([_diag_correct(r) for r in sub]),
            }

    # ── Evaluation by generator model (across pipelines) ─────────────────────
    for gen in GEN_MODELS:
        g_label = GEN_MODEL_LABELS[gen]
        sub = [r for r in eval_rows if r["gen_model"] == gen]
        summary["evaluation"]["by_generator_model"][g_label] = {
            "n":                      len(sub),
            "gate_pass_rate_pct":     pct([_gate_passed(r) for r in sub]),
            "symptom_coverage_mean":  avg([_symptom_cov(r) for r in sub]),
            "quality_ratio_mean":     avg([_quality_ratio(r) for r in sub]),
            "diagnosis_correct_1st_pct": pct([_diag_correct_1(r) for r in sub]),
            "diagnosis_correct_any_pct": pct([_diag_correct(r) for r in sub]),
        }

    # ── Per ICD code (across all pipelines+models) ────────────────────────────
    icd_groups: dict[str, list] = {}
    for r in eval_rows:
        icd = r.get("icd_code", "?")
        icd_groups.setdefault(icd, []).append(r)

    for icd, recs in sorted(icd_groups.items()):
        summary["evaluation"]["by_icd_code"][icd] = {
            "diagnosis_name":             recs[0].get("diagnosis_name", ""),
            "n":                          len(recs),
            "gate_pass_rate_pct":         pct([_gate_passed(r) for r in recs]),
            "symptom_coverage_mean":      avg([_symptom_cov(r) for r in recs]),
            "quality_ratio_mean":         avg([_quality_ratio(r) for r in recs]),
            "diagnosis_correct_any_pct":  pct([_diag_correct(r) for r in recs]),
            "diagnosis_correct_1st_pct":  pct([_diag_correct_1(r) for r in recs]),
            # per pipeline
            "per_pipeline": {
                PIPELINE_LABELS[p]: {
                    "diagnosis_correct_any_pct": pct(
                        [_diag_correct(r) for r in recs if r["pipeline"] == p]
                    )
                }
                for p in PIPELINES
                if any(r["pipeline"] == p for r in recs)
            },
        }

    # ── Gate and quality question stats ──────────────────────────────────────
    from vis_res.fig_questions import question_stats
    q_stats = question_stats(eval_rows)
    summary["evaluation"]["gate_questions"]    = q_stats["gate_failure_rate_pct"]
    summary["evaluation"]["quality_questions"] = q_stats["quality_no_rate_pct"]

    return summary


# ── Plain-text narrative outline ──────────────────────────────────────────────

def build_narrative(summary: dict) -> str:
    lines: list[str] = []
    W = 72

    def h1(t): lines.append("=" * W); lines.append(t.upper()); lines.append("=" * W)
    def h2(t): lines.append(""); lines.append(t); lines.append("-" * len(t))
    def li(t): lines.append(f"  * {t}")
    def br():  lines.append("")

    h1("Virtual-patient pipeline — evaluation summary")
    br()
    m = summary["meta"]
    li(f"Pipelines evaluated:    {', '.join(m['pipelines'])}")
    li(f"Generator models:       {', '.join(m['generator_models'].values())}")
    li(f"Scenarios generated:    {m['total_scenarios_generated']}")
    li(f"Evaluation records:     {m['total_evaluation_records']}")

    # ── FHIR validation ───────────────────────────────────────────────────────
    h1("1. FHIR Structural Validation")
    lines.append(
        "Each generated scenario was validated against the FHIR R4 specification.\n"
        "Two metrics are reported: structural validity (res_struct=true) and the\n"
        "number of code-level errors (ATC, SNOMED CT, LOINC)."
    )
    br()
    for key, v in summary["fhir_validation"].items():
        h2(key)
        li(f"Scenarios evaluated:          {v['n']}")
        li(f"Structurally valid:           {v['structural_validity_pct']:.1f}%")
        li(f"Scenarios with 0 code errors: {v['zero_code_errors_pct']:.1f}%")
        li(f"Mean code errors / scenario:  {v['mean_code_errors']:.2f}  (SD {v['sd_code_errors']:.2f})")
        li(f"Total code errors:            {v['total_code_errors']}")

    # ── Evaluation by pipeline ────────────────────────────────────────────────
    h1("2. Evaluation Results by Pipeline and Generator Model")
    lines.append(
        "Each scenario was evaluated by a different LLM than the one that generated it\n"
        "(cross-validation). Metrics: gate pass rate, symptom coverage, quality ratio,\n"
        "and diagnosis accuracy (first attempt / any attempt up to 3)."
    )
    for p_label, by_gen in summary["evaluation"]["by_pipeline"].items():
        h2(p_label)
        for g_label, v in by_gen.items():
            lines.append(f"  Generator: {g_label}  |  Evaluator: {v['evaluator_model']}  |  n={v['n']}")
            li(f"Gate pass rate:            {v['gate_pass_rate_pct']:.1f}%")
            li(f"Symptom coverage:          {v['symptom_coverage_mean']*100:.1f}%  (SD {v['symptom_coverage_sd']*100:.1f}%)")
            li(f"Quality ratio:             {v['quality_ratio_mean']*100:.1f}%  (SD {v['quality_ratio_sd']*100:.1f}%)")
            li(f"Diagnosis accuracy (1st):  {v['diagnosis_correct_1st_pct']:.1f}%")
            li(f"Diagnosis accuracy (any):  {v['diagnosis_correct_any_pct']:.1f}%")
            br()

    # ── Model comparison ──────────────────────────────────────────────────────
    h1("3. Generator Model Comparison (across all pipelines)")
    for g_label, v in summary["evaluation"]["by_generator_model"].items():
        h2(g_label)
        li(f"n:                         {v['n']}")
        li(f"Gate pass rate:            {v['gate_pass_rate_pct']:.1f}%")
        li(f"Symptom coverage:          {v['symptom_coverage_mean']*100:.1f}%")
        li(f"Quality ratio:             {v['quality_ratio_mean']*100:.1f}%")
        li(f"Diagnosis accuracy (1st):  {v['diagnosis_correct_1st_pct']:.1f}%")
        li(f"Diagnosis accuracy (any):  {v['diagnosis_correct_any_pct']:.1f}%")

    # ── Gate questions ────────────────────────────────────────────────────────
    h1("4. Gate Questions — Failure Rate (across all pipelines and models)")
    lines.append(
        "A gate question 'fails' when the answer matches the rejection trigger\n"
        "(e.g. RF1='YES' means 'contradicting finding present' → rejection).\n"
        "Sorted by failure rate descending."
    )
    br()
    gq = summary["evaluation"]["gate_questions"]
    sorted_gq = sorted(gq.items(), key=lambda kv: kv[1]["failure_rate_pct"], reverse=True)
    for qid, v in sorted_gq:
        flag = "(reject on YES)" if v["reject_on"] == "YES" else "(reject on NO)"
        lines.append(f"  {qid:5s}  {v['failure_rate_pct']:5.1f}%  {flag}  {v['text']}")

    # ── Quality questions ─────────────────────────────────────────────────────
    h1("5. Quality Questions — NO Answer Rate (across all pipelines and models)")
    lines.append(
        "A high NO rate means the scenario often failed to satisfy that criterion.\n"
        "Sorted by NO rate descending."
    )
    br()
    qq = summary["evaluation"]["quality_questions"]
    sorted_qq = sorted(qq.items(), key=lambda kv: kv[1]["no_rate_pct"], reverse=True)
    for qid, v in sorted_qq:
        flag = ("!! frequent" if v["no_rate_pct"] > 40
                else ("! occasional" if v["no_rate_pct"] > 20 else "  ok"))
        lines.append(f"  {qid:5s}  {v['no_rate_pct']:5.1f}%  {flag}  {v['text']}")

    # ── Per ICD code ──────────────────────────────────────────────────────────
    h1("6. Per-Diagnosis Results")
    lines.append(
        "Results aggregated over all pipelines and generator models.\n"
        "Diagnoses sorted by overall diagnosis accuracy (descending)."
    )
    br()
    by_icd = summary["evaluation"]["by_icd_code"]
    sorted_icd = sorted(
        by_icd.items(),
        key=lambda kv: kv[1]["diagnosis_correct_any_pct"],
        reverse=True,
    )
    for icd, v in sorted_icd:
        lines.append(
            f"  {icd:10s} {v['diagnosis_name'][:50]}"
        )
        li(f"n={v['n']}  gate={v['gate_pass_rate_pct']:.0f}%  "
           f"symptoms={v['symptom_coverage_mean']*100:.0f}%  "
           f"quality={v['quality_ratio_mean']*100:.0f}%  "
           f"diagnosis={v['diagnosis_correct_any_pct']:.0f}% (1st: {v['diagnosis_correct_1st_pct']:.0f}%)")
        # per pipeline breakdown
        pp = v["per_pipeline"]
        breakdown = "  |  ".join(
            f"{pl}: {vv['diagnosis_correct_any_pct']:.0f}%"
            for pl, vv in pp.items()
        )
        lines.append(f"    [{breakdown}]")

    br()
    lines.append("END OF SUMMARY")
    lines.append("=" * W)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    val_rows  = load_validation()
    eval_rows = load_evaluations()
    print(f"  Validation records: {len(val_rows)}")
    print(f"  Evaluation records: {len(eval_rows)}")

    print("Building summary...")
    summary = build_summary(val_rows, eval_rows)

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = OUT / "results_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Saved {json_path.name}")

    # ── Plain text ────────────────────────────────────────────────────────────
    narrative = build_narrative(summary)
    txt_path = OUT / "results_narrative.txt"
    txt_path.write_text(narrative, encoding="utf-8")
    print(f"  Saved {txt_path.name}")

    # Print preview
    print()
    print(narrative[:2000])
    if len(narrative) > 2000:
        print(f"\n  ... ({len(narrative)} chars total, see {txt_path.name})")


if __name__ == "__main__":
    main()
