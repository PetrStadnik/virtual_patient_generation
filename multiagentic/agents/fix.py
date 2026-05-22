"""
Fix agent — receives a failing virtual-patient scenario together with structured
evaluation feedback from the three evaluation agents and produces an improved
PatientOutput that addresses every identified issue.

Improvement priority (applied in this order):
  1. Gate failures      — medical impossibilities / contradictions (must all be fixed)
  2. Missing symptoms   — SNOMED findings expected for this diagnosis that are absent
  3. Quality NO answers — richness / completeness gaps
"""
from __future__ import annotations

import random
import time

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

from src.LLM_generation.llm_io import PatientOutput


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """You are a medical scenario improvement specialist working with virtual patient \
records for clinical education and simulation.

You receive:
  • A complete patient scenario (PatientOutput) that failed quality evaluation
  • Structured feedback categorised into three tiers

Your task is to return a fully corrected PatientOutput that fixes every listed issue
while preserving clinical realism and internal consistency.

PRIORITY ORDER:
  1. Gate failures (CRITICAL) — fix ALL of them; these are medical impossibilities,
     anatomical contradictions, or severe safety issues.
  2. Missing key symptoms — add hallmark findings that are absent but expected
     for this diagnosis (positive likelihood ratio > 3).
  3. Quality gaps — enrich sections (history, lifestyle, comorbidities, etc.)
     that scored NO to improve clinical depth.

MANDATORY RULES:
  • Keep {patientName} placeholder in patient_description and patient_history.
  • Never invent physiologically impossible combinations.
  • LOINC codes must match format NNNNN-N (e.g. 35777-0).
  • ATC codes must match format A00AA00 (e.g. A02BC01).
  • ICD-10 codes must match format X00 or X00.0.
  • Medications MUST represent comorbidity/background drugs only — do NOT include
    any standard treatment medication for the primary diagnosis.
  • Make the minimum changes required — do not rewrite unaffected parts unnecessarily.
  • The narrative texts (patient_description, patient_history) should naturally reflect
    any clinical changes you make.
"""


# ── Public function ────────────────────────────────────────────────────────────

def fix_scenario(
    model: str,
    icd_code: str,
    diagnosis_name: str,
    gender: str,
    age: int,
    current_scenario: dict,
    gate_reasons: list[str],
    symptom_posterior: float,
    absent_key_symptoms: list[str],
    quality_score: int,
    quality_max: int,
    quality_no_answers: list[str],
) -> dict:
    """
    Return an improved PatientOutput dict addressing all evaluation failures.

    Args:
        model:                 LLM model identifier (OpenAI or Anthropic)
        icd_code:              ICD-10 code of the diagnosis
        diagnosis_name:        Human-readable diagnosis name
        gender / age:          Patient demographics
        current_scenario:      PatientOutput.model_dump() of the failing scenario
        gate_reasons:          Rejection strings from GateResult
        symptom_posterior:     Symptom coverage fraction from agent2 (0.0–1.0)
        absent_key_symptoms:   Symptom names absent from the patient record
        quality_score/max:     Numeric score from agent3
        quality_no_answers:    Formatted strings "QID: question → NO: reasoning"

    Returns:
        dict — PatientOutput.model_dump() of the improved scenario
    """
    llm = AnthropicModel(model) if "claude" in model else OpenAIModel(model)
    agent = Agent(llm, output_type=PatientOutput, system_prompt=_SYSTEM, retries=7)

    # ── Build structured issues block ────────────────────────────────────────
    parts: list[str] = []

    if gate_reasons:
        parts.append("### CRITICAL GATE FAILURES — fix ALL (non-negotiable):")
        for r in gate_reasons:
            parts.append(f"  • {r}")

    if absent_key_symptoms:
        parts.append(
            f"\n### MISSING SYMPTOMS  "
            f"(current symptom coverage: {symptom_posterior:.1%}, target >= 70%):"
        )
        parts.append("These SNOMED findings are expected for this diagnosis but absent from the record:")
        for s in absent_key_symptoms:
            parts.append(f"  - {s}")

    if quality_no_answers:
        ratio = quality_score / quality_max if quality_max else 0
        parts.append(
            f"\n### QUALITY GAPS  ({quality_score}/{quality_max} = {ratio:.0%}, target ≥ 77%):"
        )
        parts.append("Enrich the scenario so the following questions can be answered YES:")
        for q in quality_no_answers:
            parts.append(f"  • {q}")

    issues_block = "\n".join(parts) if parts else "No specific issues listed — general improvement pass."

    prompt = (
        f"Diagnosis: {diagnosis_name} ({icd_code})\n"
        f"Patient: {gender}, {age} years old\n\n"
        f"Current scenario:\n"
        f"{PatientOutput.model_validate(current_scenario).model_dump_json(indent=2)}\n\n"
        f"Evaluation feedback:\n{issues_block}\n\n"
        "Return the complete improved PatientOutput."
    )

    # ── Run with exponential back-off for overload errors ────────────────────
    for attempt in range(8):
        try:
            return agent.run_sync(prompt).output.model_dump()
        except ModelHTTPError as exc:
            if exc.status_code == 529:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"    [fix] Model overloaded — retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("fix_scenario: all retries exhausted (model overloaded)")
