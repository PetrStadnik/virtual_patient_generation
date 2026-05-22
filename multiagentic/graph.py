"""
LangGraph pipeline for the agentic virtual-patient scenario generator.

Graph topology:
                     ┌────────┐
              START ─► generate├──► evaluate ──┬─► save ──► END
                     └────────┘        ▲       │
                                       │  fix ◄─┘  (when not done)
                                       └──────────

Termination conditions (checked after every evaluate step):
  • All three quality gates pass, OR
  • max_iterations is reached (best-effort result is still saved)

State fields annotated with `Annotated[list, operator.add]` are *accumulated*
across iterations; all other fields are replaced on each update.
"""
from __future__ import annotations
from multiagentic.agents.generation import generate_scenario
import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, TypedDict
from src.LLM_generation.llm_io import FullPatient, PatientOutput
from langgraph.graph import END, START, StateGraph

from evaluace.agent1_gate import evaluate_gate
from evaluace.agent2_symptoms import evaluate_symptoms
from evaluace.agent3_quality import QUALITY_QUESTIONS, evaluate_quality


# ── State definition ──────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Fixed throughout the run ──────────────────────────────────────────
    icd_code: str
    diagnosis_name: str
    gender: str
    age: int
    patient_country: str
    patient_name: str
    model: str
    max_iterations: int
    min_symptom_posterior: float
    min_quality_ratio: float

    # ── Mutable: current scenario ─────────────────────────────────────────
    scenario: PatientOutput
    iteration: int
    is_done: bool
    done_reason: str

    # ── Mutable: latest evaluation snapshot (for fix agent) ───────────────
    gate_passed: bool
    gate_reasons: list[str]
    symptom_posterior: float
    absent_key_symptoms: list[str]
    quality_score: int
    quality_max: int
    quality_no_answers: list[str]  # "QID: question → NO: reasoning"

    # ── Accumulated across iterations ─────────────────────────────────────
    history: Annotated[list[dict], operator.add]

    # ── Set by save_node ──────────────────────────────────────────────────
    output_path: str


# ── Node: generate ────────────────────────────────────────────────────────────

def generate_node(state: PipelineState) -> dict:
    print(
        f"\n[GEN] Generating initial scenario — "
        f"{state['icd_code']} / {state['diagnosis_name']} "
        f"({state['gender']}, {state['age']} y)"
    )
    patient_output = generate_scenario(
        model=state["model"],
        icd_code=state["icd_code"],
        diagnosis_name=state["diagnosis_name"],
        gender=state["gender"],
        age=state["age"],
    )
    return {"scenario": patient_output}


# ── Node: evaluate ────────────────────────────────────────────────────────────

def evaluate_node(state: PipelineState) -> dict:
    iteration = state["iteration"] + 1
    print(f"\n[EVAL] Iteration {iteration}/{state['max_iterations']}")

    scenario = PatientOutput.model_validate(state["scenario"])
    eval_input = FullPatient(
        patient=scenario,
        icd_code=state["icd_code"],
        diagnosis_name=state["diagnosis_name"],
        gender=state["gender"],
        age=state["age"],
        name=state["patient_name"]
    )

    # ── 1. Gate ───────────────────────────────────────────────────────────
    print("  [1/3] Gate evaluation...")
    gate = evaluate_gate(state["model"], eval_input)

    # ── 2. Symptom (Bayesian) ─────────────────────────────────────────────
    print("  [2/3] Symptom evaluation...")
    symptoms = evaluate_symptoms(state["model"], eval_input)

    # ── 3. Quality ────────────────────────────────────────────────────────
    print("  [3/3] Quality evaluation...")
    quality = evaluate_quality(state["model"], eval_input)

    # ── Derive helper structures ──────────────────────────────────────────
    # symptoms.symptoms_checked: list[SymptomPresence] with .symptom / .present
    absent_key = [
        p.symptom for p in symptoms.symptoms_checked if not p.present
    ]

    q_text_map = {qid: qtxt for qid, qtxt in QUALITY_QUESTIONS}
    quality_nos = [
        f"{a.question_id}: {q_text_map.get(a.question_id, '?')} -> NO: {a.reasoning}"
        for a in quality.answers
        if a.answer == "NO"
    ]

    # ── Evaluate thresholds ───────────────────────────────────────────────
    gate_ok    = gate.passed
    # symptom_coverage: fraction of SNOMED symptoms present (0.0–1.0)
    symptom_ok = symptoms.symptom_coverage >= state["min_symptom_posterior"]
    quality_ok = (
        (quality.score / quality.max_score) >= state["min_quality_ratio"]
        if quality.max_score > 0
        else True
    )
    is_perfect = gate_ok and symptom_ok and quality_ok

    gate_sym = "OK" if gate_ok else "FAIL"
    sym_sym  = "OK" if symptom_ok else "FAIL"
    qual_sym = "OK" if quality_ok else "FAIL"
    print(
        f"  Gate {gate_sym} | "
        f"Symptoms {symptoms.symptom_coverage:.1%} {sym_sym} | "
        f"Quality {quality.score}/{quality.max_score} {qual_sym} | "
        f"Perfect: {'YES' if is_perfect else 'NO'}"
    )

    iter_record = {
        "iteration": iteration,
        "gate_passed": gate_ok,
        "gate_rejection_reasons": gate.rejection_reasons,
        "symptom_coverage": symptoms.symptom_coverage,
        "absent_key_symptoms": absent_key,
        "quality_score": quality.score,
        "quality_max_score": quality.max_score,
        "quality_no_count": len(quality_nos),
        "is_perfect": is_perfect,
    }

    is_done = is_perfect or iteration >= state["max_iterations"]
    if is_perfect:
        done_reason = "All evaluations passed"
    elif iteration >= state["max_iterations"]:
        done_reason = f"Max iterations ({state['max_iterations']}) reached"
    else:
        done_reason = ""

    return {
        "iteration": iteration,
        "gate_passed": gate_ok,
        "gate_reasons": gate.rejection_reasons,
        "symptom_posterior": symptoms.symptom_coverage,   # kept for fix agent compat
        "absent_key_symptoms": absent_key,
        "quality_score": quality.score,
        "quality_max": quality.max_score,
        "quality_no_answers": quality_nos,
        "is_done": is_done,
        "done_reason": done_reason,
        # Accumulated by LangGraph via operator.add
        "history": [iter_record],
    }


# ── Conditional edge ──────────────────────────────────────────────────────────

def route_after_evaluate(state: PipelineState) -> str:
    return "save" if state["is_done"] else "fix"


# ── Node: fix ─────────────────────────────────────────────────────────────────

def fix_node(state: PipelineState) -> dict:
    from multiagentic.agents.fix import fix_scenario

    print(f"\n[FIX] Improving scenario (iteration {state['iteration']})...")
    scenario = state["scenario"]
    scenario_dict = (
        scenario.model_dump() if isinstance(scenario, PatientOutput) else scenario
    )
    improved = fix_scenario(
        model=state["model"],
        icd_code=state["icd_code"],
        diagnosis_name=state["diagnosis_name"],
        gender=state["gender"],
        age=state["age"],
        current_scenario=scenario_dict,
        gate_reasons=state["gate_reasons"],
        symptom_posterior=state["symptom_posterior"],
        absent_key_symptoms=state["absent_key_symptoms"],
        quality_score=state["quality_score"],
        quality_max=state["quality_max"],
        quality_no_answers=state["quality_no_answers"],
    )
    return {"scenario": improved}


# ── Node: save ────────────────────────────────────────────────────────────────

def finish_node(state: PipelineState) -> dict:
    last = state["history"][-1] if state["history"] else {}
    result = {
        "metadata": {
            "icd_code": state["icd_code"],
            "diagnosis_name": state["diagnosis_name"],
            "gender": state["gender"],
            "age": state["age"],
            "patient_country": state["patient_country"],
            "model": state["model"],
        },
        "scenario": state["scenario"],
        "total_iterations": state["iteration"],
        "done_reason": state["done_reason"],
        "is_perfect": last.get("is_perfect", False),
        "final_evaluation": {
            "gate_passed": state["gate_passed"],
            "symptom_posterior": state["symptom_posterior"],
            "quality_score": state["quality_score"],
            "quality_max": state["quality_max"],
        },
        "iteration_history": state["history"],
    }
    print(f"\n[FINISHED]")
    print(f"  Iterations: {state['iteration']}  |  {state['done_reason']}")
    gate_sym = "OK" if state["gate_passed"] else "FAIL"
    print(
        f"  Gate {gate_sym}  |  "
        f"Symptoms {state['symptom_posterior']:.1%}  |  "
        f"Quality {state['quality_score']}/{state['quality_max']}"
    )

    # Return the final scenario so callers can access state["scenario"]
    return {"scenario": result["scenario"]}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_pipeline():
    """Compile and return the LangGraph pipeline."""
    builder = StateGraph(PipelineState)

    builder.add_node("generate", generate_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("fix", fix_node)
    builder.add_node("save", finish_node)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"fix": "fix", "save": "save"},
    )
    builder.add_edge("fix", "evaluate")
    builder.add_edge("save", END)

    return builder.compile()
