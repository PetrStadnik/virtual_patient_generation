"""
Fully-agentic virtual patient scenario generator.

The pipeline:
  generate -> evaluate -> (fix -> evaluate)* -> save

Public API:
  run(...)                — returns the full final LangGraph state dict
  generate_patient(...)   — same signature as baseline_script.generate_patient;
                            returns PatientOutput.model_dump() (or PatientOutput
                            when dump=False).  Drop-in replacement in main.py.
"""
from __future__ import annotations
from multiagentic.graph import PipelineState, build_pipeline
from src.LLM_generation.llm_io import PatientOutput


# ── Internal pipeline runner ──────────────────────────────────────────────────

def run(
    icd_code: str,
    diagnosis_name: str,
    model: str,
    age: int,
    gender: str,
    patient_name: str = "",
    patient_country: str = "CZ",
    max_iterations: int = 5,
    min_symptom_coverage: float = 0.70,
    min_quality_ratio: float = 0.77,
) -> dict:
    """Run the full agentic pipeline and return the final LangGraph state dict.

    The final state contains a 'scenario' key with the best PatientOutput found.
    """
    pipeline = build_pipeline()

    initial_state: PipelineState = {
        "icd_code": icd_code,
        "diagnosis_name": diagnosis_name,
        "gender": gender,
        "age": age,
        "patient_name": patient_name,
        "patient_country": patient_country,
        "model": model,
        "max_iterations": max_iterations,
        "min_symptom_posterior": min_symptom_coverage,
        "min_quality_ratio": min_quality_ratio,
        "scenario": None,
        "iteration": 0,
        "is_done": False,
        "done_reason": "",
        "gate_passed": False,
        "gate_reasons": [],
        "symptom_posterior": 0.0,
        "absent_key_symptoms": [],
        "quality_score": 0,
        "quality_max": 13,
        "quality_no_answers": [],
        "history": [],
        "output_path": "",
    }

    return pipeline.invoke(initial_state)


# ── Public generate_patient — same signature as baseline_script ───────────────

def generate_patient(
    model: str,
    icd_code: str,
    diagnosis_name: str,
    gender: str,
    age: int,
    patient_name: str,
    country: str = "CZ",
    dump: bool = True,
) -> dict | PatientOutput:
    """Generate a virtual patient using the full agentic pipeline (generate + evaluate + fix loop).

    Drop-in replacement for baseline_script.generate_patient and
    agent_with_tools.generate_patient_with_tools in main.py.

    Args:
        model:          LLM model string (OpenAI or Anthropic).
        icd_code:       ICD-10 diagnosis code.
        diagnosis_name: Human-readable diagnosis name.
        gender:         "male" or "female".
        age:            Patient age in years.
        patient_name:   Patient name
        country:        Two-letter country code for name generation (default "CZ").
        dump:           If True return dict, if False return PatientOutput.

    Returns:
        PatientOutput.model_dump() dict (dump=True) or PatientOutput (dump=False).
    """
    final_state = run(
        icd_code=icd_code,
        diagnosis_name=diagnosis_name,
        model=model,
        age=age,
        gender=gender,
        patient_name= patient_name,
        patient_country=country,
    )

    scenario = final_state.get("scenario")
    if scenario is None:
        raise ValueError("multiagentic pipeline returned no scenario")

    if isinstance(scenario, PatientOutput):
        return scenario.model_dump() if dump else scenario

    # scenario is already a dict (e.g. from model_dump inside a node)
    validated = PatientOutput.model_validate(scenario)
    return validated.model_dump() if dump else validated

