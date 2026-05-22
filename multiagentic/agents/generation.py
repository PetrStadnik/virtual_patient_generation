"""
Generation agent — thin wrapper around the existing baseline_script.generate_patient.

Accepts the same arguments and returns a plain dict (PatientOutput.model_dump()).
Separating this into its own module keeps the LangGraph nodes decoupled from the
low-level retry / model-selection logic that lives in baseline_script.
"""
from __future__ import annotations
from rag.generate.run import generate_patient_rag_with_tools
from src.LLM_generation.llm_io import PatientOutput


def generate_scenario(model: str, icd_code: str, diagnosis_name: str, gender: str, age: int) -> PatientOutput:
    """Return a PatientOutput dict for the given clinical parameters."""
    return generate_patient_rag_with_tools(
        model=model,
        icd_code=icd_code,
        diagnosis_name=diagnosis_name,
        gender=gender,
        age=age,
        dump=False
    )
