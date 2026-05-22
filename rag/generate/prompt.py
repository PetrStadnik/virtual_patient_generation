"""
Prompt templates for retrieval-augmented virtual-patient scenario generation.

The LLM is asked to output strict PatientOutput JSON — the same schema used
by baseline_script.py and agent_with_tools.py — so all three pipelines produce
directly comparable outputs.

Text fields (patient_description, patient_history, diagnosis_description) embed
source citations in brackets, e.g. [PMID:12345678], so the diploma reviewer
can verify that every clinical claim is grounded in the retrieved evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.LLM_generation.llm_io import PatientOutput  # noqa: E402

SYSTEM = """\
You are a clinical scenario writer for medical-education simulations.
Generate ONE realistic virtual patient in the JSON schema shown below.
Ground every clinical claim in the EVIDENCE passages provided by the user.

Rules:
1. In the free-text fields (patient_description, patient_history,
   diagnosis_description) append the source IDs in square brackets for every
   fact drawn from the evidence, e.g. "polyuria [PMID:12345678,
   MEDLINEPLUS:E11.9]". This lets the reviewer verify grounding.
2. Do NOT invent rare or unusual findings absent from the evidence.
   If the evidence is silent on a detail, use a clinically plausible default.
3. Use {patientName} as a placeholder wherever the patient's name would appear
   in patient_description and patient_history.
4. Output ONLY a single JSON object matching the schema exactly.
   No extra keys, no prose outside the JSON.
5. Medications MUST represent comorbidity/background drugs only — do NOT include
   any medication that is a standard treatment for the primary diagnosis.
6. Code format rules:
   - ATC code: A00AA00  (e.g. A02BC01)
   - LOINC:    NNNNN-N  (e.g. 29463-7)
   - SNOMED:   integer  (e.g. 73211009)
   - ICD-10:   A00 or A00.0
"""

# Schema derived directly from the Pydantic model — stays in sync automatically.
_SCHEMA: dict = PatientOutput.model_json_schema()


def build_user_message(
    icd_code: str,
    diagnosis_name: str,
    age: int,
    sex: str,
    country: str,
    evidence: list[tuple],  # list of (Doc, float) from Retriever.search
) -> str:
    """Build the user turn that is sent to the LLM."""
    ev_lines: list[str] = []
    for doc, score in evidence:
        ev_lines.append(
            f"[{doc.id}] (source={doc.source}, similarity={score:.3f})\n"
            f"{doc.text.strip()[:1800]}\n"
        )
    return (
        "TARGET PATIENT\n"
        f"  ICD-10:    {icd_code}\n"
        f"  Diagnosis: {diagnosis_name}\n"
        f"  Age:       {age}\n"
        f"  Sex:       {sex}\n"
        f"  Country:   {country}\n\n"
        "EVIDENCE (ground every clinical fact in these passages)\n"
        + "\n---\n".join(ev_lines)
        + "\n\nOUTPUT JSON SCHEMA (write values, keep keys verbatim):\n"
        + json.dumps(_SCHEMA, indent=2)
    )
