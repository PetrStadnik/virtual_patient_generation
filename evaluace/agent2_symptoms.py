"""
agent2_symptoms.py
==================
Checks which symptoms of the diagnosis are present in the patient record.

Flow:
  1. SNOMED lookup     — deterministic, via get_disease_symptoms()
  2. RAG lookup        — extracts additional symptoms from the local RAG index
  3. LLM fallback      — if both above return nothing, an LLM lists common symptoms
  4. Checker agent     — reads the patient record, marks each symptom present/absent
  5. SymptomEvalResult — symptom list + presence flags + coverage %

Result:
  symptom_coverage: float  — fraction of symptoms present, e.g. 0.75 = 75 %
"""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from src.LLM_generation.llm_io import FullPatient
from src.symptoms_snomed import get_disease_symptoms


# ── Output models ──────────────────────────────────────────────────────────────

class SymptomPresence(BaseModel):
    symptom: str
    present: bool
    evidence: str   # direct quote from the record, or "Not documented"


class SymptomCheck(BaseModel):
    presences: list[SymptomPresence]


class SymptomList(BaseModel):
    symptoms: list[str]


class SymptomEvalResult(BaseModel):
    icd_code: str
    diagnosis_name: str
    symptoms_checked: list[SymptomPresence]
    symptom_coverage: float   # fraction present  (0.0 – 1.0)


# ── Agent: RAG symptom extractor ───────────────────────────────────────────────

_rag_extractor = Agent(
    model=None,
    output_type=SymptomList,
    system_prompt=(
        "You are a clinical symptom selector. "
        "From the provided medical passages, identify the 3–10 symptoms, signs, or clinical findings "
        "that are MOST TYPICAL and CHARACTERISTIC for the given diagnosis. "
        "Prioritise hallmark findings (high sensitivity or specificity) over incidental ones. "
        "Return short, standardized English medical terms only (e.g. 'polyuria', 'fatigue'). "
        "Exclude treatments, medications, and procedures. "
        "Return AT LEAST 3 and AT MOST 10 items."
    ),
    model_settings=ModelSettings(temperature=0, max_tokens=400),
    retries=2,
)


def _symptoms_from_rag(model: str, icd_code: str, diagnosis_name: str) -> list[str]:
    """Query the local RAG index and extract symptom names from retrieved passages."""
    try:
        from rag.retrieve.index import Retriever
        retriever = Retriever()
        query = f"{diagnosis_name} {icd_code} symptoms signs findings"
        evidence = retriever.search(query, k=5)
        if not evidence:
            return []
        passages = "\n\n---\n\n".join(
            f"[{doc.id}]\n{doc.text[:800]}" for doc, _ in evidence
        )
        print(f"  [rag] retrieved {len(evidence)} passages for symptom extraction")
        _rag_extractor.model = model
        result: SymptomList = _rag_extractor.run_sync(
            f"Diagnosis: {diagnosis_name} ({icd_code})\n\n"
            f"Select the 3–10 most typical symptoms for this diagnosis from the passages below.\n\n"
            f"Passages:\n{passages}"
        ).output
        # Enforce hard bounds
        result.symptoms = result.symptoms[:10]
        print(f"  [rag] extracted {len(result.symptoms)} symptoms: {result.symptoms[:5]}")
        return result.symptoms
    except Exception as e:
        print(f"  [rag] symptom extraction failed: {e}")
        return []


# ── Agent: LLM knowledge fallback ─────────────────────────────────────────────

_llm_fallback = Agent(
    model=None,
    output_type=SymptomList,
    system_prompt=(
        "You are a medical knowledge base. "
        "List 3–10 of the most common and characteristic symptoms/signs for the given diagnosis. "
        "Return short, standardized English medical terms only."
    ),
    model_settings=ModelSettings(temperature=0, max_tokens=300),
    retries=2,
)


def _symptoms_from_llm(model: str, icd_code: str, diagnosis_name: str) -> list[str]:
    """LLM fallback: returns a symptom list from model knowledge alone."""
    print(f"  [llm-fallback] generating symptoms for {icd_code} from model knowledge")
    _llm_fallback.model = model
    result: SymptomList = _llm_fallback.run_sync(
        f"Diagnosis: {diagnosis_name} (ICD-10: {icd_code})"
    ).output
    print(f"  [llm-fallback] got {len(result.symptoms)} symptoms")
    return result.symptoms


# ── Agent: checker ─────────────────────────────────────────────────────────────

_checker = Agent(
    model=None,
    deps_type=FullPatient,
    output_type=SymptomCheck,
    system_prompt="""You are a clinical record reviewer.

You receive a list of symptoms and a structured patient record.
For each symptom decide:
  - present: true   if the record EXPLICITLY mentions or supports it
  - present: false  if the symptom is absent or simply not mentioned
  - evidence: one short sentence quoting the exact part of the record,
              or "Not documented" if there is no mention

Rules:
  - Be strict: only mark present=true when the record explicitly supports it.
  - Do not infer beyond what is written.
  - For lab findings check value_quantity / value_string fields.
  - For drug-related symptoms check current_medications.
""",
    model_settings=ModelSettings(temperature=0, max_tokens=2000),
    retries=2,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_symptoms(model: str, eval_input: FullPatient) -> SymptomEvalResult:
    """Collect symptoms from SNOMED + RAG (+ LLM fallback), then check presence.

    Args:
        model:      pydantic-ai model string, e.g. "anthropic:claude-..."
        eval_input: patient record + diagnosis metadata (FullPatient)

    Returns:
        SymptomEvalResult with per-symptom presence flags and overall coverage %.
    """
    _checker.model = model

    # Step 1 — deterministic symptom list from SNOMED
    ds = get_disease_symptoms(eval_input.icd_code, diagnosis_name=eval_input.diagnosis_name)
    snomed_symptoms = [sf.snomed.display for sf in ds.symptoms]
    print(f"  [snomed] {len(snomed_symptoms)} symptoms found")

    # Step 2 — additional symptoms extracted from the RAG index
    rag_symptoms = _symptoms_from_rag(model, eval_input.icd_code, eval_input.diagnosis_name)

    # Merge: deduplicate case-insensitively, SNOMED names take priority
    seen = {s.lower() for s in snomed_symptoms}
    extra = [s for s in rag_symptoms if s.lower() not in seen]
    symptom_names = snomed_symptoms + extra
    print(f"  [symptoms] {len(snomed_symptoms)} SNOMED + {len(extra)} RAG = {len(symptom_names)} total")
    if snomed_symptoms:
        print(f"  [snomed symptoms] " + ", ".join(snomed_symptoms))
    if extra:
        print(f"  [rag symptoms]    " + ", ".join(extra))

    # Step 3 — LLM fallback if both sources returned nothing
    if not symptom_names:
        print(f"  [symptoms] no symptoms from SNOMED or RAG — using LLM fallback")
        symptom_names = _symptoms_from_llm(model, eval_input.icd_code, eval_input.diagnosis_name)
        if symptom_names:
            print(f"  [llm symptoms]    " + ", ".join(symptom_names))

    if not symptom_names:
        return SymptomEvalResult(
            icd_code=eval_input.icd_code,
            diagnosis_name=eval_input.diagnosis_name,
            symptoms_checked=[],
            symptom_coverage=0.0,
        )

    symptom_list = "\n".join(f"  - {name}" for name in symptom_names)

    # Step 4 — checker agent determines presence in the patient record
    print(f"  [checking] {len(symptom_names)} symptoms against patient record:")
    for name in symptom_names:
        src = "snomed" if name in snomed_symptoms else ("rag" if name in extra else "llm")
        print(f"    [{src}] {name}")
    result: SymptomCheck = _checker.run_sync(
        f"Diagnosis: {eval_input.diagnosis_name} (ICD-10: {eval_input.icd_code})\n\n"
        f"Symptoms to check:\n{symptom_list}\n\n"
        f"Patient record:\n{eval_input.patient.model_dump_json(indent=2)}",
        deps=eval_input,
    ).output

    # Step 5 — compute coverage
    n_present = sum(1 for p in result.presences if p.present)
    coverage = round(n_present / len(result.presences), 4) if result.presences else 0.0

    return SymptomEvalResult(
        icd_code=eval_input.icd_code,
        diagnosis_name=eval_input.diagnosis_name,
        symptoms_checked=result.presences,
        symptom_coverage=coverage,
    )
