"""
agent3_quality.py
=================
Quality agent — evaluates the content and completeness of a patient record.
Each question answered YES scores 1 point.

Questions: all entries from questions.json with evidence_type "supporting" or "contradicting".
"""
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from src.LLM_generation.llm_io import FullPatient


# ── Quality questions ────────────────────────────────────────────────────────────
# Answer YES = +1 point, NO = 0 points

QUALITY_QUESTIONS = [
    # -- Supporting evidence (SUP) --------------------------------------------
    ("SUP1",  "Are the most common symptoms of the disease present?"),
    ("SUP2",  "Does the diagnosis explain the majority of the patient's symptoms, and does the symptom pattern match the typical presentation of the disease?"),
    ("SUP3",  "Does the patient's medical history increase the risk of this disease?"),
    ("SUP4",  "Has the patient experienced similar episodes previously?"),
    ("SUP5",  "Is there family history of the same disease?"),
    ("SUP6",  "Is there family history of related disorders?"),
    ("SUP7",  "Does the patient have major known risk factors for this disease?"),
    ("SUP8",  "Are there comorbidities that predispose to this disease?"),
    ("SUP9",  "Does the patient's lifestyle include behaviors known to trigger this disease?"),
    ("SUP10", "Does the patient's occupation expose them to known risk factors for the disease?"),
    ("SUP11", "Do the patient's hobbies expose them to triggers associated with this disease?"),
    ("SUP12", "Could current medications contribute to the development of this disease?"),
    ("SUP13", "Is the onset and progression of symptoms temporally consistent with the natural course of this disease?"),
    ("SUP14", "Are ALL listed medications unrelated to the treatment of the primary diagnosis (i.e., they represent background/comorbidity drugs only, with no standard first-line treatment for the primary diagnosis present)?"),
]

MAX_SCORE = len(QUALITY_QUESTIONS)

# ── Models ───────────────────────────────────────────────────────────

class QualityAnswer(BaseModel):
    question_id: str
    answer: Literal["YES", "NO"]
    reasoning: str

class QualityResult(BaseModel):
    answers: list[QualityAnswer]
    score: int
    max_score: int


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM = """You are a clinical content evaluator.
For each question, answer YES if the patient record contains relevant, realistic,
and consistent data for that aspect. Answer NO if the data is missing, implausible,
or inconsistent. Answer YES if the question is not relevant for this disease. Be brief in your reasoning."""

agent = Agent(
    None,
    deps_type=FullPatient,
    output_type=list[QualityAnswer],
    system_prompt=SYSTEM,
)


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_quality(model, eval_input: FullPatient) -> QualityResult:
    agent.model = model
    questions_text = "\n".join(
        f"{q[0]}: {q[1]}" for q in QUALITY_QUESTIONS
    )
    prompt = f"""Diagnosis: {eval_input.diagnosis_name} ({eval_input.icd_code})
Gender: {eval_input.gender}, Age: {eval_input.age}

Patient data:
{eval_input.patient.model_dump_json(indent=2)}

Evaluate each quality question:
{questions_text}"""

    answers = agent.run_sync(prompt, deps=eval_input).output
    score = sum(1 for a in answers if a.answer == "YES")

    return QualityResult(
        answers=answers,
        score=score,
        max_score=len(QUALITY_QUESTIONS),
    )


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    from patient_agent import generate_patient, PatientOutput

    patient_data = generate_patient("K21.9", "Gastroesophageal reflux disease", "male", 59)
    eval_input = FullPatient(
        patient=PatientOutput.model_validate(patient_data),
        icd_code="K21.9",
        diagnosis_name="Gastroesophageal reflux disease",
        gender="male",
        age=59,
    )
    result = evaluate_quality(eval_input)
    print(_json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
