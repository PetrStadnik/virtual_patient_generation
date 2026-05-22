import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIModel
from src.LLM_generation.llm_io import FullPatient

# ── Gate questions ───────────────────────────────────────────────────────────────
# Question: (id, question_text, reject_on)
# "YES" = reject if YES
# "NO"  = reject if NO

GATE_QUESTIONS = [
    # -- Mandatory required (REQ) ------------------------------------------------
    ("REQ1", "Is this diagnosis plausible for this gender?", "NO"),
    ("REQ2", "Is this diagnosis plausible for this age group?", "NO"),
    ("REQ3", "Are the key hallmark symptoms of the diagnosis present?", "NO"),
    ("REQ4", "Are the minimum required diagnostic findings present?", "NO"),

    # -- Red flags (RF) ---------------------------------------------------
    ("RF1",  "Is there a finding that clearly contradicts this diagnosis?", "YES"),
    ("RF2",  "Is there another diagnosis that explains the symptoms better?", "YES"),
    ("RF3",  "Could the symptoms be explained as medication side effects instead of this disease?", "YES"),
    ("RF4",  "Does the patient possess the anatomical structures required for this diagnosis?", "NO"),
    ("RF5",  "Are all provided test results consistent with the expected findings for this diagnosis?", "NO"),
    ("RF6",  "Is there a test result that is impossible or highly atypical for this disease?", "YES"),
    ("RF7",  "Is there a test result that is physiologically implausible given the patient's age, gender, or medical history?", "YES"),
    ("RF8",  "Are all numeric test results within physiologically possible ranges?", "NO"),

    # -- Safety (SAF) ------------------------------------------------------
    ("SAF1", "Are any prescribed medications contraindicated given the patient's documented allergies?", "YES"),
    ("SAF2", "Are the prescribed medication doses within clinically accepted ranges for the patient's age, weight, and renal/hepatic function?", "NO"),
]

REJECT_ON_YES = {qid for qid, _, reject_on in GATE_QUESTIONS if reject_on == "YES"}
REJECT_ON_NO  = {qid for qid, _, reject_on in GATE_QUESTIONS if reject_on == "NO"}


# ── Output models ───────────────────────────────────────────────────────────

class GateAnswer(BaseModel):
    question_id: str
    answer: Literal["YES", "NO"]
    reasoning: str

class GateAnswers(BaseModel):
    answers: list[GateAnswer]

class GateResult(BaseModel):
    answers: list[GateAnswer]
    passed: bool
    rejection_reasons: list[str]


# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM = """You are a clinical evaluation agent.
Answer each gate question with YES or NO based strictly on the patient data provided.
Be concise in your reasoning. Do not infer information not explicitly present in the data."""


# ── Main function ───────────────────────────────────────────────────────────────

def evaluate_gate(model:str, eval_input: FullPatient) -> GateResult:
    if "claude" in model:
        model = AnthropicModel(model)
    else:
        model = OpenAIModel(model)

    agent = Agent(
        model,
        deps_type=FullPatient,
        output_type=GateAnswers,
        system_prompt=SYSTEM,
        retries=3,
        model_settings=ModelSettings(temperature=0)
    )

    questions_text = "\n".join(f"{q[0]}: {q[1]}" for q in GATE_QUESTIONS)
    prompt = f"""Diagnosis: {eval_input.diagnosis_name} ({eval_input.icd_code})
    Gender: {eval_input.gender}, Age: {eval_input.age}
    
    Patient data:
    {eval_input.patient.model_dump_json(indent=2)}
    
    Answer each of the following gate questions:
    {questions_text}"""

    answers = agent.run_sync(prompt, deps=eval_input).output.answers

    reasons = []
    for a in answers:
        if a.question_id in REJECT_ON_YES and a.answer == "YES":
            reasons.append(f"{a.question_id}: {a.reasoning}")
        if a.question_id in REJECT_ON_NO and a.answer == "NO":
            reasons.append(f"{a.question_id}: {a.reasoning}")

    return GateResult(
        answers=answers,
        passed=len(reasons) == 0,
        rejection_reasons=reasons,
    )




