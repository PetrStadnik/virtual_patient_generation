"""
runner.py
=========
Core simulation logic extracted from views.py.
Can be imported and called directly — no HTTP request or Django context needed.

Usage (from evaluation pipeline):
    from simulation_app.sim.runner import evaluate_diagnosis
    result = evaluate_diagnosis(model, eval_input)

Usage (from views.py):
    from simulation_app.sim.runner import run_consultation, DiagnosisEvalResult
"""

import os
import re
from typing import Optional

import httpx
import anthropic
from openai import OpenAI
from pydantic import BaseModel

from src.LLM_generation.llm_io import FullPatient

MAX_TURNS    = 6     # doctor-patient exchanges per consultation
MAX_ATTEMPTS = 3     # initial diagnosis + 2 corrections
MAX_TOKENS   = 512   # per reply

_ICD_RE = re.compile(r"FINAL DIAGNOSIS:\s*([A-Z]\d{2}\.?\d*)\s*[-–]\s*(.+)", re.IGNORECASE)


# ── Provider detection ─────────────────────────────────────────────────────────

def _is_anthropic(model: str) -> bool:
    """True for Claude / Anthropic models."""
    m = model.lower()
    return m.startswith("anthropic:") or "claude" in m


def _bare_model(model: str) -> str:
    """Strip 'anthropic:' / 'openai:' prefix if present."""
    for prefix in ("anthropic:", "openai:"):
        if model.lower().startswith(prefix):
            return model[len(prefix):]
    return model


# ── Output models ──────────────────────────────────────────────────────────────

class DiagnosisAttempt(BaseModel):
    attempt: int
    icd_code: str           # e.g. "J45.9"
    diagnosis_name: str     # e.g. "Asthma, unspecified"
    correct: bool


class DiagnosisEvalResult(BaseModel):
    correct: bool
    correct_attempt: Optional[int]   # 1 / 2 / 3, or None if all wrong
    attempts: list[DiagnosisAttempt]


# ── System prompts ─────────────────────────────────────────────────────────────

def patient_system(fp: FullPatient) -> str:
    """Build the patient role-play system prompt from a FullPatient record."""
    p    = fp.patient
    name = fp.name
    desc    = p.patient_description.replace("{patientName}", name)
    history = p.patient_history.replace("{patientName}", name)
    meds    = ", ".join(m.medication_name for m in p.medication) or "none"
    allergies = ", ".join(a.substance_name for a in p.allergies) or "none"
    return (
        f"You are playing a patient named {name} in a doctor's consultation.\n"
        f"Age: {fp.age} | Gender: {fp.gender}\n\n"
        f"Your background: {desc}\n"
        f"Medical history: {history}\n"
        f"Current medications: {meds}\n"
        f"Known allergies: {allergies}\n\n"
        "Respond naturally as a real patient. Describe your symptoms honestly when asked. "
        "Do NOT mention your diagnosis name or any ICD codes. Keep replies to 2–4 sentences."
    )


def doctor_system(final_turn: int = MAX_TURNS) -> str:
    """Build the doctor system prompt."""
    return (
        f"You are an experienced physician conducting an initial patient consultation. "
        f"Ask focused, targeted questions to narrow down the diagnosis. "
        f"You have {final_turn} exchanges in total.\n\n"
        f"On your FINAL response (exchange {final_turn}) you MUST end with exactly:\n"
        "FINAL DIAGNOSIS: [ICD-10 code] - [condition name]\n\n"
        "ICD-10 code rules:\n"
        "- Use WHO ICD-10 format only (e.g. J45.9, I10, K21.0).\n"
        "- Do NOT use ICD-10-CM, SNOMED, or any other coding system.\n"
        "- Pattern: one letter + two digits, optionally dot + one digit.\n\n"
        "Keep each response concise (2–3 sentences + one follow-up question). "
        "Do not ask multiple questions at once."
    )


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _make_openai() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(verify=False),
    )


def _make_anthropic() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        http_client=httpx.Client(verify=False),
    )


def _chat(model: str,
          system: str, history: list[dict], user_msg: str) -> tuple[str, list[dict]]:
    """Send one message and return (reply, updated_history).

    Automatically routes to the correct provider based on the model name:
      - "gpt-*" / "openai:*"    → OpenAI
      - "claude-*" / "anthropic:*" → Anthropic
    """
    bare = _bare_model(model)
    messages = history + [{"role": "user", "content": user_msg}]

    if _is_anthropic(model):
        client = _make_anthropic()
        resp   = client.messages.create(
            model=bare,
            system=system,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )
        reply = resp.content[0].text if resp.content else ""
    else:
        client = _make_openai()
        full_messages = [{"role": "system", "content": system}] + messages
        reply = client.chat.completions.create(
            model=bare, messages=full_messages, max_completion_tokens=MAX_TOKENS,
        ).choices[0].message.content or ""

    updated = messages + [{"role": "assistant", "content": reply}]
    return reply, updated


def _format_test_results(fp: FullPatient) -> str:
    """Format patient's medical tests and results as a lab report for the doctor."""
    tests = fp.patient.medical_tests
    if not tests:
        return ""
    lines = ["[LAB / DIAGNOSTIC RESULTS — ordered and available to you]"]
    for t in tests:
        value = t.value_string
        if t.value_quantity is not None and t.value_unit:
            value = f"{t.value_quantity} {t.value_unit}  ({t.value_string})"
        elif t.value_quantity is not None:
            value = f"{t.value_quantity}  ({t.value_string})"
        lines.append(f"  • {t.test_name} [{t.loinc_code}]: {value}")
    return "\n".join(lines)


def _parse_icd(text: str) -> tuple[str, str]:
    """Extract (icd_code, diagnosis_name) from a doctor message, or ('', '')."""
    m = _ICD_RE.search(text)
    if m:
        return m.group(1).strip().upper(), m.group(2).strip()
    return "", ""


def _icd_match(guessed: str, correct: str) -> bool:
    """Accept if the ICD-10 category (first 3 chars) matches."""
    return bool(guessed) and guessed[:3].upper() == correct[:3].upper()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_consultation(
    model: str,
    fp: FullPatient,
    doctor_history: list[dict],
    previous_wrong: list[DiagnosisAttempt],
) -> tuple[str, list[dict]]:
    """Run one full consultation and return (last_doctor_msg, updated_doctor_history).

    On the first attempt a fresh doctor-patient conversation is started.
    On retries the doctor_history from the previous attempt is extended with
    a feedback message and a new round of MAX_TURNS exchanges begins.
    """
    pat_sys = patient_system(fp)
    doc_sys = doctor_system(final_turn=MAX_TURNS)
    patient_history: list[dict] = []

    # On retries inject corrective feedback into the doctor's history
    if previous_wrong:
        last = previous_wrong[-1]
        feedback = (
            f"[SYSTEM] Your diagnosis was INCORRECT: {last.icd_code} — {last.diagnosis_name}. "
            f"The patient has not improved under that treatment. "
            f"Re-examine the case and provide a revised FINAL DIAGNOSIS."
        )
        doctor_history = doctor_history + [{"role": "user", "content": feedback}]

    # Doctor opens the conversation
    last_doctor_msg, doctor_history = _chat(
        model, doc_sys, doctor_history,
        "Begin the consultation. Greet the patient and ask about their chief complaint."
    )

    lab_results = _format_test_results(fp)

    for turn in range(1, MAX_TURNS + 1):
        # Patient responds
        last_patient_msg, patient_history = _chat(
            model, pat_sys, patient_history, last_doctor_msg
        )

        # Doctor responds (final turn must contain FINAL DIAGNOSIS)
        doctor_prompt = last_patient_msg
        if turn == MAX_TURNS:
            # Inject lab/diagnostic results before the doctor's final answer
            if lab_results:
                doctor_prompt += f"\n\n{lab_results}"
            doctor_prompt += "\n\n[This is your final exchange. Provide your FINAL DIAGNOSIS now.]"
        last_doctor_msg, doctor_history = _chat(
            model, doc_sys, doctor_history, doctor_prompt
        )

    return last_doctor_msg, doctor_history


def evaluate_diagnosis(model: str, eval_input: FullPatient) -> DiagnosisEvalResult:
    """Run the simulation with up to MAX_ATTEMPTS diagnosis attempts.

    After each wrong diagnosis the doctor receives corrective feedback inside
    the same conversation history and gets another full round of MAX_TURNS
    exchanges to arrive at a new FINAL DIAGNOSIS.

    Args:
        model:      Model name with optional provider prefix.
                    OpenAI:    "gpt-4o", "gpt-4o-mini"
                    Anthropic: "claude-sonnet-4-5", "anthropic:claude-opus-4-7"
        eval_input: FullPatient (patient record + correct diagnosis)

    Returns:
        DiagnosisEvalResult with per-attempt details and overall correctness.
    """
    provider    = "Anthropic" if _is_anthropic(model) else "OpenAI"
    print(f"  [diagnosis] provider={provider}, model={_bare_model(model)}")

    correct_icd = eval_input.icd_code.strip().upper()
    doctor_hist: list[dict]            = []
    attempts:    list[DiagnosisAttempt] = []

    for n in range(1, MAX_ATTEMPTS + 1):
        last_msg, doctor_hist = run_consultation(
            model, eval_input, doctor_hist, attempts
        )

        guessed_icd, guessed_name = _parse_icd(last_msg)
        is_correct = _icd_match(guessed_icd, correct_icd)

        attempt = DiagnosisAttempt(
            attempt=n,
            icd_code=guessed_icd or "—",
            diagnosis_name=guessed_name or "not provided",
            correct=is_correct,
        )
        attempts.append(attempt)

        print(
            f"  [diagnosis] attempt {n}: {attempt.icd_code} — {attempt.diagnosis_name}"
            f"  {'CORRECT' if is_correct else 'wrong (target: ' + correct_icd + ')'}"
        )

        if is_correct:
            return DiagnosisEvalResult(
                correct=True,
                correct_attempt=n,
                attempts=attempts,
            )

    return DiagnosisEvalResult(
        correct=False,
        correct_attempt=None,
        attempts=attempts,
    )
