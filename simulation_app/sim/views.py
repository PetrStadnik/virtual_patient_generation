import json
import traceback

from django.conf import settings
from django.shortcuts import render
from django.http import StreamingHttpResponse

from simulation_app.sim.runner import (
    MAX_TURNS,
    _chat,
    _parse_icd,
    _icd_match,
    patient_system,
    doctor_system,
)
from src.bundle_to_llm import bundle_from_file, bundle_to_eval_input


def _full_patient_to_scenario(fp) -> dict:
    """Convert FullPatient (from a FHIR bundle) to the scenario dict used by the web UI."""
    p = fp.patient
    return {
        "icd_code":       fp.icd_code,
        "diagnosis_name": fp.diagnosis_name,
        "gender":         fp.gender,
        "age":            fp.age,
        "patient": {
            "patient_description":   p.patient_description,
            "diagnosis_description": p.diagnosis_description,
            "patient_history":       p.patient_history,
            "medical_tests":  [t.model_dump() for t in p.medical_tests],
            "medication":     [m.model_dump() for m in p.medication],
            "allergies":      [a.model_dump() for a in p.allergies],
            "family_history": [fh.model_dump() for fh in p.family_history],
            "procedures":     [pr.model_dump() for pr in p.procedures],
        },
    }


def _load_default_scenario() -> dict:
    """Load DEFAULT_SCENARIO from FHIR bundle file if configured, otherwise use built-in."""
    bundle_path = getattr(settings, "SIMULATION_BUNDLE", "")
    if bundle_path:
        try:
            bundle = bundle_from_file(bundle_path)
            fp = bundle_to_eval_input(bundle)
            print(f"[simulation_app] Loaded scenario from bundle: {bundle_path}")
            return _full_patient_to_scenario(fp)
        except Exception as e:
            print(f"[simulation_app] WARNING: Could not load bundle '{bundle_path}': {e}")
    return _BUILTIN_DEFAULT_SCENARIO


_BUILTIN_DEFAULT_SCENARIO = {
    "icd_code": "J45.9",
    "diagnosis_name": "Asthma, unspecified",
    "gender": "female",
    "age": 34,
    "patient": {
        "patient_description": (
            "{patientName} is a 34-year-old female elementary school teacher. "
            "She is slim, slightly anxious-looking, and arrives at the clinic wearing a scarf. "
            "She enjoys hiking and cycling on weekends but lately has had to cut back due to breathing difficulties. "
            "She lives with her husband and two children in a house with a cat."
        ),
        "diagnosis_description": (
            "Asthma is a chronic inflammatory airway disease characterised by variable airflow obstruction, "
            "bronchial hyperresponsiveness and recurrent episodes of wheezing, breathlessness, chest tightness and cough."
        ),
        "patient_history": (
            "{patientName} had childhood eczema and hay fever. "
            "She was hospitalised once at age 12 for a severe wheeze. "
            "Non-smoker. No current inhaler prescription — stopped using one 5 years ago when symptoms seemed to improve."
        ),
        "medical_tests": [
            {"test_name": "Spirometry", "loinc_code": "19839-0", "category": "laboratory",
             "value_quantity": 68.0, "value_unit": "%", "value_string": "FEV1/FVC 68% — mild obstruction"},
            {"test_name": "Peak expiratory flow", "loinc_code": "19935-6", "category": "vital-signs",
             "value_quantity": 310.0, "value_unit": "L/min", "value_string": "310 L/min (75% predicted)"},
            {"test_name": "Total IgE", "loinc_code": "19146-0", "category": "laboratory",
             "value_quantity": 210.0, "value_unit": "kU/L", "value_string": "210 kU/L — elevated"},
        ],
        "medication": [
            {"medication_name": "Cetirizine", "atc_code": "R06AE07", "dose": "10 mg",
             "route": "oral", "frequency": "once daily"},
        ],
        "allergies": [
            {"substance_name": "Cat dander", "substance_code": "", "allergy_type": "allergy",
             "category": "environment", "criticality": "low", "reaction": "Rhinitis, watery eyes", "severity": "mild"},
        ],
        "family_history": [
            {"relationship": "MTH", "condition_icd": "J45.9", "condition_name": "Asthma, unspecified",
             "deceased": False, "onset_age": 28},
        ],
        "procedures": [],
    },
}

DEFAULT_SCENARIO = _load_default_scenario()


def _patient_system_from_dict(data: dict) -> str:
    """Build patient system prompt from a raw scenario dict (used by the web view)."""
    p    = data.get("patient", {})
    name = "Alex"
    desc     = p.get("patient_description", "").replace("{patientName}", name)
    history  = p.get("patient_history", "").replace("{patientName}", name)
    meds     = ", ".join(m.get("medication_name", "") for m in p.get("medication", [])) or "none"
    allergies = ", ".join(a.get("substance_name", "") for a in p.get("allergies", [])) or "none"
    return (
        f"You are playing a patient named {name} in a doctor's consultation.\n"
        f"Age: {data.get('age')} | Gender: {data.get('gender')}\n\n"
        f"Your background: {desc}\n"
        f"Medical history: {history}\n"
        f"Current medications: {meds}\n"
        f"Known allergies: {allergies}\n\n"
        "Respond naturally as a real patient. Describe your symptoms honestly when asked. "
        "Do NOT mention your diagnosis name or any ICD codes. Keep replies to 2–4 sentences."
    )


def index(request):
    return render(request, 'index.html', {
        'default_scenario': json.dumps(DEFAULT_SCENARIO, indent=2),
        'max_turns': MAX_TURNS,
    })


def simulate(request):
    raw = request.GET.get('scenario', '{}')
    try:
        data = json.loads(raw)
    except Exception:
        data = DEFAULT_SCENARIO

    model = data.get("model", "gpt-4o-mini")

    def event_stream():
        try:
            pat_sys     = _patient_system_from_dict(data)
            doc_sys     = doctor_system(final_turn=MAX_TURNS)
            doctor_hist: list[dict]  = []
            patient_hist: list[dict] = []
            last_doctor_msg = ""

            # Doctor opens
            last_doctor_msg, doctor_hist = _chat(
                model, doc_sys, doctor_hist,
                "Begin the consultation. Greet the patient briefly and ask about their chief complaint."
            )
            yield f"data: {json.dumps({'role': 'doctor', 'text': last_doctor_msg, 'turn': 1})}\n\n"

            for turn in range(1, MAX_TURNS + 1):
                # Patient responds
                last_patient_msg, patient_hist = _chat(
                    model, pat_sys, patient_hist, last_doctor_msg
                )
                yield f"data: {json.dumps({'role': 'patient', 'text': last_patient_msg, 'turn': turn})}\n\n"

                # Doctor responds (last turn: must diagnose)
                doctor_prompt = last_patient_msg
                if turn == MAX_TURNS:
                    doctor_prompt += "\n\n[This is your final exchange. Provide your FINAL DIAGNOSIS now.]"
                last_doctor_msg, doctor_hist = _chat(
                    model, doc_sys, doctor_hist, doctor_prompt
                )
                yield f"data: {json.dumps({'role': 'doctor', 'text': last_doctor_msg, 'turn': turn, 'is_final': turn == MAX_TURNS})}\n\n"

            # Reveal correct answer
            correct_icd  = data.get("icd_code", "")
            diag_name    = data.get("diagnosis_name", "")
            guessed_icd, _ = _parse_icd(last_doctor_msg)
            correct_flag = _icd_match(guessed_icd, correct_icd) if guessed_icd else None
            yield f"data: {json.dumps({'role': 'result', 'correct_icd': correct_icd, 'diagnosis_name': diag_name, 'guessed_icd': guessed_icd, 'correct': correct_flag})}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'role': 'error', 'text': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response
