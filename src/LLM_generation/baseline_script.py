from pydantic_ai import Agent
from src.LLM_generation.llm_io import PatientInput, PatientOutput
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.exceptions import ModelHTTPError
import random
import time

# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM = """You are a medical simulation specialist generating realistic virtual patients for clinical education.

Given an ICD-10 code, diagnosis, gender and age, generate a complete virtual patient profile.

Rules:
- Use placeholder {patientName} in patient_description and patient_history.
- Generate 2-5 medical tests typical for the diagnosis.
- Generate 1-6 medications appropriate for the patient profile (age, gender, lifestyle), NOT connected to the diagnosis.
- Make the patient description and history realistic and clinically accurate.
"""



def generate_patient(model: str, icd_code: str, diagnosis_name: str, gender: str, age: int, dump=True) -> dict | PatientOutput:

    if "claude" in model:
        model = AnthropicModel(model)
    else:
        model = OpenAIModel(model)

    agent = Agent(
        model,
        deps_type=PatientInput,
        output_type=PatientOutput,
        system_prompt=SYSTEM,
    )
    input_data = PatientInput(icd_code=icd_code, diagnosis_name=diagnosis_name, gender=gender, age=age)
    prompt = (f"Generate a virtual patient. "
              f"ICD-10: {icd_code} – {diagnosis_name}. "
              f"Gender: {gender}, Age: {age}.")
    
    result = None
    for attempt in range(10):
        try:
            print(f"Attempt {attempt+1}...")
            result = agent.run_sync(prompt, deps=input_data)
            break
        except ModelHTTPError as e:
            if e.status_code == 529:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Overloaded. Retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    if dump:
        return result.output.model_dump()
    else:
        print(type(result.output))
        return result.output

