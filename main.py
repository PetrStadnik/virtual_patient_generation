from __future__ import annotations
import json
import time
from pathlib import Path
from enum import StrEnum
import random
from typing import Literal

from src.LLM_generation.agent_with_tools import generate_patient_with_tools
from src.icd_lookup import validate_icd_for_fhir, ICDDiagnosisInfo
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from src.basic_data.GBD import get_max_probable_age_ang_gender
from src.basic_data.name_generator import get_name_and_surname
from src.bundle_base import BundleBase
from src.LLM_generation.baseline_script import generate_patient
from src.LLM_generation.LLM2FHIR import fill_bundle
from rag.generate.run import generate_patient_rag_with_tools
from multiagentic.run import generate_patient as generate_patient_multiagentic

PROJECT_ROOT: Path = Path(__file__).resolve().parent

class LLM_model(StrEnum):
    OPENAI = "gpt-5.4"
    ANTHROPIC = "claude-sonnet-4-6"


# =============================================================================
# SETTINGS
# =============================================================================

MODEL: str = LLM_model.OPENAI
OUTPUT_FOLDER: Path = PROJECT_ROOT / "scenarios" / "multiagent"

"""INPUT EXAMPLE
INPUT: list[dict] = [
    {
     "diagnosis_code": "I21.1",
     "patient_age": 58,
     "patient_sex": "female",
     "patient_country": "CZ"
    },
]
"""
# -- ICD-10 codes for testing
icd10_kody = [
    "J00",
    "J06.9",
    "J02.9",
    "J03.9",
    "J20.9",
    "J18.9",
    "K21.9",
    "K29.7",
    "A09.0",
    "N39.0",
    "K75.9",
    "E11.9",
    "E78.5",
    "M54.5",
    "M17.9",
    "M79.1",
    "G43.9",
    "G40.9",
    "G35",
    "G20",
    "F32.9",
    "F41.1",
    "F20.9",
    "F31.9",
    "F90.0",
    "C34.9",
    "C50.9",
    "C18.9",
    "C61",
    "B20.0"
]

INPUT = [{"diagnosis_code": c, "patient_country": "CZ"} for c in icd10_kody]
# =============================================================================
class PipelineInput(BaseModel):
    """JSON input structure for the pipeline."""

    diagnosis_code: str = Field(
        ...,
        description="ICD-10 code of the primary diagnosis",
        min_length=1,
    )
    patient_age: int | None = Field(
        default=None,
        ge=3,
        le=100,
        description="Patient age in years (optional)",
    )
    patient_sex: Literal["male", "female"] | None = Field(
        default=None,
        description='Sex: "male" or "female" (optional)',
    )

    patient_country: str | None = Field(
        min_length=2,
        max_length=2,
        default="CZ",
        description='Country code for name selection - CZ/DE/US/... (optional)',
    )

# ======================================================================

def run_pipeline(raw_input: dict):
    # validate input
    try:
        input = PipelineInput.model_validate(raw_input)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return

    # check ICD-10 code
    icd_info: ICDDiagnosisInfo = validate_icd_for_fhir(input.diagnosis_code)
    if not icd_info.valid or not icd_info.billable:
        print(f"Invalid ICD-10 code: {input.diagnosis_code}, exiting")
        exit(1)
    #print(f"ICD-10 code is valid: {input.diagnosis_code}")
    #print(f"ICD-10 description: {icd_info.description}")
    #print(f"ICD-10 block: {icd_info.block_description}")
    #print(f"ICD-10 chapter: {icd_info.chapter_description}")
    #print("===============================================")

    # -- GET AGE and GENDER according to diagnosis
    gender, age = None, None
    if not input.patient_age or not input.patient_sex:
        age, gender = get_max_probable_age_ang_gender(input.diagnosis_code)
    
    age = input.patient_age if input.patient_age else age
    gender = input.patient_sex if input.patient_sex else gender

    # -- GET NAME of the patient
    name, surname = get_name_and_surname(gender.upper()[0], input.patient_country)

    # -- GENERETE LLM content
    for model in range(2):
        if model == 0:
            MODEL = LLM_model.OPENAI
            print("Using OpenAI model")
        else:
            MODEL = LLM_model.ANTHROPIC
            print("Using Anthropic model")

        # BASELINE AGENT:
        #llm_content = generate_patient(model=MODEL, icd_code=input.diagnosis_code, diagnosis_name=icd_info.description, gender=gender, age=age)

        # AGENT WITH TOOLS:
        #llm_content = generate_patient_with_tools(model=MODEL, icd_code=input.diagnosis_code, diagnosis_name=icd_info.description, gender=gender, age=age)

        # RAG AGENT (requires built index — run `python rag/retrieve/index.py` first):
        #llm_content = generate_patient_rag(model=MODEL, icd_code=input.diagnosis_code, diagnosis_name=icd_info.description, gender=gender, age=age, country=input.patient_country)

        # RAG + TOOLS (retrieved evidence + live ATC/LOINC/SNOMED lookups):
        #llm_content = generate_patient_rag_with_tools(model=MODEL, icd_code=input.diagnosis_code, diagnosis_name=icd_info.description, gender=gender, age=age, country=input.patient_country)

        # MULTI AGENTIC (generate + evaluate loop + fix, requires langgraph):
        llm_content = generate_patient_multiagentic(model=MODEL, icd_code=input.diagnosis_code, diagnosis_name=icd_info.description, gender=gender, age=age, country=input.patient_country, patient_name=name+" "+surname)

        # -- create FHIR bundle
        bb = BundleBase(input.diagnosis_code, icd_info.description, age, gender, name, surname, input.patient_country)
        bb.bundle = fill_bundle(bb.bundle, llm_content)
        subfolder: str = "anthropic" if MODEL == LLM_model.ANTHROPIC else "openai"
        patient_id = f"{icd_info.code}_{age}_{name}{surname}"
        with open(OUTPUT_FOLDER/subfolder/(str(patient_id)+".json"), "w", encoding="utf-8") as f:
            json.dump(bb.bundle, f, indent=2, ensure_ascii=False)

        # -- validate bundle structure

        res_struct = bb.validate_bundle_structure()

        code_errors = 0
        for ie in range(10):
            ok, errors, warnings = bb.validate_bundle_all()
            if len(errors)==0:
                break
            elif len(errors)>0 and not "Unable to connect to terminology server" in errors[0]:
                for e in errors:
                    if "Unknown code" in e:
                        code_errors +=1
                break
            else:
                print("Unable to connect to terminology server, retrying...")
                time.sleep(ie+2+random.uniform(0.5, 2.5))
        print("Appending validation results to validation.jsonl...")
        with open(OUTPUT_FOLDER/"validation.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"model": subfolder, "patient_id": patient_id, "res_struct": res_struct, "all_errors": errors, "code_errors": code_errors}, ensure_ascii=False) + "\n")
    """
        if not ok:
            print(f"Validation failed with {len(errors)} errors and {len(warnings)} warnings.")
            print("Errors:")
            for e in errors:
                print(f"{e}\n")
    
            print("Warnings:")
            for w in warnings:
                print(f"{w}\n")
        else:
            print("Validation OK.")
     """

if __name__ == "__main__":
    # load models API keys from .env
    load_dotenv(PROJECT_ROOT / ".env")
    print(f"Running pipeline for {len(INPUT)} ICD-10 codes...")
    i = 0
    from src.symptoms_snomed import get_symptoms_snomed
    for input in INPUT:
        i += 1
        print(i)
        print(input)
        run_pipeline(input)
        #dia_info=validate_icd_for_fhir(input["diagnosis_code"])
        #print(dia_info.description)
        #print(dia_info.code.split(".")[0])
        #print(get_symptoms_snomed(input["diagnosis_code"]))
        #print(get_symptoms_fhir(dia_info.code, dia_info.description))
    print("Pipeline finished.")




