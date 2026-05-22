import random
import re
import time
import warnings
from pathlib import Path

import requests
from urllib3.exceptions import InsecureRequestWarning
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from src.LLM_generation.llm_io import PatientInput, PatientOutput

# Suppress SSL warnings caused by corporate/university HTTPS inspection proxies
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ── Regex ───────────────────────────────────────────────────────────────────

_ATC_RE   = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")

# ── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM = """You are a medical simulation specialist generating realistic virtual patients for clinical education.

Rules:
- Medications: call search_drug for every medication. Only include if found=true.
- Medications MUST be for comorbidities, lifestyle conditions, or pre-existing issues — NOT for treating the primary diagnosis.
- Medical tests: call search_loinc for every test. Only include if found=true.
- Allergies: call search_drug for medication allergens to get ATC code.
- Procedures: call search_snomed_procedure for every procedure. Only include if found=true.
- Family history: use valid ICD-10 codes (format A00 or A00.0).
- Vital signs: set category='vital-signs', provide value_quantity and value_unit (UCUM).
- Use {patientName} placeholder in patient_description and patient_history.
- ATC format: A00AA00. LOINC format: NNNNN-N.
"""

agent = Agent(
    None,
    deps_type=PatientInput,
    output_type=PatientOutput,
    system_prompt=SYSTEM,
    retries=3,
)


# ── Tools ──────────────────────────────────────────────────────────────────

_WIKIDATA_URL = "https://query.wikidata.org/sparql"
_WIKIDATA_HEADERS = {"User-Agent": "fhir-patient-generator/1.0"}


def _wikidata(sparql: str) -> list[dict]:
    """Execute a Wikidata SPARQL query, SSL verification disabled for proxy environments."""
    try:
        r = requests.get(
            _WIKIDATA_URL,
            params={"query": sparql, "format": "json"},
            headers=_WIKIDATA_HEADERS,
            timeout=15,
            verify=False,
        )
        return r.json()["results"]["bindings"]
    except Exception:
        return []


@agent.tool
def search_drug(ctx: RunContext[PatientInput], drug_name: str) -> dict:
    """Look up a drug by generic name. Returns WHO ATC code (format A00AA00).

    Uses Wikidata EntitySearch to find drugs by name and retrieve their ATC code
    (Wikidata property P267). Returns the first exact-level ATC code found.
    """
    sparql = f"""SELECT ?atc WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{drug_name}";
                        mwapi:language "en".
        ?d wikibase:apiOutputItem mwapi:item.
      }}
      ?d wdt:P267 ?atc .
    }} LIMIT 5"""
    rows = _wikidata(sparql)
    # Prefer the full 7-character ATC code (e.g. A02BC01) over group-level codes
    for row in rows:
        code = row["atc"]["value"]
        if _ATC_RE.match(code):
            print(f"Checking drug: {drug_name}: Found: {code}")
            return {"found": True, "drug_name": drug_name, "atc_code": code}
    if rows:
        code = rows[0]["atc"]["value"]
        print(f"Found (group-level ATC): {code}")
        return {"found": True, "drug_name": drug_name, "atc_code": code}
    print(f"Checking drug: {drug_name}: Not found")
    return {"found": False, "drug_name": drug_name, "atc_code": ""}


@agent.tool
def search_loinc(ctx: RunContext[PatientInput], test_name: str) -> dict:
    """Look up a medical test or observation by name. Returns LOINC code (format NNNNN-N).

    Uses the NLM Clinical Tables Search Service which covers the full LOINC database
    including lab tests, vital signs, and clinical observations.
    """
    try:
        r = requests.get(
            "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search",
            params={"terms": test_name, "df": "LOINC_NUM,LONG_COMMON_NAME", "maxList": 1},
            timeout=15,
            verify=False,
        )
        data = r.json()
        codes = data[1]  # array of LOINC codes
        if codes:
            code = codes[0]
            print(f"Checking test: {test_name}: Found: {code}")
            return {"found": True, "test_name": test_name, "loinc_code": code}
    except Exception:
        pass
    print(f"Checking test: {test_name}: Not found")
    return {"found": False, "test_name": test_name, "loinc_code": ""}


@agent.tool
def search_snomed_procedure(ctx: RunContext[PatientInput], procedure_name: str) -> dict:
    """Look up a medical procedure. Returns SNOMED CT code.

    Uses the CSIRO Ontoserver FHIR terminology service to search the SNOMED CT
    procedure hierarchy (ECL: <<71388002) for the best matching concept.
    """
    try:
        r = requests.get(
            "https://r4.ontoserver.csiro.au/fhir/ValueSet/$expand",
            params={
                "url": "http://snomed.info/sct?fhir_vs=ecl/<<71388002",
                "filter": procedure_name,
                "count": 1,
            },
            timeout=15,
            verify=False,
        )
        if r.status_code == 200:
            items = r.json().get("expansion", {}).get("contains", [])
            if items:
                code = items[0]["code"]
                display = items[0].get("display", procedure_name)
                print(f"Checking procedure: {procedure_name}: Found: {code} ({display})")
                return {"found": True, "procedure_name": procedure_name, "snomed_code": code}
    except Exception:
        pass
    print(f"Checking procedure: {procedure_name}: Not found")
    return {"found": False, "procedure_name": procedure_name, "snomed_code": ""}


# ── Main ───────────────────────────────────────────────────────────────

def generate_patient_with_tools(model: str, icd_code: str, diagnosis_name: str, gender: str, age: int) -> dict:
    agent.model = model
    ctx = PatientInput(icd_code=icd_code, diagnosis_name=diagnosis_name, gender=gender, age=age)
    prompt = f"Generate a virtual patient. ICD-10: {icd_code} - {diagnosis_name}. Gender: {gender}, Age: {age}."

    result = None
    for attempt in range(10):
        try:
            print(f"Attempt {attempt+1}...")
            result = agent.run_sync(prompt, deps=ctx)
            break
        except ModelHTTPError as e:
            if e.status_code == 529:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Overloaded. Retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise

    return result.output.model_dump()


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    load_dotenv(PROJECT_ROOT / ".env")
    patient = generate_patient( "claude-sonnet-4-6", "K21.9", "Gastroesophageal reflux disease", "male", 59)
    print(json.dumps(patient, indent=2, ensure_ascii=False))