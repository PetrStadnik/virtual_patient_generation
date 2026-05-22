import json
from pathlib import Path
from src.LLM_generation.llm_io import (FullPatient, PatientOutput, MedicalTest, Medication, Allergy, FamilyHistory, MedicalProcedure)



# ── Helpers ────────────────────────────────────────────────────────────

KNOWN_ROUTES = {"oral", "intravenous", "subcutaneous", "intramuscular",
                "topical", "inhaled", "rectal", "transdermal"}

def _resources(bundle: dict, rtype: str) -> list[dict]:
    return [e["resource"] for e in bundle.get("entry", [])
            if e["resource"]["resourceType"] == rtype]

def _first(bundle: dict, rtype: str) -> dict | None:
    resources = _resources(bundle, rtype)
    return resources[0] if resources else None

def _coding_code(codings: list, system: str) -> str:
    for c in codings:
        if c.get("system") == system:
            return c.get("code", "")
    return ""

def _communication_text(bundle: dict, doc_type: str) -> str:
    for r in _resources(bundle, "Communication"):
        for cat in r.get("category", []):
            for c in cat.get("coding", []):
                if c.get("code") == doc_type:
                    payload = r.get("payload", [{}])
                    return payload[0].get("contentString", "") if payload else ""
    return ""

def _parse_dosage(text: str) -> tuple[str, str, str]:
    """Parse dosage text into dose, route, and frequency."""
    tokens = text.split()
    route_idx = next((i for i, t in enumerate(tokens) if t.lower() in KNOWN_ROUTES), None)
    if route_idx is None:
        return text, "", ""
    dose      = " ".join(tokens[:route_idx])
    route     = tokens[route_idx]
    frequency = " ".join(tokens[route_idx + 1:])
    return dose, route, frequency


# ── Get data from FHIR bundle ─────────────────────────────────────────────

def _extract_patient_info(bundle: dict) -> dict:
    p = _first(bundle, "Patient")
    if not p:
        return {}
    name_entry = p.get("name", [{}])[0]
    given  = name_entry.get("given", [""])[0]
    family = name_entry.get("family", "")
    gender = p.get("gender", "")
    birth_year = int(p.get("birthDate", "0")[:4]) if p.get("birthDate") else 0
    age = None
    for tag in bundle.get("meta", {}).get("tag", []):
        if tag.get("code") == "age":
            try:
                age = int(tag.get("display", "0"))
            except ValueError:
                pass
    if age is None and birth_year:
        from datetime import date
        age = date.today().year - birth_year

    return {
        "patient_name": f"{given} {family}".strip(),
        "gender": gender,
        "age": age,
    }


def _extract_diagnosis(bundle: dict) -> dict:
    c = _first(bundle, "Condition")
    if not c:
        return {}
    coding = c.get("code", {}).get("coding", [{}])[0]
    return {
        "icd_code":              coding.get("code", ""),
        "diagnosis_name":        coding.get("display", ""),
        "diagnosis_description": (c.get("note") or [{}])[0].get("text", ""),
    }


def _extract_tests(bundle: dict) -> list[MedicalTest]:
    tests = []
    for obs in _resources(bundle, "Observation"):
        code_obj  = obs.get("code", {})
        codings   = code_obj.get("coding", [])
        test_name = code_obj.get("text", "")
        loinc = _coding_code(codings, "http://loinc.org")
        cat_code = ""
        for cat in obs.get("category", []):
            cat_code = _coding_code(
                cat.get("coding", []),
                "http://terminology.hl7.org/CodeSystem/observation-category"
            )
        category = "vital-signs" if cat_code == "vital-signs" else "laboratory"

        # Value
        vq       = obs.get("valueQuantity", {})
        val_qty  = vq.get("value")
        val_unit = vq.get("unit", "")
        val_str  = (obs.get("note") or [{}])[0].get("text", "") or obs.get("valueString", "")

        tests.append(MedicalTest(
            test_name=test_name,
            loinc_code=loinc,
            category=category,
            value_quantity=val_qty,
            value_unit=val_unit,
            value_string=val_str,
        ))
    return tests


def _extract_medications(bundle: dict) -> list[Medication]:
    meds = []
    for mr in _resources(bundle, "MedicationRequest"):
        med_cc   = mr.get("medicationCodeableConcept", {})
        atc_code = _coding_code(med_cc.get("coding", []), "http://www.whocc.no/atc")
        full_text = med_cc.get("text", "")        # "Name dose"

        dosage_text = ""
        for di in mr.get("dosageInstruction", []):
            dosage_text = di.get("text", "")
            break

        dose, route, frequency = _parse_dosage(dosage_text)
        med_name = full_text.removesuffix(dose).strip() if dose else full_text

        if not route:
            route = "oral"

        meds.append(Medication(
            medication_name=med_name or full_text,
            atc_code=atc_code,
            dose=dose,
            route=route,
            frequency=frequency,
        ))
    return meds


def _extract_allergies(bundle: dict) -> list[Allergy]:
    allergies = []
    for a in _resources(bundle, "AllergyIntolerance"):
        code_obj = a.get("code", {})
        name     = code_obj.get("text", "")
        atc_code = _coding_code(code_obj.get("coding", []), "http://www.whocc.no/atc")
        reaction = a.get("reaction", [{}])[0]
        manifst  = reaction.get("manifestation", [{}])[0].get("text", "")
        severity = reaction.get("severity", "moderate")

        allergies.append(Allergy(
            substance_name=name,
            substance_code=atc_code,
            allergy_type=a.get("type", "allergy"),
            category=a.get("category", ["medication"])[0],
            criticality=a.get("criticality", "unable-to-assess"),
            reaction=manifst,
            severity=severity,
        ))
    return allergies


def _extract_family_history(bundle: dict) -> list[FamilyHistory]:
    history = []
    for fh in _resources(bundle, "FamilyMemberHistory"):
        rel_code = fh.get("relationship", {}).get("coding", [{}])[0].get("code", "FTH")
        deceased = fh.get("deceasedBoolean", False)
        cond = fh.get("condition", [{}])[0] if fh.get("condition") else {}
        cond_code_obj = cond.get("code", {})
        icd  = _coding_code(cond_code_obj.get("coding", []), "http://hl7.org/fhir/sid/icd-10")
        name = cond_code_obj.get("text", "")
        onset = cond.get("onsetAge", {}).get("value")

        history.append(FamilyHistory(
            relationship=rel_code,
            condition_icd=icd,
            condition_name=name,
            deceased=deceased,
            onset_age=onset,
        ))
    return history


def _extract_procedures(bundle: dict) -> list[MedicalProcedure]:
    procedures = []
    for proc in _resources(bundle, "Procedure"):
        code_obj = proc.get("code", {})
        name     = code_obj.get("text", "")
        snomed   = _coding_code(code_obj.get("coding", []), "http://snomed.info/sct")
        year_raw = proc.get("performedDateTime", "")
        try:
            year = int(str(year_raw)[:4]) if year_raw else None
        except ValueError:
            year = None
        note = (proc.get("note") or [{}])[0].get("text", "")

        procedures.append(MedicalProcedure(
            procedure_name=name,
            snomed_code=snomed,
            status=proc.get("status", "completed"),
            performed_year=year,
            note=note,
        ))
    return procedures


# ── Main ─────────────────────────────────────────────────────────────

def bundle_to_llm_output(bundle: dict) -> PatientOutput:
    """Converts FHIR bundle dict to PatientOutput."""
    diag = _extract_diagnosis(bundle)
    return PatientOutput(
        patient_description   = _communication_text(bundle, "patient-description"),
        diagnosis_description = diag.get("diagnosis_description", ""),
        patient_history       = _communication_text(bundle, "patient-history"),
        medical_tests         = _extract_tests(bundle),
        medication            = _extract_medications(bundle),
        allergies             = _extract_allergies(bundle),
        family_history        = _extract_family_history(bundle),
        procedures            = _extract_procedures(bundle),
    )


def bundle_to_eval_input(bundle: dict) -> FullPatient:
    """Converts FHIR bundle dict to EvalInput."""
    info = _extract_patient_info(bundle)
    diag = _extract_diagnosis(bundle)
    patient_output = bundle_to_llm_output(bundle)
    return FullPatient(
        patient        = patient_output,
        icd_code       = diag.get("icd_code", ""),
        diagnosis_name = diag.get("diagnosis_name", ""),
        gender         = info.get("gender", ""),
        age            = info.get("age", 0),
        name           = info.get("patient_name", ""),
    )


def bundle_from_file(path: str | Path) -> dict:
    """Read FHIR bundle from file."""
    print(path)
    with (open(path, "r", encoding="utf-8") as f):
        data = json.load(f)
    return data


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bundle_to_llm.py <bundle.json>")
        sys.exit(1)

    bundle = bundle_from_file(sys.argv[1])
    output = bundle_to_llm_output(bundle)
    print(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))
