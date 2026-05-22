import copy, uuid
from datetime import date, datetime, timezone


def _urn():
    return f"urn:uuid:{uuid.uuid4()}"


def _div(text):
    return {"status": "generated", "div": f'<div xmlns="http://www.w3.org/1999/xhtml">{text}</div>'}


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _patient_ref(bundle):
    for e in bundle["entry"]:
        if e["resource"]["resourceType"] == "Patient":
            return e["fullUrl"]


def _remove(bundle, *types):
    bundle["entry"] = [e for e in bundle["entry"] if e["resource"]["resourceType"] not in set(types)]


def _first(bundle, rtype):
    for e in bundle["entry"]:
        if e["resource"]["resourceType"] == rtype:
            return e["resource"]


_COMPANION = {"59408-5": "2708-6", "3150-0": "2708-6"}
_DATA_ABSENT = {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/data-absent-reason", "code": "unknown"}]}

_REL_DISPLAY = {
    "FTH": "Father", "MTH": "Mother", "BRO": "Brother", "SIS": "Sister",
    "SON": "Son", "DAU": "Daughter", "GRFTH": "Grandfather", "GRMTH": "Grandmother",
    "UNCLE": "Uncle", "AUNT": "Aunt", "COUSN": "Cousin",
}


def fill_bundle(bundle, llm):
    b = copy.deepcopy(bundle)
    ref = _patient_ref(b)

    # Medications
    _remove(b, "Medication", "MedicationRequest")
    for med in llm.get("medication", []):
        name = med["medication_name"]
        b["entry"].append({"fullUrl": _urn(), "resource": {
            "resourceType": "MedicationRequest",
            "status": "active", "intent": "order",
            "text": _div(f"Medication order for {name}."),
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.whocc.no/atc", "code": med.get("atc_code", "")}],
                "text": f"{name} {med['dose']}",
            },
            "subject": {"reference": ref},
            "dosageInstruction": [{"text": f"{med['dose']} {med['route']} {med['frequency']}"}],
        }})

    # Observations
    _remove(b, "Observation")
    for test in llm.get("medical_tests", []):
        loinc = test.get("loinc_code", "")
        is_vital = test.get("category") == "vital-signs"
        val_qty = test.get("value_quantity")
        val_unit = test.get("value_unit", "")
        val_str = test.get("value_string", "")

        cat_code = "vital-signs" if is_vital else "laboratory"
        cat = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": cat_code}]}]

        codings = []
        if loinc:
            codings.append({"system": "http://loinc.org", "code": loinc})
            if loinc in _COMPANION:
                codings.append({"system": "http://loinc.org", "code": _COMPANION[loinc]})

        obs = {
            "resourceType": "Observation", "status": "final",
            "text": _div(test["test_name"]),
            "category": cat,
            "code": {"coding": codings, "text": test["test_name"]} if codings else {"text": test["test_name"]},
            "subject": {"reference": ref},
            "performer": [{"reference": ref}],
            "effectiveDateTime": _now(),
        }
        if val_qty is not None and val_unit:
            obs["valueQuantity"] = {"value": val_qty, "unit": val_unit, "system": "http://unitsofmeasure.org",
                                    "code": val_unit}
        elif is_vital:
            obs["dataAbsentReason"] = _DATA_ABSENT
        else:
            obs["valueString"] = val_str
        if val_str:
            obs["note"] = [{"text": val_str}]
        b["entry"].append({"fullUrl": _urn(), "resource": obs})

    # Allergies
    _remove(b, "AllergyIntolerance")
    for a in llm.get("allergies", []):
        name = a["substance_name"]
        code = a.get("substance_code", "")
        coding = [{"system": "http://www.whocc.no/atc", "code": code}] if code else []
        b["entry"].append({"fullUrl": _urn(), "resource": {
            "resourceType": "AllergyIntolerance",
            "text": _div(f"Allergy to {name}."),
            "clinicalStatus": {"coding": [
                {"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [
                {"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
                 "code": "confirmed"}]},
            "type": a.get("allergy_type", "allergy"),
            "category": [a.get("category", "medication")],
            "criticality": a.get("criticality", "unable-to-assess"),
            "code": {"coding": coding, "text": name} if coding else {"text": name},
            "patient": {"reference": ref},
            "reaction": [
                {"manifestation": [{"text": a.get("reaction", "")}], "severity": a.get("severity", "moderate")}],
        }})

    # Family history
    _remove(b, "FamilyMemberHistory")
    for fh in llm.get("family_history", []):
        rel = fh.get("relationship", "FTH")
        cond_icd = fh.get("condition_icd", "")
        cond_name = fh.get("condition_name", "")
        entry = {
            "resourceType": "FamilyMemberHistory",
            "status": "completed",
            "text": _div(f"{_REL_DISPLAY.get(rel, rel)}: {cond_name}"),
            "patient": {"reference": ref},
            "relationship": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-RoleCode", "code": rel,
                                         "display": _REL_DISPLAY.get(rel, rel)}]},
            "deceasedBoolean": fh.get("deceased", False),
        }
        if cond_icd or cond_name:
            cond = {"code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": cond_icd}],
                             "text": cond_name} if cond_icd else {"text": cond_name}}
            if fh.get("onset_age"):
                cond["onsetAge"] = {"value": fh["onset_age"], "unit": "years", "system": "http://unitsofmeasure.org",
                                    "code": "a"}
            entry["condition"] = [cond]
        b["entry"].append({"fullUrl": _urn(), "resource": entry})

    # Procedures
    _remove(b, "Procedure")
    for proc in llm.get("procedures", []):
        name = proc["procedure_name"]
        snomed = proc.get("snomed_code", "")
        year = proc.get("performed_year")
        coding = [{"system": "http://snomed.info/sct", "code": snomed}] if snomed else []
        entry = {
            "resourceType": "Procedure",
            "status": proc.get("status", "completed"),
            "text": _div(name),
            "code": {"coding": coding, "text": name} if coding else {"text": name},
            "subject": {"reference": ref},
        }
        if year:
            entry["performedDateTime"] = str(year)
        if proc.get("note"):
            entry["note"] = [{"text": proc["note"]}]
        b["entry"].append({"fullUrl": _urn(), "resource": entry})

    # Text descriptions
    for key, code in [("patient_description", "patient-description"), ("patient_history", "patient-history")]:
        text = llm.get(key, "")
        if text:
            b["entry"].append({"fullUrl": _urn(), "resource": {
                "resourceType": "Communication", "status": "completed",
                "text": _div(code),
                "category": [{"coding": [{"system": "urn:virpat:fhir:doc-type", "code": code}]}],
                "subject": {"reference": ref},
                "payload": [{"contentString": text}],
            }})

    return b