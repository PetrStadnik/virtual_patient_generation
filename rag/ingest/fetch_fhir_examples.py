"""
Fetch real, canonical FHIR R4 example Bundles from the public HL7 server.

These are the same example resources shipped with the FHIR R4 specification,
hosted on https://hl7.org/fhir/R4/. They are free to redistribute and excellent
for grounding scenario generation in the exact JSON shapes you already produce
in src/bundle_base.py.

For a *clinically* richer synthetic dataset we recommend running Synthea
locally (see rag/README.md), but Synthea is heavy (Java + multi-GB output) so
we don't fetch it inside this demo.

Output: rag/data/fhir/<name>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fhir"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Each entry is a small but realistic FHIR R4 example.
EXAMPLES: dict[str, str] = {
    # 70-year-old patient with ischaemic stroke; full bundle with conditions,
    # medications, observations, careplan.
    "patient_example_pat1": "https://hl7.org/fhir/R4/patient-example.json",
    # Diabetic encounter with conditions and observations.
    "condition_example_diabetes":
        "https://hl7.org/fhir/R4/condition-example2.json",
    # Hypertension observation pattern.
    "observation_example_bp":
        "https://hl7.org/fhir/R4/observation-example-bloodpressure.json",
    # MI / cardiology condition.
    "condition_example_stroke":
        "https://hl7.org/fhir/R4/condition-example-stroke.json",
    # Encounter resource showing typical hospital visit shape.
    "encounter_example_inpatient":
        "https://hl7.org/fhir/R4/encounter-example.json",
}


def main() -> int:
    for name, url in EXAMPLES.items():
        try:
            r = requests.get(url, timeout=20, verify=False,
                             headers={"Accept": "application/fhir+json"})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[fhir] {name}: ERROR {e}")
            continue
        out = DATA_DIR / f"{name}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        rt = data.get("resourceType", "?")
        print(f"[fhir] {name}: {rt} -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
