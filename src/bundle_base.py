import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from fhir.resources.R4B.bundle import Bundle
# FHIR R4 BUNDLE
empty_bundle = {
    "resourceType": "Bundle",
    "id": "virtual-patient-001",
    "type": "collection",
    "meta": {
        "tag": [
            {"system": "urn:fhir:tags", "code": "age", "display": "68"}
        ]
    },
    "entry": [
        # RESOURCE 1: Patient
        {
            "fullUrl": "urn:uuid:11111111-1111-4111-8111-111111111111",
            "resource": {
                "resourceType": "Patient",
                "id": "patient-001",
                "text": {
                    "status": "generated",
                    "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">Basic patient demographic data.</div>",
                },
                "name": [{"use": "official", "family": "Novak", "given": ["Jan"]}],
                "gender": "male",
            },
        },
        # RESOURCE 2: Condition - primary diagnosis
        {
            "fullUrl": "urn:uuid:22222222-2222-4222-8222-222222222222",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active",
                        }
                    ]
                },
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10",
                            "code": "I21.01",
                            "display": "ST elevation myocardial infarction involving left anterior descending coronary artery",
                        }
                    ]
                },
                "subject": {"reference": "urn:uuid:11111111-1111-4111-8111-111111111111"},
                "onsetDateTime": "2026-04-04T06:15:00Z",
            },
        },
    ],
}

class BundleBase:
    def __init__(self, diagnosis_code: str, diagnosis_name: str, patient_age: int, patient_sex: str, patient_name: str, patient_family_name: str, patient_country: str):
        self.bundle = empty_bundle.copy()
        self.bundle["entry"][0]["resource"]["id"] = "virtual-patient-001"
        self.bundle["entry"][0]["resource"]["gender"] = patient_sex
        self.bundle["meta"]["tag"][0]["display"] = str(patient_age)
        self.bundle["entry"][0]["resource"]["name"][0]["given"] = [patient_name]
        self.bundle["entry"][0]["resource"]["name"][0]["family"] = patient_family_name
        self.bundle["entry"][1]["resource"]["code"]["coding"][0]["code"] = diagnosis_code
        self.bundle["entry"][1]["resource"]["code"]["coding"][0]["display"] = diagnosis_name
        self.patient_country = patient_country

    def __str__(self) -> str:
        return  f"Patient name:\t\t {self.bundle['entry'][0]['resource']['name'][0]['given'][0]} {self.bundle['entry'][0]['resource']['name'][0]['family']}\n" \
                f"Date of birth:\t\t {self.bundle["meta"]["tag"][0]["display"]}\n" \
                f"Gender:\t\t\t {self.bundle['entry'][0]['resource']['gender']}\n" \
                f"Diagnosis code:\t\t {self.bundle['entry'][1]['resource']['code']['coding'][0]['code']}\n" \
                f"Diagnosis description:\t {self.bundle['entry'][1]['resource']['code']['coding'][0]['display']}\n"

    def validate_bundle_structure(self) -> bool:

        try:
            bundle = Bundle.parse_obj(self.bundle)
            print("OK:", bundle.type, f"– {len(bundle.entry)} entries")
            return True
        except Exception as e:
            print("ERROR:", e)
            return False

    def validate_bundle_all(self) -> tuple[bool, list[str], list[str]]:
        print("Validating bundle...")
        validator_jar_path = Path(".fhir/validator_cli.jar")
        fhir_version = "4.0.1"
        """
        Validate Bundle with the official HL7 FHIR Validator CLI.
        Returns (is_valid, validator_output).
        """
        java_bin = shutil.which("java")
        if java_bin is None:
            return False, "Java was not found on PATH. Install Java (JRE/JDK 11+)."

        if not validator_jar_path.exists():
            print("Downloading validator_cli.jar...")
            validator_jar_path.parent.mkdir(parents=True, exist_ok=True)
            # Official HL7 validator binary from org.hl7.fhir.core releases.
            url = "https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar"
            urllib.request.urlretrieve(url, validator_jar_path)
            print("validator_cli.jar downloaded successfully.")

        
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(self.bundle, f)
            f.flush()
            #print(f"Validating bundle with {fhir_version}...")
            result = subprocess.run(["java", "-Djavax.net.ssl.trustStoreType=WINDOWS-ROOT", "-jar", validator_jar_path.absolute(), f.name, "-version", fhir_version], capture_output=True, text=True, encoding="utf-8")
        ok = True
        #print(result.stdout)
        errors   = [l.strip() for l in result.stdout.splitlines() if "Error" in l]
        warnings = [l.strip() for l in result.stdout.splitlines() if "Warning" in l]
        if len(errors) > 0 or len(warnings) > 0:
            ok = False
        return ok, errors, warnings


if __name__ == "__main__":
    with open(Path().parent/"scenarios/J00_6_PeterStejskal.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    bb = BundleBase(diagnosis_code="dd", diagnosis_name="d", patient_name="s", patient_family_name="s", patient_country="cz", patient_sex="F", patient_age=20)
    bb.bundle = data
    bb.validate_bundle_all()