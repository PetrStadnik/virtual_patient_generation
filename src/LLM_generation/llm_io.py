import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional


class FullPatient(BaseModel):
    patient: PatientOutput
    icd_code: str
    diagnosis_name: str
    gender: str
    age: int
    name: str

# --- LLM input structure ---
class PatientInput(BaseModel):
    icd_code:       str   
    diagnosis_name: str   
    gender:         str  
    age:            int

_ATC_RE   = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")
_LOINC_RE = re.compile(r"^\d{3,6}-\d$")
_ICD_RE   = re.compile(r"^[A-Z]\d{2}(\.\d)?$")
 
# --- Medical test structure ---
class MedicalTest(BaseModel):
    test_name: str = Field(description="Full clinical name, e.g. 'Esophageal pH monitoring'")
    loinc_code: str = Field(description="LOINC code NNNNN-N, e.g. '35777-0'", min_length=5, max_length=7)
    category: Literal["laboratory", "vital-signs"] = Field(description="'vital-signs' for vitals (temp, SpO2, HR, BP, weight, height, RR). 'laboratory' for all others.")
    value_quantity: Optional[float] = Field(default=None, description="Numeric result, e.g. 36.8. Required for vital signs.")
    value_unit: str = Field(default="", description="UCUM unit, e.g. 'Cel', '%', 'kg', '/min', 'mm[Hg]'.")
    value_string: str = Field(description="Human-readable result text, always provide.")

    @field_validator("loinc_code")
    @classmethod
    def validate_loinc(cls, v):
        if v and not _LOINC_RE.match(v):
            raise ValueError(f"LOINC '{v}' does not match NNNNN-N")
        return v
 
# --- Medication structure ---
class Medication(BaseModel):
    medication_name: str = Field(description="Generic (INN) drug name, e.g. 'Omeprazole'")
    atc_code: str = Field(description="WHO ATC code in format A00AA00, e.g. 'A02BC01'. Empty string if not found.")
    dose: str = Field(description="Dose with unit, e.g. '20 mg'")
    route: Literal["oral", "intravenous", "subcutaneous", "intramuscular","topical", "inhaled", "rectal", "transdermal"] = Field(description="Route of administration")
    frequency: str = Field(description="Dosing frequency, e.g. 'once daily', 'twice daily', 'every 8 hours'")
 
    @field_validator("atc_code")
    @classmethod
    def validate_atc(cls, v: str) -> str:
        if v and not _ATC_RE.match(v):
            raise ValueError(f"ATC code '{v}' does not match pattern A00AA00")
        return v
 
    @field_validator("medication_name")
    @classmethod
    def strip_brand_names(cls, v: str) -> str:
        return re.sub(r"\s*\(.*?\)", "", v).strip()

# Procedure structure
class MedicalProcedure(BaseModel):
    procedure_name: str = Field(description="Name of the procedure, e.g. 'Appendectomy'")
    snomed_code: str = Field(description="SNOMED CT procedure code. Empty if not found.")
    status: Literal["completed", "not-done", "in-progress"] = Field(default="completed")
    performed_year: Optional[int] = Field(default=None, description="Approximate year performed, e.g. 2015", min_value=1900, max_value=datetime.now().year)
    note: str = Field(description="Brief clinical note about the procedure")

# Allergies structure
class Allergy(BaseModel):
    substance_name: str = Field(description="Name of the allergen, e.g. 'Amoxicillin', 'Peanut'")
    substance_code: str = Field(description="ATC code for medications, empty for food/environment.")
    allergy_type: Literal["allergy", "intolerance"] = Field(description="'allergy' for immune-mediated, 'intolerance' for non-immune")
    category: Literal["food", "medication", "environment", "biologic"]
    criticality: Literal["low", "high", "unable-to-assess"]
    reaction: str = Field(description="Reaction description, e.g. 'Anaphylaxis', 'Urticaria', 'Nausea'")
    severity: Literal["mild", "moderate", "severe"]

    @field_validator("substance_code")
    @classmethod
    def validate_atc(cls, v):
        if v and not _ATC_RE.match(v):
            raise ValueError(f"ATC '{v}' does not match A00AA00")
        return v


class FamilyHistory(BaseModel):
    relationship: Literal["FTH", "MTH", "BRO", "SIS", "SON", "DAU", "GRFTH", "GRMTH", "UNCLE", "AUNT", "COUSN"] = Field(description="Relationship code: FTH=father, MTH=mother, BRO=brother, SIS=sister, SON=son, DAU=daughter, GRFTH=grandfather, GRMTH=grandmother")
    condition_icd: str = Field( description="ICD-10 code of condition, e.g. 'C50' or 'I25.1'", min_length=3, max_length=5)
    condition_name: str = Field(description="Condition name, e.g. 'Atherosclerotic heart disease'")
    deceased: bool
    onset_age: Optional[int] = Field(default=None, description="Age of onset in years, e.g. 61")

    @field_validator("condition_icd")
    @classmethod
    def validate_icd(cls, v):
        if v and not _ICD_RE.match(v):
            raise ValueError(f"ICD-10 '{v}' invalid format")
        return v

# --- LLM output structure ---
class PatientOutput(BaseModel):
    patient_description: str = Field(description=("3-5 paragraph vivid description of the patient's appearance, symptoms and full lifestyle, hobbies, work info, family and frinds. Use {patientName} as placeholder for the patient's name."))
    diagnosis_description: str = Field(description="Clinical description of the diagnosis including pathophysiology and typical symptoms")
    patient_history: str = Field(description=("Relevant medical history: comorbidities, prior hospitalizations, lifestyle factors. Use {patientName} as placeholder."))
    medical_tests: list[MedicalTest] = Field(min_length=2, max_length=5, description="2-5 diagnostic tests appropriate for the diagnosis, each with LOINC code")
    medication: list[Medication] = Field(min_length=1, max_length=6, description="1-6 medications appropriate for the diagnosis and patient profile, each with ATC code")
    allergies: list[Allergy] = Field(min_length=0, max_length=4, description="0-4 allergies. It can be various normal allergies.")
    family_history: list[FamilyHistory] = Field(min_length=0, max_length=4, description="0-4 relevant family history entries with ICD-10 codes.")
    procedures: list[MedicalProcedure] = Field(min_length=0, max_length=4, description="0-4 past procedures with SNOMED codes.")