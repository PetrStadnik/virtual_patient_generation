"""
ICD-10 -> GBD Cause ID mapping + demographic probability distributions
"""
import random
import re

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

MAPPING_FILE    = Path(__file__).parent.parent.parent / "data" / "IHME_GBD_2021_NONFATAL_CAUSE_ICD_CODE_MAP_Y2024M05D16_0.XLSX"
PREVALENCE_FILE = Path(__file__).parent.parent.parent / "data" / "IHME-GBD_2023_DATA-e683deec-1.csv"


@dataclass
class GbdCause:
    cause_id:   int
    cause_name: str
    level:      int = 4   # GBD Cause Hierarchy Level (3 or 4)
 
 
@dataclass
class _Range:
    start_norm: str
    end_norm:   str
    cause:      GbdCause
 
 
def _norm(code: str) -> str:
    return code.strip().upper().replace(".", "")
 
 
def _parse_icd10_field(raw: str, cause: GbdCause) -> list:
    ranges = []
    for entry in re.split(r",\s*", raw.strip()):
        entry = entry.strip()
        if not entry:
            continue
        parts = re.split(r"-(?=[A-Z])", entry, maxsplit=1)
        start, end = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], parts[0])
        ranges.append(_Range(_norm(start), _norm(end), cause))
    return ranges
 
 
def _load_ranges_for_level(df: pd.DataFrame, level: float) -> list:
    """Load ICD-10 ranges for a single GBD hierarchy level."""
    subset = df[df["Cause Hierarchy Level"] == level].dropna(subset=["ICD10"])
    ranges = []
    for _, row in subset.iterrows():
        cause = GbdCause(int(row["Cause ID"]), str(row["Cause Name"]).strip(), int(level))
        ranges.extend(_parse_icd10_field(str(row["ICD10"]), cause))
    return ranges
 
 
def _load_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df = df[pd.to_numeric(df["Cause ID"], errors="coerce").notna()].copy()
    df["Cause ID"] = pd.to_numeric(df["Cause ID"]).astype(int)
    return df
 
 
class IcdToGbd:
    """Maps ICD-10 codes to GBD Cause.
 
    Lookup priority: Level 4 first, Level 3 as fallback.
    """
 
    def __init__(self, path: Path = MAPPING_FILE):
        df = _load_excel(path)
        self._ranges4 = _load_ranges_for_level(df, 4.0)
        self._ranges3 = _load_ranges_for_level(df, 3.0)
        n4 = len({r.cause.cause_id for r in self._ranges4})
        n3 = len({r.cause.cause_id for r in self._ranges3})
        #print(f"Loaded {len(self._ranges4)} ranges from {n4} GBD causes (Level 4).")
        #print(f"Loaded {len(self._ranges3)} ranges from {n3} GBD causes (Level 3, fallback).")
 
    def lookup(self, icd_code: str) -> Optional[GbdCause]:
        """Return best-matching GBD cause: Level 4 preferred, Level 3 as fallback."""
        code_norm = _norm(icd_code)
        if not code_norm:
            return None
        for r in self._ranges4:
            if r.start_norm <= code_norm <= r.end_norm:
                return r.cause
        for r in self._ranges3:
            if r.start_norm <= code_norm <= r.end_norm:
                return r.cause
        return None
 
    def cause_id(self, icd_code: str) -> Optional[int]:
        result = self.lookup(icd_code)
        return result.cause_id if result else None
 
 
def load_prevalence(path: Path = PREVALENCE_FILE) -> pd.DataFrame:
    """Load GBD prevalence CSV."""
    df = pd.read_csv(path)
    df["cause_id"] = pd.to_numeric(df["cause_id"], errors="coerce")
    df["val"]      = pd.to_numeric(df["val"],      errors="coerce").fillna(0.0)
    df["age_id"]   = pd.to_numeric(df["age_id"],   errors="coerce")
    df["sex_id"]   = pd.to_numeric(df["sex_id"],   errors="coerce")
    return df
 
 
def get_demographic_probs(
    icd_code: str,
    prev_df: pd.DataFrame,
    mapper: IcdToGbd,
) -> tuple:
    """
    Returns (age_prob, sex_prob) DataFrames for the given ICD-10 code.
 
    age_prob columns: age_id, age, probability  (sorted by age_id)
    sex_prob columns: sex, probability
 
    sex_id 3 (Both) is excluded to avoid double-counting.
    """
    cause = mapper.lookup(icd_code)
    if cause is None:
        print(f"ICD code '{icd_code}' not found in GBD mapping (Level 4 or 3).")
        return None, None
    else:
        print(f"GBD Cause: {cause.cause_id} - {cause.cause_name} (Level {cause.level})\n")
    df = prev_df[
        (prev_df["cause_id"] == cause.cause_id) &
        (prev_df["sex_id"].isin([1, 2]))
    ].copy()
 
    if df.empty:
        print(f"No prevalence data for cause_id={cause.cause_id} ({cause.cause_name}).")
        return None, None
 
    # Age probabilities: sum val across sexes for each age group, then normalise
    age_agg = (
        df.groupby(["age_id", "age_name"], sort=False)["val"]
        .sum()
        .reset_index()
        .sort_values("age_id")
    )
    total_age = age_agg["val"].sum()
    age_agg["probability"] = age_agg["val"] / total_age if total_age > 0 else 0.0
    age_prob = age_agg[["age_id", "age_name", "probability"]].rename(
        columns={"age_name": "age"}
    )
 
    # Sex probabilities: sum val across age groups for each sex, then normalise
    sex_agg = (
        df.groupby(["sex_id", "sex_name"], sort=False)["val"]
        .sum()
        .reset_index()
    )
    total_sex = sex_agg["val"].sum()
    sex_agg["probability"] = sex_agg["val"] / total_sex if total_sex > 0 else 0.0
    sex_prob = sex_agg[["sex_name", "probability"]].rename(columns={"sex_name": "sex"})
 
    return age_prob, sex_prob
 
 
def sample_demographics(
    icd_code: str,
    prev_df: pd.DataFrame,
    mapper: IcdToGbd,
) -> Optional[dict]:
    """
    Sample a single (age_group, sex) pair from the prevalence distribution.
    Use this in the virtual patient generation pipeline.
    Returns dict with age_group, sex, cause_id, cause_name, level or None.
    """
    age_prob, sex_prob = get_demographic_probs(icd_code, prev_df, mapper)
    if age_prob is None:
        return None
    age_row = age_prob.sample(weights="probability").iloc[0]
    sex_row = sex_prob.sample(weights="probability").iloc[0]
    cause = mapper.lookup(icd_code)
    return {
        "age_group":  age_row["age"],
        "sex":        sex_row["sex"],
        "cause_id":   cause.cause_id,
        "cause_name": cause.cause_name,
        "level":      cause.level,
    }
 
 
def save_plot(age_prob: pd.DataFrame, sex_prob: pd.DataFrame, output_path: str, dataset_code: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order_map = {
        "0-6 days": 1,
        "7-27 days": 2,
        "1-5 months": 3,
        "6-11 months": 4,
        "12-23 months": 5,
        "2-4 years": 6,
        "5-9 years": 7,
        "10-14 years": 8,
        "15-19 years": 9,
        "20-24 years": 10,
        "25-29 years": 11,
        "30-34 years": 12,
        "35-39 years": 13,
        "40-44 years": 14,
        "45-49 years": 15,
        "50-54 years": 16,
        "55-59 years": 17,
        "60-64 years": 18,
        "65-69 years": 19,
        "70-74 years": 20,
        "75-79 years": 21,
        "80-84 years": 22,
        "85-89 years": 23,
        "90-94 years": 24,
        "95+ years": 25
    }
    age_plot = age_prob.sort_values(by="age", key=lambda x: x.map(order_map))
    axes[0].barh(age_plot["age"], age_plot["probability"], color="#4e79a7")
    axes[0].set_title(f"P(age | ICD-10: {dataset_code})")
    axes[0].set_xlabel("Probability")
 
    axes[1].bar(sex_prob["sex"], sex_prob["probability"], color="#f28e2b")
    axes[1].set_title(f"P(sex | ICD-10: {dataset_code})")
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(0, 1)
 
    fig.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
 
def get_max_probable_age_ang_gender(icd_code: str):
    mapper = IcdToGbd()
    prev_df = load_prevalence(PREVALENCE_FILE)
    age_prob, sex_prob = get_demographic_probs(icd_code, prev_df, mapper)

    random_age, gender = None, None

    if age_prob is None:
        print(f"Could not get age and gender data.")
        return None, None
    else:
        most_probable_age = age_prob.loc[age_prob["probability"].idxmax(), "age"]
        if "95+" in most_probable_age:
            random_age = random.randint(95, 100)
        elif "months" in most_probable_age:
            random_age = random.randint(1, 2)
        elif "days" in most_probable_age:
            random_age = 1
        elif "years" in most_probable_age:
            start, end = map(int, re.findall(r"\d+", most_probable_age))
            random_age = random.randint(start, end)

        gender = random.choices(sex_prob["sex"], weights=sex_prob["probability"], k=1)[0]

        return int(random_age), str(gender).lower()


if __name__ == "__main__":
    mapper = IcdToGbd()
    csv_path = PREVALENCE_FILE
    icd_code = "C50"
 
    prev_df = load_prevalence(csv_path)
    n_causes = prev_df["cause_id"].nunique()
    print(f"Prevalence data: {len(prev_df)} rows, {n_causes} causes.\n")
 
    age_prob, sex_prob = get_demographic_probs(icd_code, prev_df, mapper)
 
    if age_prob is not None:
        cause = mapper.lookup(icd_code)
        print("Age distribution:")
        print(age_prob.to_string(index=False))
        print("\nSex distribution:")
        print(sex_prob.to_string(index=False))
 
        plot_path = str(Path(__file__).parent / f"prevalence_{icd_code.replace('.', '_')}.png")
        save_plot(age_prob, sex_prob, plot_path, icd_code)
        print(f"\nPlot saved: {plot_path}")
 
        print("\nSampled demographics (5x):")
        for _ in range(5):
            print(" ", sample_demographics(icd_code, prev_df, mapper))