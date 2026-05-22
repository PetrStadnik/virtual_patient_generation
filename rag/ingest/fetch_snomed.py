"""
Pull SNOMED CT clinical findings (symptoms) per ICD-10 demo code by reusing
the existing src.symptoms_snomed module — no need to re-implement BioPortal.

This produces small, structured documents that fit well in a RAG index because
they enumerate the clinically valid finding labels for each disease.

Output: rag/data/snomed/<code>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make project root importable regardless of how this script is launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.symptoms_snomed import get_symptoms_snomed  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "snomed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimal fallback for standalone use — setup.py drives the real seed list.
DEFAULT: list[tuple[str, str]] = []


def fetch_one(code: str, name: str) -> None:
    symptoms = get_symptoms_snomed(code, name)
    doc = {
        "id": f"SNOMED:{code}",
        "source": "snomed",
        "icd10": code,
        "diagnosis": name,
        "symptoms": symptoms,
        "url": "https://browser.ihtsdotools.org/",
    }
    out = DATA_DIR / f"{code.replace('.', '_')}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[snomed] {code} ({name}): {len(symptoms)} findings -> {out.name}")


def main() -> int:
    codes_arg = sys.argv[1:]
    targets = [(c, n) for c, n in DEFAULT if c in codes_arg] if codes_arg else DEFAULT
    for code, name in targets:
        fetch_one(code, name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
