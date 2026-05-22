"""
Fetch relevant doctor-patient dialogues from HuggingFace via the Datasets
Server REST API (no huggingface_hub / datasets library required).

Source: lavita/ChatDoctor-HealthCareMagic-100k
  100k real patient questions + doctor answers from healthcaremagic.com.
  Fields: instruction (fixed), input (patient), output (doctor).

We paginate through the dataset using the public REST API and filter by
keyword relevance for the given diagnosis, keeping the top N matches.
Results are cached locally — subsequent calls return immediately.

Output: rag/data/meddialog/<icd_code>.jsonl
Each line:
  {"id": "MEDDIALOG:<icd>:<rank>", "source": "meddialog",
   "icd10": "<icd>", "title": "<patient question[:100]>",
   "text": "Patient: ...\nDoctor: ...", "url": "<hf_url>"}

Usage
-----
  python rag/ingest/fetch_meddialog.py E11.9 "type 2 diabetes"
  python rag/ingest/fetch_meddialog.py           # runs DEFAULT list
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "meddialog"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET  = "lavita/ChatDoctor-HealthCareMagic-100k"
HF_CONFIG   = "default"
HF_SPLIT    = "train"
MAX_SCAN    = 10_000   # max records to page through (100 API pages × 100 rows)
MAX_RESULTS = 8        # dialogues to keep per ICD code


# Minimal fallback for standalone use — setup.py drives the real seed list.
DEFAULT: list[tuple[str, str]] = []


def _keywords(icd_code: str, diagnosis_name: str) -> list[str]:
    """Build a list of lowercase keywords to match against dialogue text."""
    name = diagnosis_name.lower()
    words = [w for w in re.split(r"[\s,\-/]+", name) if len(w) > 3]
    # add common synonyms / abbreviations
    extras: dict[str, list[str]] = {
        "J00":   ["cold", "runny nose", "sore throat", "nasopharyngitis"],
        "E11.9": ["diabetes", "blood sugar", "insulin", "glucose", "hba1c"],
        "I21.9": ["heart attack", "myocardial", "chest pain", "infarction"],
        "F32.9": ["depression", "depressed", "antidepressant", "sadness"],
        "M54.5": ["back pain", "lumbar", "spine"],
        "N39.0": ["uti", "urinary", "bladder", "dysuria"],
        "J18.9": ["pneumonia", "lung infection", "cough", "respiratory"],
        "K21.9": ["reflux", "gerd", "heartburn", "acid"],
        "G43.9": ["migraine", "headache"],
        "C34.9": ["lung cancer", "pulmonary", "chest mass"],
    }
    return list(set(words + extras.get(icd_code, [])))


def _score(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in the text."""
    low = text.lower()
    return sum(1 for kw in keywords if kw in low)


def _format_dialogue(record: dict) -> str:
    """Format a ChatDoctor record into readable patient-doctor dialogue."""
    patient = record.get("input", "").strip()
    doctor  = record.get("output", "").strip()
    lines = []
    if patient:
        lines.append(f"Patient: {patient}")
    if doctor:
        lines.append(f"Doctor: {doctor}")
    return "\n".join(lines)


def fetch_for_icd10(
    icd_code: str,
    diagnosis_name: str,
    max_results: int = MAX_RESULTS,
    max_scan: int = MAX_SCAN,
) -> Path:
    """Stream MedDialog, pick the best matching dialogues, save to JSONL.

    Returns the output file path.
    Raises RuntimeError if the datasets library is not installed.
    """
    # Use the HuggingFace Datasets Server REST API — plain requests, no HF library,
    # no SSL issues.  Paginates through rows until enough matches are found.
    import requests as _req
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    ROWS_API = "https://datasets-server.huggingface.co/rows"
    PAGE     = 100   # rows per API call
    MAX_PAGES = max_scan // PAGE

    out = DATA_DIR / f"{icd_code.replace('.', '_')}.jsonl"
    if out.exists() and out.stat().st_size > 50:
        print(f"[meddialog] {icd_code}: already present ({out.name}), skipping")
        return out

    keywords = _keywords(icd_code, diagnosis_name)
    print(f"[meddialog] {icd_code}: querying HF Datasets Server REST API "
          f"(keywords: {keywords[:5]}...)")

    candidates: list[tuple[int, dict]] = []
    scanned = 0

    for page in range(MAX_PAGES):
        offset = page * PAGE
        try:
            r = _req.get(
                ROWS_API,
                params={"dataset": HF_DATASET, "config": HF_CONFIG,
                        "split": HF_SPLIT, "offset": offset, "length": PAGE},
                timeout=30,
                verify=False,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[meddialog]   page {page} error: {e}")
            break

        rows = data.get("rows", [])
        if not rows:
            print(f"[meddialog]   dataset exhausted at offset {offset}")
            break

        for row_obj in rows:
            record = row_obj.get("row", {})
            scanned += 1

            # ChatDoctor format: input=patient question, output=doctor answer
            full_text = record.get("input", "") + " " + record.get("output", "")
            score = _score(full_text, keywords)
            if score > 0:
                candidates.append((score, record))

        # Early-exit once we have plenty of candidates
        if len(candidates) >= max_results * 5:
            break

    print(f"[meddialog]   scanned {scanned} records, {len(candidates)} candidates")

    # Pick top N by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:max_results]

    if not top:
        print(f"[meddialog] {icd_code}: no matching dialogues found")
        # Write empty file so we don't retry endlessly
        out.write_text("", encoding="utf-8")
        return out

    docs: list[dict] = []
    for rank, (score, record) in enumerate(top):
        text  = _format_dialogue(record)
        title = record.get("input", "")[:100].strip()
        docs.append({
            "id":     f"MEDDIALOG:{icd_code}:{rank}",
            "source": "meddialog",
            "icd10":  icd_code,
            "title":  title,
            "text":   f"Doctor-patient dialogue for {diagnosis_name}:\n\n{text}",
            "url":    f"https://huggingface.co/datasets/{HF_DATASET}",
            "score":  score,
        })

    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[meddialog] {icd_code}: saved {len(docs)} dialogues -> {out.name}")
    return out


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 2:
        targets = [(args[0], " ".join(args[1:]))]
    elif len(args) == 1:
        targets = [(args[0], args[0])]
    else:
        targets = DEFAULT

    for code, name in targets:
        try:
            fetch_for_icd10(code, name)
        except Exception as e:
            print(f"[meddialog] {code}: ERROR {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
