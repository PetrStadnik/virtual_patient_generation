"""
RAG data setup — checks every dataset and downloads anything missing.

Run once before using the RAG pipeline:

    python rag/setup.py            # check + auto-download everything
    python rag/setup.py --check    # report status only, no downloads
    python rag/setup.py --rebuild  # force re-download + rebuild index

What it manages
---------------
  rag/data/pubmed/       — PubMed case-report abstracts (NCBI E-utilities, free)
  rag/data/medlineplus/  — MedlinePlus patient summaries (NLM Connect API, free)
  rag/data/snomed/       — SNOMED CT symptom lists (reuses src.symptoms_snomed)
  rag/data/fhir/         — FHIR R4 example resources (hl7.org, free)
  rag/index/             — TF-IDF vector index (built from the data above)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

RAG_ROOT  = PROJECT_ROOT / "rag"
DATA_DIR  = RAG_ROOT / "data"
INDEX_DIR = RAG_ROOT / "index"

# ── Seed codes for initial setup (5 representative diagnoses) ─────────────────
# Dynamic on-demand indexing handles any other ICD-10 code at runtime.
SEED_CODES: list[tuple[str, str]] = [
    ("J00",   "Acute nasopharyngitis [common cold]"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("I21.9", "Acute myocardial infarction, unspecified"),
    ("F32.9", "Major depressive disorder, single episode, unspecified"),
    ("M54.5", "Low back pain"),
]

# ── Colour helpers (graceful fallback on Windows without ANSI) ─────────────────
_ANSI = sys.stdout.isatty()
def _green(s: str)  -> str: return f"\033[32m{s}\033[0m" if _ANSI else s
def _yellow(s: str) -> str: return f"\033[33m{s}\033[0m" if _ANSI else s
def _red(s: str)    -> str: return f"\033[31m{s}\033[0m" if _ANSI else s
def _bold(s: str)   -> str: return f"\033[1m{s}\033[0m"  if _ANSI else s

OK   = _green("OK")
MISS = _yellow("MISSING")
ERR  = _red("ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# Status checks
# ─────────────────────────────────────────────────────────────────────────────

def _count(directory: Path, glob: str) -> int:
    return len(list(directory.glob(glob))) if directory.exists() else 0


def check_status() -> dict[str, dict]:
    """Return a status dict for each dataset and the index."""
    return {
        "pubmed": {
            "dir":   DATA_DIR / "pubmed",
            "files": _count(DATA_DIR / "pubmed", "*.jsonl"),
            "ok":    _count(DATA_DIR / "pubmed", "*.jsonl") > 0,
        },
        "medlineplus": {
            "dir":   DATA_DIR / "medlineplus",
            "files": _count(DATA_DIR / "medlineplus", "*.json"),
            "ok":    _count(DATA_DIR / "medlineplus", "*.json") > 0,
        },
        "snomed": {
            "dir":   DATA_DIR / "snomed",
            "files": _count(DATA_DIR / "snomed", "*.json"),
            "ok":    _count(DATA_DIR / "snomed", "*.json") > 0,
        },
        "meddialog": {
            "dir":   DATA_DIR / "meddialog",
            "files": _count(DATA_DIR / "meddialog", "*.jsonl"),
            "ok":    _count(DATA_DIR / "meddialog", "*.jsonl") > 0,
        },
        "fhir": {
            "dir":   DATA_DIR / "fhir",
            "files": _count(DATA_DIR / "fhir", "*.json"),
            "ok":    _count(DATA_DIR / "fhir", "*.json") > 0,
        },
        "index": {
            "dir":   INDEX_DIR,
            "files": _count(INDEX_DIR, "*"),
            "ok":    (INDEX_DIR / "docs.jsonl").exists() and (INDEX_DIR / "tfidf.pkl").exists(),
        },
    }


def print_status(status: dict[str, dict]) -> None:
    print(_bold("\n=== RAG dataset status ==="))
    rows = [
        ("pubmed",      "rag/data/pubmed/",      "*.jsonl",  "PubMed case reports"),
        ("medlineplus", "rag/data/medlineplus/",  "*.json",   "MedlinePlus summaries"),
        ("snomed",      "rag/data/snomed/",       "*.json",   "SNOMED CT findings"),
        ("meddialog",   "rag/data/meddialog/",   "*.jsonl",  "MedDialog doctor-patient dialogues"),
        ("fhir",        "rag/data/fhir/",         "*.json",   "FHIR R4 examples"),
        ("index",       "rag/index/",             "*",        "TF-IDF vector index"),
    ]
    for key, path, _, desc in rows:
        s    = status[key]
        flag = OK if s["ok"] else MISS
        n    = s["files"]
        unit = "files" if n != 1 else "file"
        print(f"  {flag}  {path:<30}  {n:>3} {unit}   ({desc})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_pubmed() -> bool:
    print(_bold("\n[1/5] Fetching PubMed abstracts..."))
    try:
        from rag.ingest.fetch_pubmed import fetch_query
        for _code, name in SEED_CODES:
            q = name.lower().split(",")[0]  # use first part of diagnosis name as query
            fetch_query(q, retmax=8)
            time.sleep(0.5)
        return True
    except Exception as exc:
        print(f"  {ERR}: {exc}")
        print("  PubMed uses the free NCBI E-utilities API — no key required.")
        print("  If this fails, check your internet connection.")
        return False


def _run_medlineplus() -> bool:
    print(_bold("\n[2/5] Fetching MedlinePlus summaries..."))
    try:
        import json
        from rag.ingest.fetch_medlineplus import fetch_for_icd10, DATA_DIR as ML_DIR
        for code, _name in SEED_CODES:
            try:
                doc = fetch_for_icd10(code)
            except Exception as e:
                print(f"  {_yellow('WARN')}  {code}: {e}")
                continue
            if not doc:
                print(f"  {_yellow('WARN')}  {code}: no MedlinePlus entry")
                continue
            out = ML_DIR / f"{code.replace('.', '_')}.json"
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {_green('+')}  {code}: {doc['title']}")
        return True
    except Exception as exc:
        print(f"  {ERR}: {exc}")
        print("  MedlinePlus Connect is a free NLM API — no key required.")
        return False


def _run_snomed() -> bool:
    print(_bold("\n[3/5] Generating SNOMED CT findings..."))
    print("  (uses src.symptoms_snomed — may call BioPortal/OLS4/Wikidata APIs)")
    try:
        from rag.ingest.fetch_snomed import fetch_one
        for code, name in SEED_CODES:
            try:
                fetch_one(code, name)
            except Exception as e:
                print(f"  {_yellow('WARN')}  {code}: {e}")
        return True
    except Exception as exc:
        print(f"  {ERR}: {exc}")
        return False


def _run_meddialog() -> bool:
    print(_bold("\n[4/5] Fetching doctor-patient dialogues from HuggingFace..."))
    print(f"  (lavita/ChatDoctor-HealthCareMagic-100k, {len(SEED_CODES)} seed codes)")
    try:
        from rag.ingest.fetch_meddialog import fetch_for_icd10
        for code, name in SEED_CODES:
            try:
                fetch_for_icd10(code, name)
            except Exception as e:
                print(f"  {_yellow('WARN')}  {code}: {e}")
        return True
    except Exception as exc:
        print(f"  {ERR}: {exc}")
        return False


def _run_fhir() -> bool:
    print(_bold("\n[5/6] Downloading FHIR R4 examples from hl7.org..."))
    try:
        import json
        import urllib3
        import requests
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        from rag.ingest.fetch_fhir_examples import EXAMPLES, DATA_DIR as FHIR_DIR
        for name, url in EXAMPLES.items():
            out = FHIR_DIR / f"{name}.json"
            if out.exists():
                print(f"  {_green('=')}  {name}: already present")
                continue
            try:
                r = requests.get(url, timeout=20, verify=False,
                                 headers={"Accept": "application/fhir+json"})
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  {_yellow('WARN')}  {name}: {e}")
                continue
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {_green('+')}  {name}: {data.get('resourceType', '?')}")
        return True
    except Exception as exc:
        print(f"  {ERR}: {exc}")
        print("  FHIR examples are hosted at https://hl7.org/fhir/R4/ — no key required.")
        return False


def _build_index() -> bool:
    print(_bold("\n[6/6] Building TF-IDF index..."))
    try:
        from rag.retrieve.index import build
        build()
        return True
    except Exception as exc:
        print(f"  {ERR}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Check and set up RAG datasets for the virtual-patient pipeline."
    )
    ap.add_argument("--check",   action="store_true",
                    help="Only report status, do not download anything.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-download all datasets and rebuild index from scratch.")
    args = ap.parse_args(argv)

    status = check_status()
    print_status(status)

    if args.check:
        all_ok = all(s["ok"] for s in status.values())
        if all_ok:
            print(_green("All datasets and index are present. Ready to use RAG."))
        else:
            missing = [k for k, s in status.items() if not s["ok"]]
            print(_yellow(f"Missing: {', '.join(missing)}"))
            print("Run  python rag/setup.py  to download missing data.")
        return 0 if all_ok else 1

    # ── Download any missing datasets (or all, if --rebuild) ─────────────────
    needs_index_rebuild = False

    if args.rebuild or not status["pubmed"]["ok"]:
        ok = _run_pubmed()
        needs_index_rebuild = needs_index_rebuild or ok

    if args.rebuild or not status["medlineplus"]["ok"]:
        ok = _run_medlineplus()
        needs_index_rebuild = needs_index_rebuild or ok

    if args.rebuild or not status["snomed"]["ok"]:
        ok = _run_snomed()
        needs_index_rebuild = needs_index_rebuild or ok

    if args.rebuild or not status["meddialog"]["ok"]:
        ok = _run_meddialog()
        needs_index_rebuild = needs_index_rebuild or ok

    if args.rebuild or not status["fhir"]["ok"]:
        ok = _run_fhir()
        needs_index_rebuild = needs_index_rebuild or ok

    # ── (Re)build index ───────────────────────────────────────────────────────
    if args.rebuild or not status["index"]["ok"] or needs_index_rebuild:
        _build_index()

    # ── Final status ──────────────────────────────────────────────────────────
    status = check_status()
    print(_bold("\n=== Final status ==="))
    print_status(status)

    all_ok = all(s["ok"] for s in status.values())
    if all_ok:
        print(_green("RAG pipeline is ready."))
        print("You can now run:  python main.py  (with RAG agent enabled)")
    else:
        missing = [k for k, s in status.items() if not s["ok"]]
        print(_red(f"Still missing: {', '.join(missing)}"))
        print()
        if "snomed" in missing:
            print("  SNOMED data requires network access to BioPortal/OLS4/Wikidata.")
            print("  If it still fails, check your .env for any required API keys.")
        if "index" in missing:
            print("  Run:  python rag/retrieve/index.py  to rebuild the index manually.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
