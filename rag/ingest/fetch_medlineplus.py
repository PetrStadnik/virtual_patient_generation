"""
Fetch MedlinePlus (NLM patient-education) summaries via the Connect API.

MedlinePlus Connect lets you query by ICD-10-CM code and returns an Atom feed
with the patient-facing topic summary, plus links to clinical info. This is
ideal for grounding the symptom narrative in your virtual-patient scenarios.

Output: rag/data/medlineplus/<icd_code>.json
  {"id": "MEDLINEPLUS:<code>", "source": "medlineplus", "icd10": ...,
   "title": ..., "summary": ..., "url": ...}
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONNECT = "https://connect.medlineplus.gov/service"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "medlineplus"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
}


def fetch_for_icd10(code: str) -> dict | None:
    params = {
        # ICD-10-CM diagnosis OID
        "mainSearchCriteria.v.cs": "2.16.840.1.113883.6.90",
        "mainSearchCriteria.v.c": code,
        "knowledgeResponseType": "application/atom+xml",
    }
    r = requests.get(CONNECT, params=params, timeout=20, verify=False)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    entry = root.find("atom:entry", NS)
    if entry is None:
        return None
    title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
    link_el = entry.find("atom:link", NS)
    url = link_el.attrib.get("href", "") if link_el is not None else ""
    if not summary:
        return None
    return {
        "id": f"MEDLINEPLUS:{code}",
        "source": "medlineplus",
        "icd10": code,
        "title": title,
        "summary": summary,
        "url": url,
    }


# Minimal fallback for standalone use — setup.py drives the real seed list.
DEFAULT_CODES: list[str] = []


def main(argv: list[str]) -> int:
    codes = argv[1:] or DEFAULT_CODES
    for code in codes:
        try:
            doc = fetch_for_icd10(code)
        except Exception as e:  # network or parse failure — keep going
            print(f"[medlineplus] {code}: ERROR {e}")
            continue
        if not doc:
            print(f"[medlineplus] {code}: no entry")
            continue
        out = DATA_DIR / f"{code.replace('.', '_')}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[medlineplus] {code}: '{doc['title']}' -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
