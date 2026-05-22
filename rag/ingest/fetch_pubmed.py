"""
Fetch PubMed case reports for a given clinical query via NCBI E-utilities.

E-utilities are free and unauthenticated, but throttled (3 req/s without an
API key, 10 req/s with one). We deliberately keep batches small so the diploma
demo is reproducible without registering for a key.

Output: rag/data/pubmed/<slug>.jsonl
Each line is one document:
  {"id": "PMID:...", "source": "pubmed", "title": ..., "abstract": ...,
   "journal": ..., "year": ..., "url": ...}
"""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pubmed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def search_pmids(query: str, retmax: int = 10) -> list[str]:
    params = {
        "db": "pubmed",
        "term": f"({query}) AND case reports[pt]",
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    r = requests.get(ESEARCH, params=params, timeout=20, verify=False)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: Iterable[str]) -> list[dict]:
    pmids = list(pmids)
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    r = requests.get(EFETCH, params=params, timeout=30, verify=False)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    docs: list[dict] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        title = (art.findtext(".//ArticleTitle") or "").strip()
        # Abstracts may have multiple labelled sections (BACKGROUND/METHODS/...)
        abs_parts: list[str] = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.attrib.get("Label")
            text = "".join(ab.itertext()).strip()
            abs_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(p for p in abs_parts if p)
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = (art.findtext(".//PubDate/Year")
                or art.findtext(".//PubDate/MedlineDate")
                or "").strip()[:4]
        if not abstract:
            continue  # skip records without abstracts (no value for RAG)
        docs.append({
            "id": f"PMID:{pmid}",
            "source": "pubmed",
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return docs


def fetch_query(query: str, retmax: int = 10) -> Path:
    """Run a query, save JSONL, return the file path."""
    print(f"[pubmed] searching: {query!r}")
    pmids = search_pmids(query, retmax=retmax)
    print(f"[pubmed]   found {len(pmids)} PMIDs")
    time.sleep(0.4)  # be polite
    docs = fetch_abstracts(pmids)
    print(f"[pubmed]   fetched {len(docs)} abstracts")
    out = DATA_DIR / f"{_slugify(query)}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[pubmed]   wrote {out.relative_to(DATA_DIR.parent.parent.parent)}")
    return out


# Minimal fallback for standalone use — setup.py drives the real seed list.
DEFAULT_QUERIES: list[str] = []


def main(argv: list[str]) -> int:
    queries = argv[1:] or DEFAULT_QUERIES
    for q in queries:
        fetch_query(q, retmax=8)
        time.sleep(0.4)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
