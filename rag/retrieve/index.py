"""
Build a tiny but real RAG index over rag/data/.

Design choice
-------------
For a thesis demo we want a vector store that:
  * has zero install pain (no FAISS, no Torch, no Docker),
  * is fully reproducible from rag/data/ alone,
  * can be swapped for OpenAI/Anthropic embeddings without changing callers.

We therefore use **TF-IDF + cosine similarity** as the default backend. It
gives genuinely useful retrieval on biomedical case-report text, and the
``Retriever`` class exposes the same ``search()`` signature you'd get from
FAISS. To upgrade to dense embeddings later, replace ``_fit`` and ``_embed``
to call ``OpenAI().embeddings.create(...)``.

Index files written to rag/index/:
  * docs.jsonl    — flat, one document per line (id, source, text, meta)
  * tfidf.pkl     — fitted TfidfVectorizer + document matrix
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Doc:
    id: str
    source: str
    text: str
    meta: dict[str, Any]


# -- Loaders ---------------------------------------------------------------

def _load_pubmed() -> Iterable[Doc]:
    for p in sorted((DATA_DIR / "pubmed").glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yield Doc(
                    id=d["id"],
                    source="pubmed",
                    text=f"{d['title']}\n\n{d['abstract']}",
                    meta={"title": d["title"], "journal": d.get("journal", ""),
                          "year": d.get("year", ""), "url": d["url"],
                          "query_file": p.name},
                )


def _load_medlineplus() -> Iterable[Doc]:
    for p in sorted((DATA_DIR / "medlineplus").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        yield Doc(
            id=d["id"],
            source="medlineplus",
            text=f"{d['title']}\n\n{d['summary']}",
            meta={"title": d["title"], "icd10": d["icd10"], "url": d["url"]},
        )


def _load_snomed() -> Iterable[Doc]:
    for p in sorted((DATA_DIR / "snomed").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        symptoms = d.get("symptoms") or []
        if not symptoms:
            continue
        text = (f"SNOMED CT clinical findings associated with "
                f"{d['diagnosis']} (ICD-10 {d['icd10']}):\n- "
                + "\n- ".join(symptoms))
        yield Doc(
            id=d["id"],
            source="snomed",
            text=text,
            meta={"diagnosis": d["diagnosis"], "icd10": d["icd10"],
                  "url": d["url"]},
        )


def _summarize_fhir(d: dict) -> str:
    """Compress a FHIR resource to the parts that matter for grounding."""
    rt = d.get("resourceType", "?")
    parts = [f"FHIR resourceType: {rt}"]
    if rt == "Patient":
        parts.append(f"gender={d.get('gender')}, birthDate={d.get('birthDate')}")
        parts.append(f"name={d.get('name', [{}])[0].get('text', '')}")
    elif rt == "Condition":
        code = d.get("code", {})
        parts.append("condition.code.text=" + code.get("text", ""))
        for c in code.get("coding", []):
            parts.append(f"  coding {c.get('system','')} {c.get('code','')} "
                         f"{c.get('display','')}")
        if "clinicalStatus" in d:
            parts.append("clinicalStatus="
                         + json.dumps(d["clinicalStatus"]))
    elif rt == "Observation":
        parts.append("observation.code="
                     + json.dumps(d.get("code", {})))
        if "component" in d:
            for c in d["component"]:
                parts.append("  component=" + json.dumps(c))
    elif rt == "Encounter":
        parts.append("class=" + json.dumps(d.get("class", {})))
        parts.append("type=" + json.dumps(d.get("type", [])))
    parts.append("\n--- raw JSON (truncated) ---")
    parts.append(json.dumps(d, ensure_ascii=False)[:1500])
    return "\n".join(parts)


def _load_meddialog() -> Iterable[Doc]:
    for p in sorted((DATA_DIR / "meddialog").glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield Doc(
                    id=d["id"],
                    source="meddialog",
                    text=d["text"],
                    meta={"title": d.get("title", ""), "icd10": d.get("icd10", ""),
                          "url": d.get("url", "")},
                )


def _load_fhir() -> Iterable[Doc]:
    for p in sorted((DATA_DIR / "fhir").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        yield Doc(
            id=f"FHIR:{p.stem}",
            source="fhir",
            text=_summarize_fhir(d),
            meta={"resourceType": d.get("resourceType"),
                  "url": f"https://hl7.org/fhir/R4/{p.stem}.json"},
        )


def load_all() -> list[Doc]:
    docs: list[Doc] = []
    for loader in (_load_pubmed, _load_medlineplus, _load_snomed, _load_fhir, _load_meddialog):
        docs.extend(loader())
    return docs


# -- Index build / load ----------------------------------------------------

def build() -> None:
    docs = load_all()
    if not docs:
        raise RuntimeError("No documents found under rag/data/. "
                           "Run rag/ingest/*.py first.")
    out = INDEX_DIR / "docs.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    vec = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), max_features=20000,
    )
    matrix = vec.fit_transform([d.text for d in docs])

    with (INDEX_DIR / "tfidf.pkl").open("wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": matrix,
                     "ids": [d.id for d in docs]}, f)
    by_src: dict[str, int] = {}
    for d in docs:
        by_src[d.source] = by_src.get(d.source, 0) + 1
    print(f"[index] indexed {len(docs)} docs: {by_src}")
    print(f"[index] vocab size: {len(vec.vocabulary_)}")


def _indexed_icd_codes() -> set[str]:
    """Return the set of ICD-10 codes that already have documents in the index."""
    docs_file = INDEX_DIR / "docs.jsonl"
    if not docs_file.exists():
        return set()
    codes: set[str] = set()
    for line in docs_file.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        meta = d.get("meta", {})
        icd = meta.get("icd10") or meta.get("query_file", "")
        if icd:
            codes.add(icd.replace("_", "."))
        # also parse id like SNOMED:E11.9 / MEDLINEPLUS:E11.9
        doc_id: str = d.get("id", "")
        if ":" in doc_id:
            codes.add(doc_id.split(":", 1)[1])
    return codes


def ensure_indexed(icd_code: str, diagnosis_name: str = "") -> bool:
    """Ensure data for *icd_code* exists in the index, fetching if necessary.

    Returns True if the index was rebuilt (new data was fetched), False if
    the code was already covered and no action was needed.
    """
    # Normalise: "J06.9" and "J06_9" are the same
    code_norm = icd_code.upper().strip()

    already = _indexed_icd_codes()
    if code_norm in already:
        return False  # already indexed, nothing to do

    print(f"[index] {code_norm} not in index — fetching data...")
    fetched = False

    # 1. MedlinePlus
    try:
        import json as _json
        from rag.ingest.fetch_medlineplus import fetch_for_icd10, DATA_DIR as ML_DIR
        doc = fetch_for_icd10(code_norm)
        if doc:
            out = ML_DIR / f"{code_norm.replace('.', '_')}.json"
            out.write_text(_json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[index]   fetched medlineplus: {doc.get('title', code_norm)}")
            fetched = True
        else:
            print(f"[index]   medlineplus: no entry for {code_norm}")
    except Exception as e:
        print(f"[index]   medlineplus error: {e}")

    # 2. SNOMED (via src.symptoms_snomed)
    try:
        from rag.ingest.fetch_snomed import fetch_one
        name = diagnosis_name or code_norm
        fetch_one(code_norm, name)
        print(f"[index]   fetched snomed: {name}")
        fetched = True
    except Exception as e:
        print(f"[index]   snomed error: {e}")

    # 3. PubMed — one targeted query
    try:
        from rag.ingest.fetch_pubmed import fetch_query
        query = f"{diagnosis_name or code_norm} case report"
        path = fetch_query(query, retmax=6)
        if path.stat().st_size > 10:
            fetched = True
    except Exception as e:
        print(f"[index]   pubmed error: {e}")

    # 4. MedDialog — stream HuggingFace for relevant doctor-patient dialogues
    try:
        from rag.ingest.fetch_meddialog import fetch_for_icd10 as _md_fetch
        path = _md_fetch(code_norm, diagnosis_name or code_norm)
        if path.stat().st_size > 50:
            fetched = True
    except Exception as e:
        print(f"[index]   meddialog error: {e}")

    # Rebuild index if anything new was fetched
    if fetched:
        print(f"[index] rebuilding index...")
        build()
        return True

    print(f"[index] no new data found for {code_norm}, proceeding with existing index")
    return False


class Retriever:
    def __init__(self) -> None:
        with (INDEX_DIR / "tfidf.pkl").open("rb") as f:
            obj = pickle.load(f)
        self._vec: TfidfVectorizer = obj["vectorizer"]
        self._matrix = obj["matrix"]
        self._ids: list[str] = obj["ids"]
        self._docs: dict[str, Doc] = {}
        with (INDEX_DIR / "docs.jsonl").open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self._docs[d["id"]] = Doc(**d)

    def search(self, query: str, k: int = 5,
               sources: list[str] | None = None) -> list[tuple[Doc, float]]:
        qv = self._vec.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = np.argsort(-sims)
        out: list[tuple[Doc, float]] = []
        for idx in order:
            doc = self._docs[self._ids[idx]]
            if sources and doc.source not in sources:
                continue
            score = float(sims[idx])
            if score <= 0:
                break
            out.append((doc, score))
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    build()
