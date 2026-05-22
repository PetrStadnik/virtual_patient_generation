"""
RAG-grounded virtual-patient generation.

Public API
----------
generate_patient(model, icd_code, diagnosis_name, gender, age, ...)
    RAG-grounded generation — evidence injected into user message.

generate_patient_rag_with_tools(model, icd_code, diagnosis_name, gender, age, ...)
    RAG + tool-verified codes — combines retrieved evidence with live
    lookups for ATC (drugs), LOINC (tests) and SNOMED (procedures).

CLI (demo / verification)
-------------------------
  python rag/generate/run.py --persona rag/demos/personas.json --idx 0
  python rag/generate/run.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import re
import warnings

import httpx
import requests
from urllib3.exceptions import InsecureRequestWarning
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from rag.retrieve.index import Retriever, ensure_indexed
from rag.generate.prompt import SYSTEM, build_user_message
from src.LLM_generation.llm_io import PatientInput, PatientOutput

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

OUT_DIR = PROJECT_ROOT / "rag" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def _provider_from_model(model: str) -> str:
    """Infer provider string from a model name (e.g. 'claude-...' → 'anthropic')."""
    return "anthropic" if "claude" in model.lower() else "openai"


def _ssl_http() -> httpx.AsyncClient:
    """Return an AsyncClient with SSL verification disabled (proxy environments)."""
    return httpx.AsyncClient(verify=False)


def _make_llm(model: str):
    """Create a pydantic-ai LLM model with verify=False httpx transport."""
    _http = _ssl_http()
    if "claude" in model.lower():
        return AnthropicModel(model, provider=AnthropicProvider(http_client=_http))
    return OpenAIModel(model, provider=OpenAIProvider(http_client=_http))


def _make_agent(model: str) -> Agent:
    """Create a pydantic-ai Agent that outputs a PatientOutput."""
    return Agent(_make_llm(model), output_type=PatientOutput, system_prompt=SYSTEM)


def _default_model(provider: str) -> str:
    return "claude-sonnet-4-6" if provider == "anthropic" else "gpt-4o-mini"


# ── RAG + tools agent ──────────────────────────────────────────────────────

_ATC_RE = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")
_WIKIDATA_URL = "https://query.wikidata.org/sparql"
_WIKIDATA_HEADERS = {"User-Agent": "fhir-patient-generator/1.0"}

# Combined system prompt: RAG grounding + tool-verification rules
_SYSTEM_RAG_TOOLS = (
    SYSTEM.rstrip()
    + """

ADDITIONAL RULES — Code verification via tools:
- Medications: call search_drug for every medication. Only include if found=true.
- Medications MUST be for comorbidities or pre-existing conditions — NOT for treating the primary diagnosis.
- Medical tests: call search_loinc for every test. Only include if found=true.
- Allergies: call search_drug for allergens to retrieve the ATC code.
- Procedures: call search_snomed_procedure for every procedure. Only include if found=true.
"""
)

# Agent is created fresh per call (model changes); tools are registered below.
_tools_agent: Agent | None = None
_tools_agent_model: str = ""


def _get_tools_agent(model: str) -> Agent:
    """Return (or rebuild) the tools agent for the given model."""
    global _tools_agent, _tools_agent_model
    if _tools_agent is None or _tools_agent_model != model:
        _tools_agent = Agent(
            _make_llm(model),
            deps_type=PatientInput,
            output_type=PatientOutput,
            system_prompt=_SYSTEM_RAG_TOOLS,
            retries=3,
        )
        _register_tools(_tools_agent)
        _tools_agent_model = model
    return _tools_agent


def _register_tools(agent: Agent) -> None:
    """Register the three medical lookup tools on an agent."""

    @agent.tool
    def search_drug(ctx: RunContext[PatientInput], drug_name: str) -> dict:
        """Look up a drug by generic name. Returns WHO ATC code (format A00AA00)."""
        sparql = f"""SELECT ?atc WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search "{drug_name}";
                            mwapi:language "en".
            ?d wikibase:apiOutputItem mwapi:item.
          }}
          ?d wdt:P267 ?atc .
        }} LIMIT 5"""
        try:
            r = requests.get(_WIKIDATA_URL,
                             params={"query": sparql, "format": "json"},
                             headers=_WIKIDATA_HEADERS, timeout=15, verify=False)
            rows = r.json()["results"]["bindings"]
        except Exception:
            rows = []
        for row in rows:
            code = row["atc"]["value"]
            if _ATC_RE.match(code):
                print(f"  [tool] search_drug({drug_name!r}) -> ATC {code}")
                return {"found": True, "drug_name": drug_name, "atc_code": code}
        if rows:
            code = rows[0]["atc"]["value"]
            print(f"  [tool] search_drug({drug_name!r}) -> ATC {code} (group-level)")
            return {"found": True, "drug_name": drug_name, "atc_code": code}
        print(f"  [tool] search_drug({drug_name!r}) -> not found")
        return {"found": False, "drug_name": drug_name, "atc_code": ""}

    @agent.tool
    def search_loinc(ctx: RunContext[PatientInput], test_name: str) -> dict:
        """Look up a medical test by name. Returns LOINC code (format NNNNN-N)."""
        try:
            r = requests.get(
                "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search",
                params={"terms": test_name, "df": "LOINC_NUM,LONG_COMMON_NAME", "maxList": 1},
                timeout=15, verify=False,
            )
            codes = r.json()[1]
            if codes:
                print(f"  [tool] search_loinc({test_name!r}) -> {codes[0]}")
                return {"found": True, "test_name": test_name, "loinc_code": codes[0]}
        except Exception:
            pass
        print(f"  [tool] search_loinc({test_name!r}) -> not found")
        return {"found": False, "test_name": test_name, "loinc_code": ""}

    @agent.tool
    def search_snomed_procedure(ctx: RunContext[PatientInput], procedure_name: str) -> dict:
        """Look up a medical procedure. Returns SNOMED CT code."""
        try:
            r = requests.get(
                "https://r4.ontoserver.csiro.au/fhir/ValueSet/$expand",
                params={"url": "http://snomed.info/sct?fhir_vs=ecl/<<71388002",
                        "filter": procedure_name, "count": 1},
                timeout=15, verify=False,
            )
            if r.status_code == 200:
                items = r.json().get("expansion", {}).get("contains", [])
                if items:
                    code = items[0]["code"]
                    display = items[0].get("display", procedure_name)
                    print(f"  [tool] search_snomed_procedure({procedure_name!r}) -> {code} ({display})")
                    return {"found": True, "procedure_name": procedure_name,
                            "snomed_code": code}
        except Exception:
            pass
        print(f"  [tool] search_snomed_procedure({procedure_name!r}) -> not found")
        return {"found": False, "procedure_name": procedure_name, "snomed_code": ""}


# ── Main public function ───────────────────────────────────────────────────

def generate_patient(
    model: str,
    icd_code: str,
    diagnosis_name: str,
    gender: str,
    age: int,
    country: str = "CZ",
    k: int = 6,
    dump: bool = True,
) -> dict | PatientOutput:
    """RAG-grounded patient generation.

    Parameters mirror baseline_script.generate_patient and
    agent_with_tools.generate_patient_with_tools so the function is a
    drop-in replacement in main.py.

    Returns
    -------
    dict          when dump=True  (default, matches other generators)
    PatientOutput when dump=False
    """
    provider = _provider_from_model(model)
    print(f"[rag] {icd_code} / {diagnosis_name}  ({gender}, {age}y)  model={model}")
    ensure_indexed(icd_code, diagnosis_name)
    retriever = Retriever()

    age_band = "elderly" if age >= 65 else ("adolescent" if age < 18 else "adult")
    query = f"{diagnosis_name} {icd_code} {age_band} {gender}"
    evidence = retriever.search(query, k=k)

    if not evidence:
        raise RuntimeError(
            "Retriever returned 0 evidence — build the index first:\n"
            "  python rag/retrieve/index.py"
        )

    print(f"[rag] retrieved {len(evidence)} passages:")
    for doc, score in evidence:
        print(f"  {score:.3f}  [{doc.source}]  {doc.id}")

    agent = _make_agent(model)
    user_msg = build_user_message(icd_code, diagnosis_name, age, gender, country, evidence)

    t0 = time.time()
    output: PatientOutput = agent.run_sync(user_msg).output
    print(f"[rag] done in {time.time() - t0:.1f}s")

    return output.model_dump() if dump else output


def generate_patient_rag_with_tools(
    model: str,
    icd_code: str,
    diagnosis_name: str,
    gender: str,
    age: int,
    country: str = "CZ",
    k: int = 6,
    dump: bool = True,
) -> dict | PatientOutput:
    """RAG-grounded generation with live tool lookups for ATC/LOINC/SNOMED codes.

    Combines the two approaches:
      - Retrieved evidence passages are injected into the user message (RAG)
      - The agent also has access to search_drug / search_loinc /
        search_snomed_procedure tools for verified medical codes (tools)

    Same return-value contract as generate_patient and
    generate_patient_with_tools — drop-in replacement in main.py.
    """
    import random
    from pydantic_ai.exceptions import ModelHTTPError

    provider = _provider_from_model(model)
    print(f"[rag+tools] {icd_code} / {diagnosis_name}  ({gender}, {age}y)  model={model}")
    ensure_indexed(icd_code, diagnosis_name)
    retriever = Retriever()

    age_band = "elderly" if age >= 65 else ("adolescent" if age < 18 else "adult")
    query = f"{diagnosis_name} {icd_code} {age_band} {gender}"
    evidence = retriever.search(query, k=k)

    if not evidence:
        raise RuntimeError(
            "Retriever returned 0 evidence — build the index first:\n"
            "  python rag/retrieve/index.py"
        )

    print(f"[rag+tools] retrieved {len(evidence)} passages:")
    for doc, score in evidence:
        print(f"  {score:.3f}  [{doc.source}]  {doc.id}")

    agent = _get_tools_agent(model)
    deps  = PatientInput(icd_code=icd_code, diagnosis_name=diagnosis_name,
                         gender=gender, age=age)
    user_msg = build_user_message(icd_code, diagnosis_name, age, gender, country, evidence)

    t0 = time.time()
    for attempt in range(8):
        try:
            output: PatientOutput = agent.run_sync(user_msg, deps=deps).output
            break
        except ModelHTTPError as exc:
            if exc.status_code == 529:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[rag+tools] model overloaded — retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("generate_patient_rag_with_tools: all retries exhausted")

    print(f"[rag+tools] done in {time.time() - t0:.1f}s")
    return output.model_dump() if dump else output


# ── CLI entry-point (demo / thesis verification) ───────────────────────────

def run_one(
    persona: dict,
    *,
    k: int = 6,
    provider: str = "anthropic",
    model: str | None = None,
    save_trace: bool = True,
) -> Path | dict:
    """Run one persona dict and optionally save a trace JSON for verification.

    persona keys: icd10, diagnosis, age, sex, country (optional)
    """
    icd_code       = persona["icd10"]
    diagnosis_name = persona["diagnosis"]
    age            = persona["age"]
    sex            = persona["sex"]
    country        = persona.get("country", "CZ")

    print(f"\n=== {diagnosis_name} ({icd_code}, {age}y {sex}) ===")

    retriever = Retriever()
    age_band = "elderly" if age >= 65 else ("adolescent" if age < 18 else "adult")
    query = f"{diagnosis_name} {icd_code} {age_band} {sex}"
    evidence = retriever.search(query, k=k)

    print(f"[retrieve] query={query!r}")
    for doc, score in evidence:
        print(f"  {score:.3f}  {doc.id}  ({doc.source})")

    if not evidence:
        raise RuntimeError("Retriever returned 0 evidence — index is empty")

    _model = model or _default_model(provider)
    agent = _make_agent(_model)
    user_msg = build_user_message(icd_code, diagnosis_name, age, sex, country, evidence)

    t0 = time.time()
    output: PatientOutput = agent.run_sync(user_msg).output
    dt = time.time() - t0
    print(f"[llm] {provider} returned in {dt:.1f}s")

    if not save_trace:
        return output.model_dump()

    trace = {
        "persona": persona,
        "query": query,
        "provider": provider,
        "model": _model,
        "evidence": [
            {
                "id": d.id,
                "source": d.source,
                "score": s,
                "url": d.meta.get("url", ""),
                "snippet": d.text[:300],
            }
            for d, s in evidence
        ],
        "scenario": output.model_dump(),
    }
    fname = (
        f"{icd_code.replace('.', '_')}"
        f"_{age}_{sex}_{provider}.json"
    )
    out = OUT_DIR / fname
    out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out.relative_to(PROJECT_ROOT)}")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="rag/demos/personas.json")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args(argv)

    personas = json.loads(Path(args.persona).read_text(encoding="utf-8"))
    targets = personas if args.all else [personas[args.idx]]
    for p in targets:
        try:
            run_one(p, k=args.k, provider=args.provider, model=args.model)
        except Exception as e:
            print(f"[error] {p['icd10']}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
