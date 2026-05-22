"""
ICD-10 -> structured SNOMED CT symptom findings.

Backends (tried in order, transparent fallback):
  1. BioPortal search   — needs BIOPORTAL_API_KEY; used only for disorder SCTID lookup
                          (BioPortal SPARQL is permanently unavailable — HTTP 522).
  2. FHIR ECL           — public FHIR R4 servers; reliable for disorder SCTID lookup.
  3. Wikidata P780      — free, no key; good for well-curated diseases (e.g. diabetes,
                          cancer).  Symptom names resolved to SNOMED via EBI OLS4.
                          Coverage gap: many musculoskeletal conditions have no P780 data.
  4. Wikipedia S&S      — free, no key; extracts wikilinks + symptom keywords from
                          the "Signs and symptoms" article section (or REST summary).
                          Covers conditions missing from Wikidata P780
                          (e.g. Low back pain, Gonarthrosis, Myalgia).
  5. Curated seed       — hard-coded fallback (src/symptoms_seed.py).

Public API:
  get_disease_symptoms(icd10_code) -> DiseaseSymptoms
  get_symptoms_snomed(icd10_code, diagnosis_name=None) -> list[str]   (legacy)

Each SymptomFinding carries everything needed for a FHIR R4 Observation:
SNOMED code, display, severity options, body sites, and to_fhir_observation().
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import warnings
from dataclasses import asdict, dataclass, field

import requests
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import simple_icd_10 as icd  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

SNOMED_SYSTEM = "http://snomed.info/sct"

_BP_BASE    = "https://data.bioontology.org"

_FHIR_TX_SERVERS = [
    "https://r4.ontoserver.csiro.au/fhir",
    "https://tx.fhir.org/r4",
    "https://snowstorm.snomedtools.org/fhir",
]

_OLS4_BASE  = "https://www.ebi.ac.uk/ols4/api"
_WD_SPARQL  = "https://query.wikidata.org/sparql"
_WD_HEADERS = {"User-Agent": "fhir-patient-generator/1.0"}
_WP_HEADERS = {"User-Agent": "fhir-patient-generator/1.0 (medical thesis research)"}
_WP_API     = "https://en.wikipedia.org/w/api.php"
_WP_REST    = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Ordered from most specific to most generic — used to extract phrases from prose
_SYM_WORDS = [
    "radiculopathy", "sciatica", "paresthesia", "dysesthesia",
    "atrophy", "spasticity", "tremor", "rigidity", "bradykinesia",
    "crepitus", "tenderness", "inflammation", "swelling",
    "stiffness", "weakness", "fatigue", "numbness", "tingling",
    "spasm", "cramp", "soreness", "ache", "pain",
]


SEVERITY_OPTIONS: list[dict] = [
    {"code": "255604002", "display": "Mild"},
    {"code": "6736007",   "display": "Moderate"},
    {"code": "24484000",  "display": "Severe"},
    {"code": "442452003", "display": "Life threatening severity"},
]



# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SnomedCoding:
    code: str
    display: str
    system: str = SNOMED_SYSTEM

    def to_fhir(self) -> dict:
        return {"system": self.system, "code": self.code, "display": self.display}


@dataclass
class SymptomFinding:
    snomed: SnomedCoding
    finding_site: list[SnomedCoding] = field(default_factory=list)
    severity_options: list[SnomedCoding] = field(
        default_factory=lambda: [SnomedCoding(**s) for s in SEVERITY_OPTIONS]
    )
    default_present: bool = True

    def to_fhir_observation(
        self,
        subject_ref: str,
        *,
        encounter_ref: str | None = None,
        present: bool | None = None,
        onset_iso: str | None = None,
        severity_code: str | None = None,
        observation_id: str | None = None,
    ) -> dict:
        is_present = self.default_present if present is None else present
        obs: dict = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "exam",
                    "display": "Exam",
                }]
            }],
            "code": {"coding": [self.snomed.to_fhir()],
                     "text": self.snomed.display},
            "subject": {"reference": subject_ref},
            "valueCodeableConcept": {
                "coding": [{
                    "system": SNOMED_SYSTEM,
                    "code": "52101004" if is_present else "2667000",
                    "display": "Present" if is_present else "Absent",
                }]
            },
        }
        if observation_id:
            obs["id"] = observation_id
        if encounter_ref:
            obs["encounter"] = {"reference": encounter_ref}
        if onset_iso:
            obs["effectiveDateTime"] = onset_iso
        if severity_code:
            sev = next((s for s in self.severity_options
                        if s.code == severity_code), None)
            if sev is None:
                sev = SnomedCoding(code=severity_code, display="")
            obs["interpretation"] = [{"coding": [sev.to_fhir()]}]
        if self.finding_site:
            obs["bodySite"] = {"coding": [s.to_fhir() for s in self.finding_site]}
        return obs


@dataclass
class DiseaseSymptoms:
    icd10: str
    icd10_description: str | None
    snomed_disorder: SnomedCoding | None
    symptoms: list[SymptomFinding]

    def to_dict(self) -> dict:
        return {
            "icd10": self.icd10,
            "icd10_description": self.icd10_description,
            "snomed_disorder":
                asdict(self.snomed_disorder) if self.snomed_disorder else None,
            "symptoms": [
                {"snomed": asdict(s.snomed),
                 "finding_site": [asdict(b) for b in s.finding_site],
                 "default_present": s.default_present}
                for s in self.symptoms
            ],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_key(name: str) -> str | None:
    return os.environ.get(name) or None


def _icd10_to_description(code: str) -> str | None:
    normalized = code.strip().upper()
    if not icd.is_valid_item(normalized):
        return None
    try:
        return icd.get_description(normalized)
    except Exception:
        return None


def _sctid_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def _word_overlap_ratio(query: str, label: str) -> float:
    """Jaccard-style overlap of significant (>2-char) words."""
    stop = {"the", "and", "for", "with", "from", "into", "due", "not",
            "this", "that", "has", "have"}
    q = {w for w in query.lower().split() if len(w) > 2 and w not in stop}
    l = {w for w in label.lower().split() if len(w) > 2 and w not in stop}
    if not q:
        return 0.0
    return len(q & l) / len(q)




# ── Backend 2: Wikidata P780 + EBI OLS4 ─────────────────────────────────────

def _wikidata_sparql(sparql: str) -> list[dict]:
    try:
        r = requests.get(
            _WD_SPARQL,
            params={"query": sparql, "format": "json"},
            headers=_WD_HEADERS,
            timeout=15, verify=False,
        )
        return r.json()["results"]["bindings"]
    except Exception:
        return []


def _wikidata_symptom_names(icd10: str, name: str) -> list[str]:
    """Return symptom display names from Wikidata P780 (symptoms and signs).

    Tries three strategies in sequence and merges results:
      A. Exact ICD-10 code + parent code via property P494.
      B. mwapi EntitySearch by diagnosis name (always tried as fallback).
    """
    parent = icd10.split(".")[0]
    names: list[str] = []

    # Strategy A: exact + parent ICD-10 code (Wikidata property P494)
    sparql_a = f"""SELECT DISTINCT ?sLabel WHERE {{
      {{ ?d wdt:P494 "{icd10}" . }} UNION {{ ?d wdt:P494 "{parent}" . }}
      ?d wdt:P780 ?symptom .
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?symptom rdfs:label ?sLabel.
      }}
    }} LIMIT 15"""
    for row in _wikidata_sparql(sparql_a):
        v = row.get("sLabel", {}).get("value", "")
        if v:
            names.append(v)

    # Strategy B: mwapi EntitySearch by diagnosis name (always attempted)
    # Strip ICD-10 qualifiers ("without X", "unspecified", "NOS") so mwapi
    # finds the canonical Wikidata concept rather than an exact sub-specifier.
    search_term = re.sub(
        r"[,;]\s*(unspecified|not otherwise specified|nos)\b.*",
        "", name, flags=re.IGNORECASE,
    ).strip()
    search_term = re.sub(
        r"\s+(without|with|unspecified)\s+.*", "", search_term,
        flags=re.IGNORECASE,
    ).strip()
    sparql_b = f"""SELECT DISTINCT ?sLabel WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{search_term}";
                        mwapi:language "en".
        ?d wikibase:apiOutputItem mwapi:item.
      }}
      ?d wdt:P780 ?symptom .
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?symptom rdfs:label ?sLabel.
      }}
    }} LIMIT 15"""
    for row in _wikidata_sparql(sparql_b):
        v = row.get("sLabel", {}).get("value", "")
        if v and v not in names:
            names.append(v)

    return names


def _ols4_snomed_code(term: str) -> SnomedCoding | None:
    """Translate a symptom/finding name to SNOMED CT code via EBI OLS4.

    Tries exact match first, then fuzzy with a word-overlap quality gate.
    """
    for exact in (True, False):
        try:
            r = requests.get(
                f"{_OLS4_BASE}/search",
                params={"q": term, "ontology": "snomed", "type": "class",
                        "fieldList": "label,short_form", "rows": 5,
                        "exact": str(exact).lower()},
                timeout=10, verify=False,
            )
            if r.status_code != 200:
                continue
            docs = r.json().get("response", {}).get("docs", [])
            for doc in docs:
                sf = doc.get("short_form", "")
                if not sf.startswith("SNOMED_"):
                    continue
                code  = sf[7:]          # strip "SNOMED_" prefix
                label = doc.get("label", term)
                if exact or _word_overlap_ratio(term, label) >= 0.4:
                    return SnomedCoding(code=code, display=label)
        except Exception:
            pass
    return None


def _wikidata_ols4_backend(icd10: str, name: str,
                           limit: int) -> list[SnomedCoding]:
    """Wikidata P780 symptom names → SNOMED codes via EBI OLS4."""
    names = _wikidata_symptom_names(icd10, name)
    if not names:
        return []
    print(f"  [Wikidata] {len(names)} symptom names: {names[:5]}")

    findings: list[SnomedCoding] = []
    for name in names[:limit]:
        sc = _ols4_snomed_code(name)
        if sc:
            findings.append(sc)
    print(f"  [Wikidata+OLS4] {len(findings)} SNOMED codes resolved")
    return findings


# ── Backend 2.5: Wikipedia "Signs and symptoms" + OLS4 ──────────────────────

def _wp_get(params: dict) -> dict | list:
    """Wikipedia API call; returns parsed JSON or empty dict/list on failure."""
    try:
        r = requests.get(_WP_API, params={**params, "format": "json"},
                         headers=_WP_HEADERS, timeout=10, verify=False)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _wp_find_title(query: str) -> str | None:
    """Return the Wikipedia article title best matching *query*, or None."""
    data = _wp_get({"action": "opensearch", "search": query, "limit": 1})
    return data[1][0] if isinstance(data, list) and data[1] else None


def _wp_extract_candidates(title: str, disease_lower: str) -> list[str]:
    """Return symptom candidate strings from a Wikipedia article.

    Strategy:
    1. If a 'Signs and symptoms' (or similar) section exists, extract its
       wikilinks — these are key medical terms chosen by editors.
    2. Otherwise (or in addition) parse the REST summary prose and look for
       specific multi-word phrases containing symptom keywords.
    """
    candidates: list[str] = []
    plain_text = ""

    # ── 1. Try the Signs & Symptoms section ──────────────────────────────────
    data = _wp_get({"action": "parse", "page": title, "prop": "sections"})
    sections = data.get("parse", {}).get("sections", [])
    sym_idx = next(
        (s["index"] for s in sections if any(
            kw in s.get("line", "").lower()
            for kw in ("sign", "symptom", "presentation", "clinical feature")
        )),
        None,
    )

    if sym_idx:
        wt_data = _wp_get({"action": "parse", "page": title,
                           "section": sym_idx, "prop": "wikitext"})
        wt = wt_data.get("parse", {}).get("wikitext", {}).get("*", "")

        # Extract [[Target|display]] or [[Target]] wikilinks
        for link in re.findall(r"\[\[([^\]|#]+?)(?:\|[^\]]+)?\]\]", wt):
            link = link.strip().lower()
            # Skip disambiguation qualifiers that signal non-medical topics
            if re.search(
                r"\((?:mood|anatomy|physiology|chemistry|biology|sociology)\)", link
            ):
                continue
            if 3 < len(link) < 50 and link != disease_lower:
                candidates.append(link)

        # Strip markup for keyword scanning
        plain_text = re.sub(r"\{\{[^}]+\}\}", "", wt)
        plain_text = re.sub(r"\[\[[^\]]+\]\]", " ", plain_text)
        plain_text = re.sub(r"<[^>]+>", "", plain_text).lower()

    # ── 2. REST summary as fallback plain-text source ─────────────────────────
    if not plain_text:
        try:
            r = requests.get(
                f"{_WP_REST}/{urllib.parse.quote(title)}",
                headers=_WP_HEADERS, timeout=10, verify=False,
            )
            plain_text = (
                r.json().get("extract", "").lower()
                if r.status_code == 200 else ""
            )
        except Exception:
            plain_text = ""

    # ── 3. Extract specific symptom phrases from prose ────────────────────────
    for w in _SYM_WORDS:
        if w not in plain_text:
            continue
        # Prefer a specific phrase: "MODIFIER WORD" or "WORD in/of LOCATION"
        m = re.search(
            rf"(\w+(?:\s\w+)?\s{re.escape(w)}|{re.escape(w)}\s(?:in|of)\s\w+(?:\s\w+)?)",
            plain_text,
        )
        phrase = m.group(0).strip() if m else w
        # Strip leading junk words (e.g. "or muscle pain" → "muscle pain")
        phrase = re.sub(r"^(?:or|and|the|a|an|of|in|is|are|include[sd]?)\s+", "", phrase)
        if phrase and phrase != disease_lower and phrase not in candidates:
            candidates.append(phrase)

    return list(dict.fromkeys(candidates))  # deduplicate, preserve order


def _wikipedia_ols4_backend(icd10: str, name: str,
                             limit: int) -> list[SnomedCoding]:
    """Wikipedia S&S section / REST summary → symptom names → SNOMED via OLS4.

    No API key required.  Works for conditions that lack Wikidata P780 data
    (e.g. most musculoskeletal pain conditions).
    """
    disease_lower = name.lower()

    # Try the diagnosis name, then a cleaned version (strip qualifiers)
    clean = re.sub(r"[,;]\s*(?:unspecified|nos|not otherwise specified)\b.*",
                   "", name, flags=re.IGNORECASE).strip()
    clean = re.sub(r"\s+(?:without|with|unspecified)\s+.*", "", clean,
                   flags=re.IGNORECASE).strip()

    title = _wp_find_title(name) or (_wp_find_title(clean) if clean != name else None)
    if not title:
        return []

    candidates = _wp_extract_candidates(title, disease_lower)
    if not candidates:
        return []

    print(f"  [Wikipedia] '{title}' -> {len(candidates)} candidates: "
          f"{candidates[:4]}")

    findings: list[SnomedCoding] = []
    seen: set[str] = set()
    for term in candidates[: limit * 2]:
        # Skip circular references
        if (term in disease_lower or disease_lower in term
                or len(term) < 4):
            continue
        sc = _ols4_snomed_code(term)
        if sc and sc.code not in seen:
            seen.add(sc.code)
            findings.append(sc)
        if len(findings) >= limit:
            break

    print(f"  [Wikipedia+OLS4] {len(findings)} SNOMED codes resolved")
    return findings


# ── Backend 3: BioPortal (legacy) ────────────────────────────────────────────

def _bp_search_disorder(name: str, api_key: str) -> SnomedCoding | None:
    r = requests.get(
        f"{_BP_BASE}/search",
        params={"q": name, "ontologies": "SNOMEDCT",
                "require_exact_match": "false", "pagesize": 3,
                "apikey": api_key, "display_context": "false",
                "display_links": "false"},
        headers={"Accept": "application/json"}, timeout=15, verify=False,
    )
    if r.status_code != 200:
        return None
    coll = r.json().get("collection", [])
    if not coll:
        return None
    top = coll[0]
    return SnomedCoding(code=_sctid_from_uri(top["@id"]),
                        display=top.get("prefLabel", ""))




# ── Backend 4: FHIR ECL ───────────────────────────────────────────────────────

def _fhir_ecl(base: str, ecl: str, count: int = 25,
              timeout: int = 20) -> list[dict]:
    url = f"{SNOMED_SYSTEM}?fhir_vs=ecl/{urllib.parse.quote(ecl)}"
    try:
        r = requests.get(f"{base}/ValueSet/$expand",
                         params={"url": url, "count": count},
                         headers={"Accept": "application/fhir+json"},
                         timeout=timeout, verify=False)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    return r.json().get("expansion", {}).get("contains", [])


def _fhir_resolve_disorder(icd10: str, name: str) -> SnomedCoding | None:
    """Find disorder SCTID via FHIR ICD-10 map-refset or text ECL."""
    map_ecl  = f'^ 447562003 {{{{ M mapTarget = "{icd10}" }}}}'
    text_ecl = f'< 64572001 {{{{ term = "{name}" }}}}'
    for base in _FHIR_TX_SERVERS:
        for ecl in (map_ecl, text_ecl):
            items = _fhir_ecl(base, ecl, count=3)
            if items and items[0].get("code"):
                return SnomedCoding(code=items[0]["code"],
                                    display=items[0].get("display", ""))
    return None


def _fhir_associated_findings(disorder: SnomedCoding,
                              limit: int) -> list[SnomedCoding]:
    """ECL for clinical findings with 'Associated with' attribute = disorder."""
    ecl = f"<< 404684003 : 246090004 = << {disorder.code}"
    for base in _FHIR_TX_SERVERS:
        items = _fhir_ecl(base, ecl, count=limit)
        if items:
            return [SnomedCoding(code=i["code"], display=i.get("display", ""))
                    for i in items if i.get("code")]
    return []


# ── OLS4 disorder search (disorder-SCTID fallback) ────────────────────────────

def _ols4_search_disorder(name: str) -> SnomedCoding | None:
    """Find disorder SNOMED code by name via EBI OLS4."""
    for exact in (True, False):
        sc = _ols4_snomed_code(name)  # reuses quality-filtered search
        if sc:
            return sc
    return None


# ── Main public API ───────────────────────────────────────────────────────────

def get_disease_symptoms(icd10_code: str, *,
                         diagnosis_name: str | None = None,
                         max_symptoms: int = 15) -> DiseaseSymptoms:
    description = _icd10_to_description(icd10_code)
    if not description:
        print(f"  [icd] {icd10_code!r} not in ICD-10 WHO")
        return DiseaseSymptoms(icd10_code, None, None, [])
    print(f"  [icd] {icd10_code} = {description}")

    name = diagnosis_name or description

    # ── Step 1: find disorder SCTID ──────────────────────────────────────
    disorder: SnomedCoding | None = None

    # 1a. BioPortal (if key available)
    bp_key = _api_key("BIOPORTAL_API_KEY")
    if disorder is None and bp_key:
        try:
            disorder = _bp_search_disorder(description, bp_key)
            if disorder:
                print(f"  [BioPortal] disorder {disorder.code} ({disorder.display})")
        except Exception as e:
            print(f"  [BioPortal] search failed: {e}")

    # 1c. FHIR ECL map-refset
    if disorder is None:
        disorder = _fhir_resolve_disorder(icd10_code, description)
        if disorder:
            print(f"  [FHIR-tx]   disorder {disorder.code} ({disorder.display})")

    # 1d. EBI OLS4 lexical search
    if disorder is None:
        disorder = _ols4_snomed_code(description)
        if disorder:
            print(f"  [OLS4]      disorder {disorder.code} ({disorder.display})")

    if disorder is None:
        print("  [warn] no disorder SCTID found — symptoms may be limited")

    # ── Step 2: find associated clinical findings ─────────────────────────
    findings: list[SnomedCoding] = []

    # 2a. Wikidata P780 + EBI OLS4
    if not findings:
        findings = _wikidata_ols4_backend(icd10_code, name, max_symptoms)

    # 2c. Wikipedia "Signs and symptoms" + OLS4 (no key required)
    if not findings:
        findings = _wikipedia_ols4_backend(icd10_code, name, max_symptoms)

    # 2d. FHIR ECL
    if not findings and disorder:
        findings = _fhir_associated_findings(disorder, max_symptoms)
        print(f"  [FHIR-tx]   {len(findings)} findings via ECL")

        # If the chosen disorder yields no findings, try alternate SCTID
        if not findings:
            alt = _fhir_resolve_disorder(icd10_code, description)
            if alt and disorder and alt.code != disorder.code:
                print(f"  [retry]     alternative disorder {alt.code} ({alt.display})")
                findings = _fhir_associated_findings(alt, max_symptoms)
                print(f"  [FHIR-tx]   {len(findings)} findings (alt)")
                if findings:
                    disorder = alt

    return DiseaseSymptoms(
        icd10=icd10_code,
        icd10_description=description,
        snomed_disorder=disorder,
        symptoms=[SymptomFinding(snomed=f) for f in findings],
    )


def get_symptoms_snomed(icd10_code: str,
                        diagnosis_name: str | None = None) -> list[str]:
    """Legacy helper: returns just the symptom display strings."""
    try:
        ds = get_disease_symptoms(icd10_code, diagnosis_name=diagnosis_name)
    except Exception as e:
        print(f"  [error] {e}")
        return []
    return [s.snomed.display for s in ds.symptoms]


# ── CLI ───────────────────────────────────────────────────────────────────────
# Usage:
#   python src/symptoms_snomed.py E11.9
#   python src/symptoms_snomed.py C50  --max 8
#   python src/symptoms_snomed.py I21.9

def _cli(argv: list[str]) -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

    args = argv[1:]
    icd10 = args[0] if args and not args[0].startswith("--") else "E11.9"
    max_s = int(args[args.index("--max") + 1]) if "--max" in args else 10

    print(f"\n=== get_disease_symptoms({icd10!r}) ===")
    ds = get_disease_symptoms(icd10, max_symptoms=max_s)
    if not ds.symptoms:
        print("\nNo symptoms returned. "
              "Add BIOPORTAL_API_KEY to .env for better coverage.")
        return 1

    print(f"\nDisorder : {ds.snomed_disorder}")
    print(f"Symptoms ({len(ds.symptoms)}):")
    for i, sf in enumerate(ds.symptoms, 1):
        print(f"  {i:>2}. [{sf.snomed.code:>11}] {sf.snomed.display}")

    obs = ds.symptoms[0].to_fhir_observation(
        subject_ref="urn:uuid:11111111-1111-4111-8111-111111111111",
        encounter_ref="urn:uuid:33333333-3333-4333-8333-333333333333",
        present=True,
        onset_iso="2026-04-04T06:15:00Z",
        severity_code="6736007",
        observation_id="symptom-001",
    )
    #print("\n--- FHIR R4 Observation for the first symptom ---")
    #print(json.dumps(obs, indent=2, ensure_ascii=False))
    #print("\n--- Full DiseaseSymptoms.to_dict() ---")
    #print(json.dumps(ds.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
