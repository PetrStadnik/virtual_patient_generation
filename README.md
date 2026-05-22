# Virtual Patient Pipeline

A research pipeline that generates synthetic FHIR R4 patient scenarios using LLMs, evaluates them with a multi-agent evaluation suite, and benchmarks doctor-diagnosis accuracy via a simulated consultation app.

---

## Project structure

```
fhir1/
├── main.py                        # Entry point — batch generation
├── .env                           # API keys (not included)
├── data/                          # GBD / IHME source datasets 
│
├── src/                           # Core shared library
│   ├── LLM_generation/
│   │   ├── llm_io.py              # Pydantic data models (PatientOutput, FullPatient, …)
│   │   ├── LLM2FHIR.py            # Converts PatientOutput → FHIR R4 Bundle JSON
│   │   ├── baseline_script.py     # Generator: plain LLM prompt (no tools, no RAG)
│   │   └── agent_with_tools.py    # Generator: LLM + medical lookup tools
│   ├── basic_data/
│   │   ├── GBD.py                 # ICD-10 → age/sex probability from IHME GBD 2023
|   |   ├── eurostat_data.py       # finally not used 
│   │   └── name_generator.py      # Country-aware patient name generator
│   ├── bundle_base.py             # FHIR bundle builder + structural validator
│   ├── bundle_to_llm.py           # Parses FHIR bundle JSON → PatientOutput / FullPatient
│   ├── icd_lookup.py              # ICD-10 code validation and hierarchy lookup
│   └── symptoms_snomed.py         # SNOMED CT symptom lookup (BioPortal → FHIR ECL → OLS4 → Wikidata)
│
├── multiagentic/                  # Generator: LangGraph generate → evaluate → fix loop
│   ├── run.py                     # Public API — generate_patient()
│   ├── graph.py                   # LangGraph pipeline definition
│   └── agents/
│       ├── generation.py          # Initial generation node
│       └── fix.py                 # Fix node (corrects failed evaluations)
│
├── rag/                           # Generator: RAG-augmented generation
│   ├── setup.py                   # One-time dataset download + index build
│   ├── ingest/                    # Data fetchers (one per source)
│   │   ├── fetch_pubmed.py
│   │   ├── fetch_medlineplus.py
│   │   ├── fetch_snomed.py
│   │   ├── fetch_fhir_examples.py
│   │   └── fetch_meddialog.py
│   ├── retrieve/
│   │   └── index.py               # TF-IDF index builder + Retriever
│   ├── generate/
│   │   ├── prompt.py              # System prompt for RAG generation
│   │   └── run.py                 # generate_patient_rag_with_tools()
│   ├── data/                      # Raw fetched documents
│   └── index/                     # Built index artefacts
│
├── evaluace/                      # Evaluation pipeline (4 agents)
│   ├── evaluation_pipeline.py     # Orchestrator — runs all 4 agents on a scenario
│   ├── agent1_gate.py             # Gate: medical plausibility check
│   ├── agent2_symptoms.py         # Symptoms: SNOMED + RAG + LLM coverage check
│   └── agent3_quality.py          # Quality: structured YES/NO questionnaire
│
├── simulation_app/                # Django app — interactive doctor-patient simulation
│   ├── manage.py
│   ├── core/settings.py           # Django settings + SIMULATION_BUNDLE path
│   └── sim/
│       ├── runner.py              # evaluate_diagnosis() — API
│       └── views.py               # Web UI streaming consultation
│
├── scenarios/                     # Generated FHIR bundle JSONs (output)
│   └── {pipeline}/{model}/*.json
│
├── evaluace/results/              # Evaluation results (output)
│   └── {pipeline}_{model}.jsonl
│
└── vis_res/                       # Visualisation scripts → thesis figures
    ├── run_all.py
    ├── fig_validation.py
    ├── fig_evaluation.py
    ├── fig_questions.py
    ├── export_summary.py
    └── figures/                   # Generated PDFs and PNGs (output)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create `.env`

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
BIOPORTAL_API_KEY=...           # used for SNOMED disorder lookup
```

### 3. Download GBD data

Place these two files in `data/`:

- `IHME_GBD_2021_NONFATAL_CAUSE_ICD_CODE_MAP_Y2024M05D16_0.XLSX` — ICD-10 → GBD cause mapping
- `IHME-GBD_2023_DATA-e683deec-1.csv` — age/sex prevalence distributions

Both are freely downloadable from [IHME GBD Results](https://vizhub.healthdata.org/gbd-results/).

### 4. Build RAG index (optional, only for RAG / multiagentic pipelines)

```bash
python rag/setup.py           # download 5 seed codes + build index
python rag/setup.py --check   # show what is already downloaded
python rag/setup.py --rebuild # re-download everything + rebuild
```

---

## Data models (`src/LLM_generation/llm_io.py`)

All generators and evaluators share these Pydantic models:


| Model              | Purpose                                                                              |
| ------------------ | ------------------------------------------------------------------------------------ |
| `PatientOutput`    | LLM output: description, history, tests, meds, allergies, family history, procedures |
| `PatientInput`     | LLM input: icdcode, diagnosisname, gender, age                                       |
| `FullPatient`      | Wraps `PatientOutput` + icdcode, diagnosisname, gender, age, name                    |
| `MedicalTest`      | LOINC-coded lab/vital test with value                                                |
| `Medication`       | ATC-coded drug with dose, route, frequency                                           |
| `Allergy`          | Allergen with severity and reaction                                                  |
| `FamilyHistory`    | Relative with ICD-10 condition                                                       |
| `MedicalProcedure` | SNOMED-coded past procedure                                                          |


Validators enforce code formats: ATC `A00AA00`, LOINC `NNNNN-N`, ICD-10 `X00.0`.

---

## Generation pipelines

### Settings (in `main.py`)

```python
MODEL: str = LLM_model.OPENAI   # "gpt-5.4"  or  "claude-sonnet-4-6"
OUTPUT_FOLDER: Path = PROJECT_ROOT / "scenarios" / "multiagent"

INPUT = [
    {"diagnosis_code": "E11.9", "patient_country": "CZ"},
    {"diagnosis_code": "I21.9", "patient_country": "CZ"},
]
```

Run:

```bash
python main.py
```

Output: `scenarios/{pipeline}/{model}/{icd}_{age}_{name}.json` — one FHIR R4 Bundle per patient.

---

### Baseline (`src/LLM_generation/baseline_script.py`)

Single LLM call with a structured prompt. Fastest and cheapest.

```python
from src.LLM_generation.baseline_script import generate_patient
result: dict = generate_patient(
    model="gpt-5.4",
    icd_code="E11.9",
    diagnosis_name="Type 2 diabetes mellitus",
    gender="female",
    age=58,
)
# result is PatientOutput.model_dump()
```

---

### Agent with tools (`src/LLM_generation/agent_with_tools.py`)

LLM with three real-time medical lookup tools:


| Tool                            | Source              | Returns                |
| ------------------------------- | ------------------- | ---------------------- |
| `search_drug(name)`             | Wikidata            | ATC code + description |
| `search_loinc(name)`            | NLM Clinical Tables | LOINC code             |
| `search_snomed_procedure(name)` | CSIRO Ontoserver    | SNOMED CT code         |


```python
from src.LLM_generation.agent_with_tools import generate_patient_with_tools
result: dict = generate_patient_with_tools(
    model="gpt-5.4",
    icd_code="E11.9",
    diagnosis_name="Type 2 diabetes mellitus",
    gender="female",
    age=58,
)
```

---

### Multiagentic (`multiagentic/run.py`)

LangGraph pipeline: **generate → evaluate → fix** (up to 5 iterations). The scenario is only accepted when it passes all three quality thresholds.

```python
from multiagentic.run import generate_patient
result: dict = generate_patient(
    model="gpt-5.4",
    icd_code="E11.9",
    diagnosis_name="Type 2 diabetes mellitus",
    gender="female",
    age=58,
)
```

Thresholds (configurable in `multiagentic/run.py → run()`):


| Parameter              | Default | Meaning                                              |
| ---------------------- | ------- | ---------------------------------------------------- |
| `max_iterations`       | 5       | Maximum generate+fix cycles                          |
| `min_symptom_coverage` | 0.70    | Minimum fraction of expected SNOMED symptoms present |
| `min_quality_ratio`    | 0.77    | Minimum quality score ratio                          |


---

### RAG (`rag/generate/run.py`)

Retrieval-augmented generation: fetches real biomedical evidence for the ICD code on demand, builds/updates a TF-IDF index, then passes the top-k passages to the LLM as grounding context.

**RAG index sources:**


| Source                | Fetched by               | Stored in                     |
| --------------------- | ------------------------ | ----------------------------- |
| PubMed case reports   | `fetch_pubmed.py`        | `rag/data/pubmed/*.jsonl`     |
| MedlinePlus summaries | `fetch_medlineplus.py`   | `rag/data/medlineplus/*.json` |
| SNOMED symptoms       | `fetch_snomed.py`        | `rag/data/snomed/*.json`      |
| FHIR R4 examples      | `fetch_fhir_examples.py` | `rag/data/fhir/*.json`        |
| ChatDoctor dialogues  | `fetch_meddialog.py`     | `rag/data/meddialog/*.jsonl`  |


**How retrieval works:** query `"{diagnosis} {icd_code} {age_band} {gender}"` → TF-IDF cosine similarity → top-6 passages → injected into system prompt.

```python
from rag.generate.run import generate_patient_rag_with_tools
result: dict = generate_patient_rag_with_tools(
    model="gpt-5.4",
    icd_code="E11.9",
    diagnosis_name="Type 2 diabetes mellitus",
    gender="female",
    age=58,
    k=6,   # number of retrieved passages
)
```

If the ICD code is not yet in the index, `ensure_indexed()` fetches data automatically before generation.

---

## Demographics (`src/basic_data/GBD.py`)

Samples realistic age and sex from IHME GBD 2023 prevalence data.

**How the probability is computed:**

1. Filter GBD rows for the matched cause (ICD-10 → GBD cause via `IcdToGbd.lookup()`).
2. Sum prevalence rate (`val`, cases per 100 000) across sexes for each age group.
3. Normalise by the total → `P(age_group | diagnosis)`.
4. Sex probability analogously.
5. Age: pick the most probable group, sample uniformly within its year range.
6. Sex: weighted random sample.

Note: `val` is a prevalence *rate*, not an absolute count.

```python
from src.basic_data.GBD import get_max_probable_age_ang_gender
age, gender = get_max_probable_age_ang_gender("E11.9")
# e.g. age=67, gender="female"
```

---

## Evaluation pipeline (`evaluace/evaluation_pipeline.py`)

Runs four agents on a single `FullPatient` and returns a result dict.

```python
from evaluace.evaluation_pipeline import evaluate_scenario
from src.bundle_to_llm import bundle_from_file, bundle_to_eval_input

bundle = bundle_from_file("scenarios/multiagent/anthropic/E11.9_58_JanaKratochvílová.json")
fp = bundle_to_eval_input(bundle)

result = evaluate_scenario(model="gpt-5.4", eval_input=fp)
```

**Returns:**

```json
{
  "timestamp": "...",
  "model": "gpt-5.4",
  "icd_code": "E11.9",
  "gate":     { "passed": true, "reasons": [] },
  "symptoms": { "symptom_coverage": 0.83, "symptoms_checked": [...] },
  "quality":  { "score": 11, "max": 13, "ratio": 0.85, "no_answers": [] },
  "diagnosis":{ "correct": true, "correct_attempt": 1, "attempts": [...] }
}
```

**Batch evaluation** (all scenarios in a folder):

```python
# In evaluace/evaluation_pipeline.py — configure and run directly:
MODEL         = "gpt-5.4"
SCENARIOS_DIR = PROJECT_ROOT / "scenarios" / "multiagent" / "anthropic"
OUTPUT_FILE   = PROJECT_ROOT / "evaluace" / "results" / "multiagent_anthropic.jsonl"
```

```bash
python evaluace/evaluation_pipeline.py
```

### Agent 1 — Gate (`evaluace/agent1_gate.py`)

Eight YES/NO questions checking medical plausibility. Scenario fails if any `reject_on` condition is triggered.

```python
from evaluace.agent1_gate import evaluate_gate
gate = evaluate_gate(model="gpt-5.4", eval_input=fp)
# gate.passed: bool,  gate.reasons: list[str]
```

### Agent 2 — Symptoms (`evaluace/agent2_symptoms.py`)

Checks how many expected SNOMED symptoms are present in the scenario, using three cascading sources:

1. **SNOMED CT** — symptoms from BioPortal / Wikidata for the ICD code
2. **RAG sub-agent** — 3–10 most typical symptoms extracted from the RAG index
3. **LLM fallback** — if both above return nothing

```python
from evaluace.agent2_symptoms import evaluate_symptoms
syms = evaluate_symptoms(model="gpt-5.4", eval_input=fp)
# syms.symptom_coverage: float (0.0–1.0)
# syms.symptoms_checked: list of {symptom, present, source}
```

> Note: RAG must be indexed beforehand (`rag/setup.py` or prior generation run).

### Agent 3 — Quality (`evaluace/agent3_quality.py`)

Structured YES/NO questionnaire (13 questions) covering completeness and consistency of the scenario. Score = YES answers / total.

```python
from evaluace.agent3_quality import evaluate_quality
q = evaluate_quality(model="gpt-5.4", eval_input=fp)
# q.score: int,  q.max: int,  q.ratio: float,  q.no_answers: list[str]
```

### Agent 4 — Diagnosis simulation (`simulation_app/sim/runner.py`)

The doctor LLM holds up to 6 consultation turns with the patient LLM. If the diagnosis is wrong, it gets corrective feedback and up to 2 more attempts. Match is on the 3-character ICD-10 category (e.g. `E11` matches `E11.9`).

```python
from simulation_app.sim.runner import evaluate_diagnosis
diag = evaluate_diagnosis(model="gpt-5.4", eval_input=fp)
# diag.correct: bool
# diag.correct_attempt: int | None  (1, 2, or 3)
# diag.attempts: list[DiagnosisAttempt]
```

---

## Simulation app (`simulation_app/`)

Interactive web UI for a single doctor-patient consultation.

**Start server:**

```bash
python simulation_app/manage.py runserver
# open http://127.0.0.1:8000
```

**Load a FHIR bundle as the default scenario** — set in `.env` or directly in `simulation_app/core/settings.py`:

```env
SIMULATION_BUNDLE=scenarios/multiagent/anthropic/E11.9_58_JanaKratochvílová.json
```

```python
# simulation_app/core/settings.py
SIMULATION_BUNDLE = BASE_DIR.parent / "scenarios/multiagent/anthropic/E11.9_58_JanaKratochvílová.json"
```

If not set, a built-in asthma demo scenario is used.

---

## FHIR utilities (`src/`)

### `bundle_to_llm.py` — Parse FHIR bundle

```python
from src.bundle_to_llm import bundle_from_file, bundle_to_eval_input, bundle_to_llm_output

bundle = bundle_from_file("scenarios/multiagent/anthropic/E11.9_58_Jana.json")
fp: FullPatient     = bundle_to_eval_input(bundle)   # includes metadata (age, icd, name)
po: PatientOutput   = bundle_to_llm_output(bundle)   # clinical content only
```

### `bundle_base.py` — Build & validate FHIR bundle

```python
from src.bundle_base import BundleBase
bb = BundleBase(patient_output, icd_code="E11.9", gender="female", age=58, name="Jana Kratochvílová")
json_str = bb.to_json()
is_valid = bb.validate_bundle_structure()   # True/False — checks required FHIR fields
```

### `icd_lookup.py` — ICD-10 lookup

```python
from src.icd_lookup import validate_icd_for_fhir, ICDDiagnosisInfo
info: ICDDiagnosisInfo = validate_icd_for_fhir("E11.9")
# info.code, info.description, info.is_valid, info.children
```

---

## Visualisation (`vis_res/`)

Generates thesis-quality figures from evaluation results.

**Data sources:**


| Path                                        | Content                                                 |
| ------------------------------------------- | ------------------------------------------------------- |
| `scenarios/{pipeline}/validation.jsonl`     | FHIR structural validation (one record per scenario)    |
| `evaluace/results/{pipeline}_{model}.jsonl` | Evaluation results (gate, symptoms, quality, diagnosis) |


Naming: `{pipeline}` ∈ `{baseline, tools_agent, rag, multiagent}`, `{model}` ∈ `{openai, anthropic}`.

```bash
python vis_res/run_all.py          # all figures + JSON/text summary
python vis_res/fig_validation.py   # FHIR validation figures only
python vis_res/fig_evaluation.py   # evaluation metric figures only
python vis_res/fig_questions.py    # gate + quality question failure rates
python vis_res/export_summary.py   # results_summary.json + results_narrative.txt
```

**Output figures** (`vis_res/figures/`):


| File                             | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `val_01_structural_validity`     | % structurally valid FHIR bundles           |
| `val_02_code_errors_mean`        | Mean code errors per scenario (±SD)         |
| `val_03_zero_code_errors`        | % scenarios with zero code errors           |
| `val_04_error_type_breakdown`    | ATC / SNOMED / LOINC error split            |
| `eval_01_gate_pass_rate`         | Medical plausibility gate pass rate         |
| `eval_02_symptom_coverage`       | Symptom coverage distribution (box plot)    |
| `eval_03_quality_score`          | Quality score ratio distribution (box plot) |
| `eval_04_diagnosis_accuracy`     | Diagnosis accuracy: 1st vs any attempt      |
| `eval_05_summary_radar`          | Radar chart — all pipelines on 5 metrics    |
| `eval_06_model_comparison`       | GPT-5.4 vs Claude Sonnet 4.6 per pipeline   |
| `eval_07_per_diagnosis_accuracy` | Accuracy per ICD-10 code                    |
| `eval_08_heatmap_icd_pipeline`   | Heatmap: ICD-10 × pipeline accuracy         |
| `q_01_gate_failure_rate`         | Most frequent gate rejection reasons        |
| `q_02_quality_no_rate`           | Most frequent quality NO answers            |
| `q_03_gate_by_pipeline`          | Gate failures broken down by pipeline       |
| `q_04_quality_by_pipeline`       | Quality failures broken down by pipeline    |


Each figure is saved as both `.pdf` (vector, for thesis) and `.png` (preview).

---

## Key dependencies

```
pydantic>=2          # data models and validation
pydantic-ai          # LLM agents (baseline, tools, RAG, evaluation)
langgraph            # agentic pipeline (multiagentic)
openai               # OpenAI API
anthropic            # Anthropic API
httpx                # async HTTP (SSL verify=False for corporate proxy)
requests             # sync HTTP
scikit-learn         # TF-IDF index (RAG)
pandas / numpy       # GBD data processing + visualisation
matplotlib / seaborn # visualisation
django               # simulation web app
python-icd10         # ICD-10 code hierarchy
python-dotenv        # .env loading
and others ...
```

