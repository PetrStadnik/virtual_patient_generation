from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.bundle_to_llm import bundle_from_file, bundle_to_eval_input
from evaluace.agent1_gate import evaluate_gate
from src.LLM_generation.llm_io import FullPatient
from evaluace.agent2_symptoms import evaluate_symptoms
from evaluace.agent3_quality import evaluate_quality
from simulation_app.sim.runner import evaluate_diagnosis


def evaluate_scenario(model: str, eval_input: FullPatient) -> dict:
    """Run the full evaluation pipeline for one patient scenario.

    Returns a dict with results from all four agents so callers can
    aggregate and persist results across multiple scenarios.
    """
    gate_result      = evaluate_gate(model, eval_input)
    symptoms_result  = evaluate_symptoms(model, eval_input)
    quality_result   = evaluate_quality(model, eval_input)
    diagnosis_result = evaluate_diagnosis(model, eval_input)

    return {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "model":          model,
        "icd_code":       eval_input.icd_code,
        "diagnosis_name": eval_input.diagnosis_name,
        "gender":         eval_input.gender,
        "age":            eval_input.age,
        "gate":           gate_result.model_dump(),
        "symptoms":       symptoms_result.model_dump(),
        "quality":        quality_result.model_dump(),
        "diagnosis":      diagnosis_result.model_dump(),
    }


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record as a single line to a .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from dotenv import load_dotenv

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env")

    # ── Configuration ──────────────────────────────────────────────────────────
    MODEL          = "gpt-5.4" #"claude-sonnet-4-6"
    SCENARIOS_DIR  = PROJECT_ROOT / "scenarios" / "multiagent" / "anthropic"
    # Output file: results/<model>_<date>.jsonl  (one record per patient)
    safe_model     = MODEL.replace(":", "_").replace("/", "_")
    OUTPUT_FILE    = PROJECT_ROOT / "evaluace" / "results" / "multiagent_anthropic.jsonl"

    scenarios = sorted(SCENARIOS_DIR.rglob("*"))
    scenarios = [s for s in scenarios if s.is_file()]

    if not scenarios:
        print(f"No scenario files found in {SCENARIOS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Model      : {MODEL}")
    print(f"Scenarios  : {len(scenarios)}")
    print(f"Output     : {OUTPUT_FILE}\n")

    for i, scenario_path in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario_path.name}")
        try:
            fhir_bundle = bundle_from_file(scenario_path)
            eval_input  = bundle_to_eval_input(fhir_bundle)
            result      = evaluate_scenario(MODEL, eval_input)

            # Add source file for traceability
            result["source_file"] = scenario_path.name

            _append_jsonl(OUTPUT_FILE, result)
            print(f"  -> saved to {OUTPUT_FILE.name}\n")

        except Exception as e:
            error_record = {
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "model":       MODEL,
                "source_file": scenario_path.name,
                "error":       str(e),
            }
            _append_jsonl(OUTPUT_FILE, error_record)
            print(f"  ERROR: {e}\n")

