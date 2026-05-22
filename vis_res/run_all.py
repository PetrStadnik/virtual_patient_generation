"""
run_all.py
==========
Run all visualisation scripts in the correct order.

Usage:
    python vis_res/run_all.py

Figures are saved to vis_res/figures/ as both .pdf (thesis quality) and .png (preview).
"""
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEPARATOR = "-" * 60


def run_script(name: str, module_main):
    print(f"\n{SEPARATOR}")
    print(f"  {name}")
    print(SEPARATOR)
    t0 = time.time()
    module_main()
    print(f"  Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    from vis_res.fig_validation import main as val_main
    from vis_res.fig_evaluation import main as eval_main
    from vis_res.fig_questions  import main as q_main
    from vis_res.export_summary  import main as sum_main

    print("=" * 60)
    print("  Virtual-patient pipeline - result visualisation")
    print("=" * 60)

    run_script("1 / 4  FHIR validation figures",       val_main)
    run_script("2 / 4  Evaluation pipeline figures",   eval_main)
    run_script("3 / 4  Gate + quality question figures", q_main)
    run_script("4 / 4  Text + JSON summary export",    sum_main)

    print("\n" + "=" * 60)
    print("  Figures:   vis_res/figures/")
    print("  Summary:   vis_res/results_summary.json")
    print("  Narrative: vis_res/results_narrative.txt")
    print("=" * 60)
