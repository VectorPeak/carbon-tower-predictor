from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STEPS = [
    PROJECT_ROOT / "scripts" / "step1_merge_16_vars.py",
    PROJECT_ROOT / "scripts" / "step2_feature_selection_lag_correlation.py",
    PROJECT_ROOT / "scripts" / "step3_dataset_diff_prediction.py",
    PROJECT_ROOT / "scripts" / "step4_model_training.py",
    PROJECT_ROOT / "scripts" / "step5_rolling_evaluation.py",
]


def run_go() -> int:
    for step in STEPS:
        print(f"\n=== Running {step.name} ===", flush=True)
        subprocess.run([sys.executable, str(step)], cwd=PROJECT_ROOT, check=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"go", "-go"}:
        return run_go()

    print("Usage:")
    print("  carbon-pipeline go")
    print("  carbon-pipeline -go")
    print("  carbon-go")
    return 0


def main_go() -> int:
    return run_go()


if __name__ == "__main__":
    raise SystemExit(main())
