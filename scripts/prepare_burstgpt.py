"""Prepare pinned BurstGPT windows. Run from the repository root."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_triage.workload.burstgpt import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path, help="Optional local copy; pinned SHA256 is still required")
    parser.add_argument("--requests-per-window", type=int, default=256)
    parser.add_argument("--train-windows", type=int, default=32)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--test-windows", type=int, default=16)
    args = parser.parse_args()
    manifest = prepare(args.data_dir, args.source, args.requests_per_window,
                       dict(train=args.train_windows, validation=args.validation_windows, test=args.test_windows))
    print(json.dumps(dict(counts=manifest["counts"], windows=len(manifest["windows"]),
                          source_sha256=manifest["source_sha256"],
                          manifest=str(args.data_dir / "processed" / "manifest.json")), indent=2))


if __name__ == "__main__":
    main()
