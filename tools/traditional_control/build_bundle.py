"""Build the reproducible Taili traditional-control bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from products.taili.traditional_control.packaging import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    result = build_bundle(args.output_root)
    print(result["artifact_digest"])
    print(result["archive"])
    print(result["manifest"])
    print(result["run_manifest"])


if __name__ == "__main__":
    main()
