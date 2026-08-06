#!/usr/bin/env python3
"""Run annotator → train → export (after landmark #56 annotations exist)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="SHGNet-56 pipeline entry")
    p.add_argument(
        "step",
        choices=["annotate", "train", "export", "all"],
        help="annotate=Gradio | train=3-stage | export=ONNX | all=train+export",
    )
    p.add_argument("--skip-stage3", action="store_true")
    args, rest = p.parse_known_args()

    if args.step == "annotate":
        return subprocess.call([sys.executable, str(ROOT / "annotator" / "app.py")] + rest)

    if args.step in ("train", "all"):
        cmd = [sys.executable, "-m", "train.train"] + rest
        if args.skip_stage3:
            cmd.append("--skip-stage3")
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            return rc
    if args.step in ("export", "all"):
        return subprocess.call([sys.executable, "-m", "train.export_onnx"] + rest, cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
