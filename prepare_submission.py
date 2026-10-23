#!/usr/bin/env python3
"""Build and verify the submission ZIP from an explicit allowlist.

  python prepare_submission.py                 # -> dist/zentromart_voiceops_2026.zip
  python prepare_submission.py --name team_x   # -> dist/team_x_voiceops.zip

Checks performed after packaging: required files present, forbidden files absent (.env,
virtual environments, dataset copy, runtime artefacts), no secret values from .env inside any
packaged file, clean extraction, and `python run.py --help` working from the extracted copy.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCLUDE = [
    "README.md", "ARCHITECTURE.md", "PROJECT_PLAN.md", "FINAL_STATUS.md", "TEST_REPORT.md", "DEMO_STATUS.md",
    "demo_script.md", "requirements.txt", "requirements-dev.txt", "requirements-lock.txt", ".env.example",
    ".gitignore", "run.py", "prepare_submission.py", "report.pdf", "demo.mp4",
    "src/**/*.py", "web/static/*", "tests/*.py", "scripts/*.py",
    "reports/report.pdf", "reports/eval/*.json", "reports/eval/*.jsonl", "reports/eval/*.md",
    "reports/eval_iterations/**/*",
    "sample_outputs/**/*",
]
REQUIRED = ["README.md", ".env.example", "requirements.txt", "run.py", "src/zentromart/agent.py",
            "web/static/index.html", "report.pdf", "sample_outputs"]
FORBIDDEN = [".env", ".venv/*", "venv/*", "dataset/*", "runtime/*", "*/__pycache__/*", "*.pyc",
             "*customers.csv", "*orders.csv", "*order_items.csv", "*support_logs.csv", "*/kb/*.md",
             "*tickets_created.csv", "*.zip"]


def collect() -> list[Path]:
    files: set[Path] = set()
    for pattern in INCLUDE:
        for p in ROOT.glob(pattern):
            if p.is_file():
                files.add(p)
    out = []
    for p in sorted(files):
        rel = p.relative_to(ROOT).as_posix()
        if any(fnmatch.fnmatch(rel, f) for f in FORBIDDEN):
            continue
        out.append(p)
    return out


def secret_values() -> list[str]:
    vals = []
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if "KEY" in k.upper() and len(v) >= 12:
                    vals.append(v)
    for k, v in os.environ.items():
        if k.endswith("_API_KEY") and len(v) >= 12:
            vals.append(v)
    return list(set(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="zentromart")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{args.name}_voiceops_2026.zip"
    files = collect()
    top = "VoiceOps2026_ZentroMart"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, f"{top}/{p.relative_to(ROOT).as_posix()}")
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB, {len(files)} files)")

    problems = []
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            problems.append(f"corrupt member {bad}")
        names = [n[len(top) + 1:] for n in zf.namelist()]
        for req in REQUIRED:
            if not any(n == req or n.startswith(req.rstrip("/") + "/") for n in names):
                problems.append(f"missing required: {req}")
        for n in names:
            if any(fnmatch.fnmatch(n, f) for f in FORBIDDEN):
                problems.append(f"forbidden file packaged: {n}")
        secrets = secret_values()
        for n in names:
            data = zf.read(f"{top}/{n}")
            for s in secrets:
                if s.encode() in data:
                    problems.append(f"SECRET VALUE FOUND in {n}")
        if "demo.mp4" not in names:
            print("NOTE: demo.mp4 not packaged (see DEMO_STATUS.md)")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        proj = Path(tmp) / top
        r = subprocess.run([sys.executable, "run.py", "--help"], cwd=proj, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or "--audio-dir" not in r.stdout:
            problems.append(f"run.py --help failed in extracted copy: {r.stderr[-300:]}")
        else:
            print("Extracted copy: `python run.py --help` OK")
    if problems:
        print("VERIFICATION FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"Verification passed: {len(names)} files, required files present, no forbidden files, no secrets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
