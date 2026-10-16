from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SECTION = (ROOT / "report/sections/07_limits.tex").read_text()
PATH_SOURCES = (ROOT / "report/sections/06_analysis.tex").read_text() + SECTION
ROOTED = ("results/", "eval_sets/", "data/", "scripts/", "src/", "notebooks/", "report/")


def listed_paths():
    return re.findall(r"\\path\{((?:[^{}]|\{[^{}]*\})*)\}", PATH_SOURCES)


def expand_braces(pattern):
    found = re.search(r"\{([^{}]*)\}", pattern)
    if not found:
        return [pattern]
    return [name for option in found.group(1).split(",") for name in expand_braces(pattern[:found.start()] + option + pattern[found.end():])]


def matches(pattern, base=ROOT):
    return all(any(base.glob(p.rstrip("/") + ("/*" if p.endswith("/") else ""))) for p in expand_braces(pattern))


def test_every_repository_path_in_the_reproducibility_table_exists():
    assert "\\label{tab:paths}" in PATH_SOURCES
    paths = listed_paths()
    assert len(paths) >= 14
    assert expand_braces("results/{sft,dpo_hh}/config.json") == ["results/sft/config.json", "results/dpo_hh/config.json"]
    missing = []
    for p in paths:
        if p.startswith(ROOTED):
            ok = matches(p)
        elif p.startswith("."):
            ok = matches("*" + p, ROOT / "results/generations")
        else:
            ok = matches(p, ROOT / "results/judgments")
        if not ok:
            missing.append(p)
    assert missing == []


def test_the_adapter_and_metrics_paths_point_at_files_the_report_describes():
    assert (ROOT / "results/lora/final_adapter/persona_lora_r16_lr5e-5_rpo1/adapter_model.safetensors").exists()
    assert len(list((ROOT / "results/lora/runs").glob("*_metrics.json"))) == 12
    assert len(list((ROOT / "notebooks").glob("gen-ai-pa2-*.ipynb"))) == 14


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_results_are_unchanged_since_the_commit_named_in_the_report():
    named = re.search(r"since commit \\texttt\{([0-9a-f]+)\}", SECTION).group(1)
    last = subprocess.run(["git", "log", "-1", "--format=%h", "--", "results"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert last.startswith(named) or named.startswith(last)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "results"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert dirty == ""
