from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import report_extras as rx
import summarize_judgments as sj

MAIN = (ROOT / "report/main.tex").read_text()
TITLE = " ".join((ROOT / "report/sections/title.tex").read_text().split())
ANALYSIS = (ROOT / "report/sections/06_analysis.tex").read_text()
TRAINING = (ROOT / "report/sections/04_training.tex").read_text()


def stats(name):
    return sj.stats(sj.load_last(ROOT / f"results/judgments/{name}_judged.jsonl"))


def test_heading_and_author_block_follow_the_requested_form():
    assert "\\newcommand{\\reportassignment}{Programming Assignment 2}" in MAIN
    assert "\\newcommand{\\reporttitle}{Instruction Tuning and Direct Preference Optimization on a Small Language Model}" in MAIN
    assert "\\reportassignment:\\\\ Instruction Tuning and Direct Preference Optimization\\\\ on a Small Language Model" in TITLE
    for macro, value in (("reportauthor", "Syed Muhammad Ashhar Shah"), ("reportroll", "25030009"), ("reportprogram", "MSCS"), ("reportinstructor", "Dr Agha Ali Raza")):
        assert f"\\newcommand{{\\{macro}}}{{{value}}}" in MAIN and f"\\{macro}" in TITLE
    assert "ABSTRACT" in TITLE and "Additional Key Words and Phrases" in TITLE


def test_layout_is_single_column_with_no_cover_page():
    assert "\\documentclass[10pt,letterpaper]{article}" in MAIN
    for word in ("twocolumn", "titlepage", "coverpage", "table*", "figure*"):
        assert word not in MAIN + TITLE + ANALYSIS + TRAINING
    assert "\\input{sections/01_repo_data}" in MAIN and "00_intro" not in MAIN
    sections = re.findall(r"\\input\{sections/(\d\d)_", MAIN)
    assert sections == ["01", "02", "03", "04", "05", "06", "07"]


def test_abstract_numbers_match_the_results():
    hh, uf, rep = stats("dpo_hh_vs_sft_hh300"), stats("dpo_uf_vs_sft_alpaca300"), stats("dpo_uf_vs_sft_alpaca300_rep1p1")
    assert f"win rate {hh['win_rate_excl_ties']:.3f}, $p={hh['sign_test_p']:.3f}$" in TITLE
    assert f"AlpacaEval ({uf['win_rate_excl_ties']:.3f})" in TITLE and f"a tie ({rep['win_rate_excl_ties']:.3f})" in TITLE
    for needle in ("Qwen2.5-0.5B", "30,000 UltraChat examples", "15,000 HH-RLHF pairs", "15,000 UltraFeedback pairs", "300 fixed prompts"):
        assert needle in TITLE


def test_figure_two_is_referenced_for_the_claims_it_supports():
    assert ANALYSIS.count("\\label{fig:behavior}") == 1 and "assets/fig_behavior.png" in ANALYSIS
    for panel in ("a", "b", "d"):
        assert f"Figure~\\ref{{fig:behavior}}{panel}" in ANALYSIS
    assert "Figure~\\ref{fig:behavior}c and d" in TRAINING


def test_compact_figure_is_built_at_its_print_size(tmp_path):
    import matplotlib.image as mpimg
    rx.fig_behavior(ROOT, tmp_path)
    image = mpimg.imread(tmp_path / "fig_behavior.png")
    assert image.shape[:2] == (int(3.3 * 300), int(7.0 * 300))


def test_standalone_figures_are_still_built_for_the_repository(tmp_path):
    rx.fig_lengths(ROOT, tmp_path)
    rx.fig_dpo_rewards(ROOT, tmp_path)
    assert (tmp_path / "fig_lengths.png").exists() and (tmp_path / "fig_dpo_rewards.png").exists()


@pytest.mark.skipif(shutil.which("pdfinfo") is None or not (ROOT / "report/main.pdf").exists(), reason="needs a built PDF")
def test_built_report_fits_the_eight_page_limit_including_references():
    info = subprocess.run(["pdfinfo", str(ROOT / "report/main.pdf")], capture_output=True, text=True, check=True).stdout
    assert int(re.search(r"Pages:\s+(\d+)", info).group(1)) <= 8
