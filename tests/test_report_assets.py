from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import make_report_assets as mra


def test_assets_build_from_the_committed_results(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(ROOT), "--out", str(tmp_path)])
    mra.main()
    names = {p.name for p in tmp_path.iterdir()}
    assert {"table_config.md", "table_generation.md", "table_judge.md", "table_lora_sweep.md", "fig_lora_sweep.png",
            "fig_winrates.png", "fig_training.png", "key_numbers.json", "table_data.md", "table_eval_sets.md", "table_deviations.md",
            "table_judge_reliability.md", "table_judge_consistency.md", "table_judge_bias.md", "table_hh_categories.md",
            "table_alpaca_sources.md", "table_hh_refusals.md", "fig_dpo_rewards.png", "fig_lora_training.png", "fig_lengths.png",
            "fig_hh_categories.png", "examples.md", "examples_persona.md", "index.md", "table_data.tex", "table_eval_sets.tex", "table_deviations.tex", "table_lora_sweep.tex", "table_dpo_results.tex", "table_lora_results.tex"} <= names
    selection = json.loads((ROOT / "results/lora/v2/persona_v2_selection.json").read_text())
    key = json.loads((tmp_path / "key_numbers.json").read_text())
    assert key["persona"]["selected"] == selection["selected"]
    sweep = (tmp_path / "table_lora_sweep.md").read_text()
    assert sweep.count("(selected)") == 1 and selection["selected"] in sweep
    sweep_tex = (tmp_path / "table_lora_sweep.tex").read_text()
    assert sweep_tex.count("(selected)") == 1 and sweep_tex.count(r"\\") == 1 + 1 + 12
    deviations = (tmp_path / "table_deviations.md").read_text()
    assert "validation batch size" in deviations and "| resource |" in deviations
    judge = (tmp_path / "table_judge.md").read_text()
    assert "POSITIVE CONTROL" in judge and "(selected)" in judge
    assert "win rate (ties = 1/2)" in judge and "Judge: google/gemini-3.6-flash" in judge and "excluded from both win rates" in judge
    config = (tmp_path / "table_config.md").read_text()
    assert "generation settings (evaluation)" in config and "rpo_alpha" in config


def test_assets_script_tolerates_missing_persona_results(tmp_path):
    empty_root = tmp_path / "root"
    empty_root.mkdir()
    assert mra.persona_context(empty_root) is None
    assert mra.judged_row(empty_root, "anything") is None
