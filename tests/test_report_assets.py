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
            "fig_winrates.png", "fig_training.png", "key_numbers.json"} <= names
    selection = json.loads((ROOT / "results/lora/v2/persona_v2_selection.json").read_text())
    key = json.loads((tmp_path / "key_numbers.json").read_text())
    assert key["persona"]["selected"] == selection["selected"]
    sweep = (tmp_path / "table_lora_sweep.md").read_text()
    assert sweep.count("(selected)") == 1 and selection["selected"] in sweep
    judge = (tmp_path / "table_judge.md").read_text()
    assert "POSITIVE CONTROL" in judge and "(selected)" in judge
    config = (tmp_path / "table_config.md").read_text()
    assert "generation settings (evaluation)" in config and "rpo_alpha" in config


def test_assets_script_tolerates_missing_persona_results(tmp_path):
    empty_root = tmp_path / "root"
    empty_root.mkdir()
    assert mra.persona_context(empty_root) is None
    assert mra.judged_row(empty_root, "anything") is None
