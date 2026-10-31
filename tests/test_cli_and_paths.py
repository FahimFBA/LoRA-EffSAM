# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""CLI dispatch and path configuration."""
import importlib

import pytest

from lora_effsam import cli, paths


def test_every_subcommand_module_has_main():
    for cmd, (module, _) in cli.COMMANDS.items():
        if module is None:
            continue
        assert callable(getattr(importlib.import_module(module), "main")), cmd


def test_usage_and_unknown_command(capsys):
    assert cli.main([]) == 0
    assert "train" in capsys.readouterr().out
    assert cli.main(["no-such-command"]) == 2


def test_subcommand_help_reaches_module(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["train", "--help"])
    assert e.value.code == 0
    assert "--deep-supervision" in capsys.readouterr().out


def test_paths_follow_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LORA_EFFSAM_RUNS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("LORA_EFFSAM_DATA_DIR", str(tmp_path / "d"))
    assert paths.runs_dir() == tmp_path / "r" and paths.data_dir() == tmp_path / "d"
    monkeypatch.delenv("LORA_EFFSAM_RUNS_DIR")
    assert paths.runs_dir() == paths.REPO_ROOT / "runs"
