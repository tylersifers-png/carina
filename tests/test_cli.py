"""The carina command: global options, help, and configuration."""

import pytest

from carina import cli, settings


def test_help_and_unknown_commands(capsys):
    assert cli.main(["--help"]) == 0
    assert "refresh market" in capsys.readouterr().out
    assert cli.main(["nonsense"]) == 2


def test_global_options_configure_directories(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setitem(cli.COMMANDS, "verify", lambda argv: seen.update(data=settings.data(), argv=argv) or 0)
    assert cli.main(["--data", str(tmp_path), "--decisions", str(tmp_path / "d"), "verify", "-x"]) == 0
    assert seen == dict(data=tmp_path, argv=["-x"])
    assert settings.decisions() == tmp_path / "d"


def test_missing_directories_are_reported(monkeypatch):
    settings.reset()
    monkeypatch.delenv("CARINA_DATA", raising=False)
    with pytest.raises(RuntimeError, match="--data or set CARINA_DATA"):
        settings.data()
    monkeypatch.setenv("CARINA_DATA", "/somewhere")
    assert str(settings.data()) == "/somewhere"
    with pytest.raises(ValueError, match="Unknown decision file"):
        settings.decision("other.json")


def test_decision_files_are_schema_checked(tmp_path):
    (tmp_path / "thresholds.json").write_text('{"schema": "old"}')
    settings.configure(decisions=tmp_path)
    with pytest.raises(ValueError, match="carina-thresholds-1.0"):
        settings.thresholds("market")
