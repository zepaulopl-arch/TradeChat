from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_report_command_writes_txt_instead_of_printing_signal():
    text = (ROOT / "app" / "commands" / "signal_command.py").read_text(encoding="utf-8")
    report_block = text.split("def _report", 1)[1].split("def run", 1)[0]
    assert "write_txt_report" in report_block
    assert "print_signal(signal)" not in report_block
    assert "report written" in report_block
    assert "report_path" not in report_block


def test_removed_daily_report_toggle_is_not_in_cli_or_config():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert "daily" not in cfg
    cli = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    assert "--report" not in cli
    assert "generate_report" not in cli


def test_txt_report_renderer_exists_and_contains_audit_sections():
    report = (ROOT / "app" / "report.py").read_text(encoding="utf-8")
    assert "def write_txt_report" in report
    assert "def render_txt_report" in report
    assert "TRADECHAT AUDIT REPORT" in report
    assert "DECISION PATH" in report
    assert "BASE ENGINES AND ARBITER" in report
    assert "AUTOTUNE SUMMARY" in report
