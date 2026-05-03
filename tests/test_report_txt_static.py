from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_report_command_writes_txt_instead_of_printing_signal():
    cli = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    report_block = cli.split("def cmd_report", 1)[1].split("def cmd_daily", 1)[0]
    assert "write_txt_report" in report_block
    assert "print_signal(signal)" not in report_block
    assert "report written:" in report_block


def test_daily_generate_report_is_yaml_controlled_and_not_cli_flag():
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["daily"]["generate_report"] is False
    cli = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    assert "--report" not in cli
    assert "generate_report" in cli


def test_txt_report_renderer_exists_and_contains_audit_sections():
    report = (ROOT / "app" / "report.py").read_text(encoding="utf-8")
    assert "def write_txt_report" in report
    assert "def render_txt_report" in report
    assert "TRADEGEM AUDIT REPORT" in report
    assert "BASE ENGINES AND ARBITER" in report
    assert "AUTOTUNE SUMMARY" in report
