from app.cli import build_parser


def test_validate_matrix_parser_surface_is_human_facing():
    parser = build_parser()
    args = parser.parse_args(
        [
            "validate",
            "matrix",
            "--universe",
            "ibov",
            "--jobs",
            "6",
            "--resume",
            "--skip-data-audit",
        ]
    )
    assert args.command == "validate"
    assert args.tickers == ["matrix"]
    assert args.universe == "ibov"
    assert args.jobs == 6
    assert args.resume is True
    assert args.skip_data_audit is True


def test_validate_report_parser_surface_is_human_facing():
    parser = build_parser()
    args = parser.parse_args(["validate", "report", "logs/policy_matrix/x", "--out-dir", "out"])
    assert args.command == "validate"
    assert args.tickers == ["report", "logs/policy_matrix/x"]
    assert args.out_dir == "out"


def test_validate_command_uses_internal_matrix_tools():
    from pathlib import Path

    text = Path("app/commands/validate_command.py").read_text(encoding="utf-8")
    assert "from tools import run_policy_matrix" in text
    assert "from tools import analyze_policy_matrix" in text
    assert "--skip-full-universe" in text
