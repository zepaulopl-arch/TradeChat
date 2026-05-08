from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_service_extracts_cli_runtime_helpers():
    text = (ROOT / "app" / "pipeline_service.py").read_text(encoding="utf-8")
    for token in ["def canonical_ticker", "def resolve_tickers", "def build_current_dataset", "def make_signal"]:
        assert token in text


def test_cli_is_parser_shell_and_handlers_own_command_logic():
    cli_text = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    handlers_text = (ROOT / "app" / "cli_handlers.py").read_text(encoding="utf-8")
    assert "from . import cli_handlers as handlers" in cli_text
    assert "def cmd_validate" in handlers_text
    assert "def cmd_portfolio" in handlers_text
    assert "run_pybroker_replay" not in cli_text
    assert len(cli_text.splitlines()) < 140


def test_portfolio_service_centralizes_state_and_signal_access():
    text = (ROOT / "app" / "portfolio_service.py").read_text(encoding="utf-8")
    for token in ["def get_state_db_path", "def load_portfolio_state", "def save_portfolio_state", "def load_latest_signal", "sqlite3.connect"]:
        assert token in text
    assert "portfolio.json" not in text


def test_scoring_module_owns_signal_score_formula():
    text = (ROOT / "app" / "scoring.py").read_text(encoding="utf-8")
    assert "SIGNAL_PRIORITY_MAP" in text
    assert "def signal_score" in text
    assert "math.sqrt" in text


def test_methodology_service_owns_methodology_checks():
    text = (ROOT / "app" / "methodology_service.py").read_text(encoding="utf-8")
    for token in [
        "def check_no_target_features",
        "def check_temporal_split",
        "def check_validation_has_baselines",
        "def check_refine_removal_uses_shadow_artifacts",
        "def methodology_report",
    ]:
        assert token in text


def test_batch_and_simulator_services_exist():
    batch_text = (ROOT / "app" / "batch_service.py").read_text(encoding="utf-8")
    sim_text = (ROOT / "app" / "simulator_service.py").read_text(encoding="utf-8")
    evaluation_text = (ROOT / "app" / "evaluation_service.py").read_text(encoding="utf-8")
    refine_text = (ROOT / "app" / "refine_service.py").read_text(encoding="utf-8")
    ranking_text = (ROOT / "app" / "ranking_service.py").read_text(encoding="utf-8")
    rebalance_text = (ROOT / "app" / "rebalance_service.py").read_text(encoding="utf-8")
    monitor_text = (ROOT / "app" / "portfolio_monitor_service.py").read_text(encoding="utf-8")
    for token in ["def safe_worker_count", "def train_one_asset", "def diagnose_one_asset"]:
        assert token in batch_text
    for token in ["def run_pybroker_replay", "def simulation_dir", "StrategyConfig", "YFinance", "mode: str = \"replay\"", "pybroker_walkforward_shadow"]:
        assert token in sim_text
    for token in ["def evaluate_baselines", "zero_return_no_trade", "buy_and_hold_equal_weight", "last_return_long_flat"]:
        assert token in evaluation_text
    assert "evaluate_baselines" in sim_text
    for token in ["def collect_refine_summary", "def run_feature_removal", "def render_removal_summary", "family_relevance_share_pct"]:
        assert token in refine_text
    for token in ["def collect_ranked_signals", "def render_ranking"]:
        assert token in ranking_text
    for token in ["def rebalance_portfolio", "def render_rebalance_summary"]:
        assert token in rebalance_text
    for token in ["def get_live_price", "def render_live_portfolio"]:
        assert token in monitor_text


def test_model5_ui_layer_exists_and_cli_uses_it_for_simulation():
    ui_text = (ROOT / "app" / "ui" / "model5.py").read_text(encoding="utf-8")
    cli_text = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    handlers_text = (ROOT / "app" / "cli_handlers.py").read_text(encoding="utf-8")
    validation_view_text = (ROOT / "app" / "validation_view.py").read_text(encoding="utf-8")
    for token in [
        "def render_header",
        "def render_section",
        "def render_key_values",
        "def render_table",
        "def render_operational_closing",
        "def render_badge",
        "def render_callout",
    ]:
        assert token in ui_text
    assert "from .ui import model5 as ui5" in validation_view_text
    assert "render_validation_summary" in handlers_text
    assert "screen_title" in cli_text
    assert "PYBROKER" in validation_view_text
    assert "--verbose" in cli_text
