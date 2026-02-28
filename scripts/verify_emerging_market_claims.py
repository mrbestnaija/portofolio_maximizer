from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    try:
        return needle in path.read_text(encoding="utf-8")
    except Exception:
        return False


def _has_xtb_execution_adapter(repo_root: Path) -> bool:
    """
    Return True only when a concrete XTB execution adapter exists.

    Broker config alone is not enough to claim live equity execution support.
    The repo must include an XTB-specific execution client/adapter in a runtime
    code path (not just secrets/config/tests).
    """
    candidate_patterns = (
        "execution/xtb*.py",
        "brokers/xtb*.py",
        "connectors/xtb*.py",
    )
    for pattern in candidate_patterns:
        for path in repo_root.glob(pattern):
            if path.is_file():
                return True
    return False


def _claim(status: str, summary: str, evidence: list[str]) -> Dict[str, Any]:
    return {
        "status": status,
        "summary": summary,
        "evidence": evidence,
    }


def audit_claims(root: Path | None = None) -> Dict[str, Any]:
    repo_root = Path(root) if root is not None else ROOT

    openbb_cfg = _load_yaml(repo_root / "config" / "openbb_config.yml").get("openbb", {})
    xtb_cfg = _load_yaml(repo_root / "config" / "xtb_config.yml").get("xtb", {})
    sentiment_cfg = _load_yaml(repo_root / "config" / "sentiment.yml")
    forecasting_cfg = _load_yaml(repo_root / "config" / "forecasting_config.yml").get("forecasting", {})

    openbb_extractor = repo_root / "etl" / "openbb_extractor.py"
    frontier_markets = repo_root / "etl" / "frontier_markets.py"
    data_universe = repo_root / "etl" / "data_universe.py"
    signal_validator = repo_root / "ai_llm" / "signal_validator.py"
    var_backtest = repo_root / "forcester_ts" / "var_backtest.py"
    walk_forward = repo_root / "forcester_ts" / "walk_forward_learner.py"
    horizon_backtest = repo_root / "scripts" / "run_horizon_consistent_backtest.py"
    lob_simulator = repo_root / "execution" / "lob_simulator.py"
    order_manager = repo_root / "execution" / "order_manager.py"

    nigeria_cfg = openbb_cfg.get("nigeria", {}) if isinstance(openbb_cfg, dict) else {}
    xtb_instruments = xtb_cfg.get("instruments", {}) if isinstance(xtb_cfg, dict) else {}
    xtb_forex = xtb_instruments.get("forex", {}) if isinstance(xtb_instruments, dict) else {}
    xtb_commodities = xtb_instruments.get("commodities", {}) if isinstance(xtb_instruments, dict) else {}
    xtb_stocks = xtb_instruments.get("stocks", {}) if isinstance(xtb_instruments, dict) else {}

    claims: Dict[str, Dict[str, Any]] = {}

    frontier_enabled = bool(nigeria_cfg.get("enabled")) and _file_contains(openbb_extractor, "Nigeria market support")
    frontier_listed = _file_contains(frontier_markets, "FRONTIER_MARKET_TICKERS_BY_REGION")
    frontier_filtering = _file_contains(data_universe, "resolve_ticker_universe")
    if frontier_enabled and frontier_listed and frontier_filtering:
        claims["frontier_market_data"] = _claim(
            "implemented",
            "Frontier/emerging market data routing is wired with Nigeria and curated frontier ticker support.",
            [
                "config/openbb_config.yml:nigeria",
                "etl/openbb_extractor.py: Nigeria market support",
                "etl/frontier_markets.py: FRONTIER_MARKET_TICKERS_BY_REGION",
                "etl/data_universe.py: resolve_ticker_universe",
            ],
        )
    else:
        claims["frontier_market_data"] = _claim(
            "unsupported",
            "Frontier/emerging market data routing is not fully wired.",
            [],
        )

    commodity_fx_enabled = bool(xtb_forex.get("enabled")) and bool(xtb_commodities.get("enabled"))
    if commodity_fx_enabled:
        claims["commodity_fx_execution"] = _claim(
            "implemented",
            "Execution config supports both commodity and FX instruments.",
            [
                "config/xtb_config.yml: instruments.forex.enabled",
                "config/xtb_config.yml: instruments.commodities.enabled",
            ],
        )
    else:
        claims["commodity_fx_execution"] = _claim(
            "unsupported",
            "Execution config does not enable both commodity and FX instruments.",
            [],
        )

    stock_enabled = bool(xtb_stocks.get("enabled"))
    xtb_stock_adapter = _has_xtb_execution_adapter(repo_root)
    if stock_enabled and xtb_stock_adapter:
        stock_status = "implemented"
        stock_summary = "Stock execution is enabled in broker config and an XTB execution adapter is present."
    elif stock_enabled:
        stock_status = "partial"
        stock_summary = (
            "Broker config enables stocks and frontier equity tickers exist, "
            "but no XTB execution adapter was found in runtime code."
        )
    else:
        stock_status = "partial"
        stock_summary = "Stock execution hooks exist, but broker-side stocks are disabled by default."
    claims["emerging_market_equity_execution"] = _claim(
        stock_status,
        stock_summary,
        [
            "config/xtb_config.yml: instruments.stocks",
            "etl/frontier_markets.py: curated frontier equity tickers",
            *(
                ["execution/: xtb adapter present"]
                if xtb_stock_adapter
                else []
            ),
        ],
    )

    sentiment_enabled = bool(sentiment_cfg.get("enabled"))
    sentiment_providers = sentiment_cfg.get("sources", {}) if isinstance(sentiment_cfg, dict) else {}
    sentiment_has_providers = bool(sentiment_providers)
    if sentiment_has_providers and not sentiment_enabled:
        claims["sentiment_news_monitoring"] = _claim(
            "dormant",
            "News/sentiment scaffolding exists but is gated off by default.",
            [
                "config/sentiment.yml: enabled=false",
                "config/sentiment.yml: sources.news.providers",
            ],
        )
    elif sentiment_has_providers:
        claims["sentiment_news_monitoring"] = _claim(
            "implemented",
            "News/sentiment monitoring is configured and enabled.",
            [
                "config/sentiment.yml",
            ],
        )
    else:
        claims["sentiment_news_monitoring"] = _claim(
            "unsupported",
            "No sentiment/news monitoring scaffold was found.",
            [],
        )

    weather_context = repo_root / "utils" / "weather_context.py"
    signal_router = repo_root / "models" / "signal_router.py"
    ts_signal_generator = repo_root / "models" / "time_series_signal_generator.py"
    paper_engine = repo_root / "execution" / "paper_trading_engine.py"
    weather_overlay = all(
        (
            _file_contains(signal_validator, "def _apply_weather_risk_overlay"),
            _file_contains(signal_validator, "hydrate_signal_weather_context("),
            _file_contains(weather_context, "def hydrate_signal_weather_context"),
            _file_contains(ts_signal_generator, "extract_weather_context("),
            _file_contains(signal_router, "'weather_context': weather_context"),
            _file_contains(paper_engine, 'decision_context.get("expected_return_net")'),
        )
    )
    claims["weather_risk_overlay"] = _claim(
        "implemented" if weather_overlay else "unsupported",
        (
            "Structured weather context is wired through validation, TS provenance, routing, and the net-edge execution gate."
            if weather_overlay
            else "Weather-risk overlay and its downstream execution wiring are incomplete."
        ),
        [
            "ai_llm/signal_validator.py: _apply_weather_risk_overlay + hydrate_signal_weather_context",
            "utils/weather_context.py: hydrate_signal_weather_context",
            "models/time_series_signal_generator.py: extract_weather_context",
            "models/signal_router.py: root weather_context promotion",
            "execution/paper_trading_engine.py: decision_context.expected_return_net fallback",
        ] if weather_overlay else [],
    )

    backtesting_enabled = (
        _file_contains(var_backtest, "class VaRBacktester")
        and _file_contains(walk_forward, "class WalkForwardLearner")
        and _file_contains(horizon_backtest, "def run_horizon_backtest")
        and bool(forecasting_cfg.get("var_backtest", {}).get("enabled"))
    )
    claims["backtesting_and_optimization"] = _claim(
        "implemented" if backtesting_enabled else "unsupported",
        (
            "VaR, walk-forward, and horizon backtesting paths are present."
            if backtesting_enabled
            else "Backtesting coverage is incomplete."
        ),
        [
            "forcester_ts/var_backtest.py",
            "forcester_ts/walk_forward_learner.py",
            "scripts/run_horizon_consistent_backtest.py",
            "config/forecasting_config.yml: var_backtest",
        ] if backtesting_enabled else [],
    )

    liquidity_controls = _file_contains(lob_simulator, "def simulate_market_order_fill") and _file_contains(
        order_manager, "mid_slippage_bps"
    )
    claims["liquidity_and_slippage_controls"] = _claim(
        "implemented" if liquidity_controls else "unsupported",
        (
            "Order-book slippage simulation and execution slippage persistence are present."
            if liquidity_controls
            else "Liquidity/slippage controls are incomplete."
        ),
        [
            "execution/lob_simulator.py: simulate_market_order_fill",
            "execution/order_manager.py: mid_slippage_bps",
        ] if liquidity_controls else [],
    )

    for claim_key, needle, summary in (
        (
            "geopolitical_policy_monitoring",
            "geopolitical",
            "No explicit geopolitical policy monitor was found in code/config.",
        ),
        (
            "cross_border_arbitrage",
            "arbitrage",
            "No cross-border arbitrage implementation was found in code/config.",
        ),
        (
            "esg_screening",
            "ESG",
            "No ESG screening implementation was found in code/config.",
        ),
    ):
        found = any(
            _file_contains(path, needle)
            for path in (signal_validator, repo_root / "scripts" / "run_auto_trader.py", repo_root / "config" / "sentiment.yml")
        )
        claims[claim_key] = _claim(
            "implemented" if found else "unsupported",
            f"Detected '{needle}' support." if found else summary,
            [f"keyword:{needle}"] if found else [],
        )

    counts = Counter(entry["status"] for entry in claims.values())
    return {
        "as_of": date.today().isoformat(),
        "claims": claims,
        "summary": {
            "implemented": counts.get("implemented", 0),
            "partial": counts.get("partial", 0),
            "dormant": counts.get("dormant", 0),
            "unsupported": counts.get("unsupported", 0),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local support for emerging-market product claims.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    payload = audit_claims()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Emerging-market claim audit ({payload['as_of']})")
        for claim, entry in sorted(payload["claims"].items()):
            print(f"- {claim}: {entry['status']} :: {entry['summary']}")
        summary = payload["summary"]
        print(
            "Summary: "
            f"implemented={summary['implemented']} "
            f"partial={summary['partial']} "
            f"dormant={summary['dormant']} "
            f"unsupported={summary['unsupported']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
