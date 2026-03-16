from __future__ import annotations

from etl.data_universe import ResolvedUniverse
from scripts import run_etl_pipeline as mod


def test_prepare_ticker_universe_uses_shared_resolver_for_discovery(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_resolver(**kwargs):
        calls.append(kwargs)
        return ResolvedUniverse(
            tickers=["AAPL", "MSFT"],
            active_source="alpha_vantage",
            universe_source="alpha_vantage_universe",
            notes={"info": "shared"},
        )

    monkeypatch.setattr(mod, "resolve_ticker_universe", fake_resolver)

    universe = mod._prepare_ticker_universe(
        tickers_argument="AAPL,MSFT",
        discovery_cfg={"enabled": True, "loader": "alpha_vantage"},
        active_source="alpha_vantage",
        include_frontier_tickers=True,
        use_ticker_discovery=True,
        refresh_ticker_universe=True,
    )

    assert universe.tickers == ["AAPL", "MSFT"]
    assert universe.universe_source == "alpha_vantage_universe"
    assert len(calls) == 1
    assert calls[0]["base_tickers"] == []
    assert calls[0]["include_frontier"] is True
    assert calls[0]["active_source"] == "alpha_vantage"
    assert calls[0]["use_discovery"] is True
    assert calls[0]["discovery_config"] == {"enabled": True, "loader": "alpha_vantage"}
    assert calls[0]["refresh_discovery"] is True


def test_prepare_ticker_universe_falls_back_to_manual_tickers_via_shared_resolver(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    responses = [
        ResolvedUniverse(
            tickers=[],
            active_source="alpha_vantage",
            universe_source="none",
            notes={},
        ),
        ResolvedUniverse(
            tickers=["AAPL"],
            active_source="alpha_vantage",
            universe_source="explicit+frontier",
            notes={},
        ),
    ]

    def fake_resolver(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(mod, "resolve_ticker_universe", fake_resolver)

    universe = mod._prepare_ticker_universe(
        tickers_argument="AAPL",
        discovery_cfg={"enabled": True, "loader": "alpha_vantage"},
        active_source="alpha_vantage",
        include_frontier_tickers=False,
        use_ticker_discovery=True,
        refresh_ticker_universe=False,
    )

    assert universe.tickers == ["AAPL"]
    assert len(calls) == 2
    assert calls[0]["use_discovery"] is True
    assert calls[0]["base_tickers"] == []
    assert calls[1]["use_discovery"] is False
    assert calls[1]["base_tickers"] == ["AAPL"]
