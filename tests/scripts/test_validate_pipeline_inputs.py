"""Tests for scripts/validate_pipeline_inputs.py (V1-V6 pre-flight checks)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 200, start: str = "2020-01-01",
                   constant: bool = False) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a datetime index."""
    dates = pd.date_range(start, periods=n, freq="B", tz="UTC")
    if constant:
        close = np.full(n, 100.0)
    else:
        rng = np.random.default_rng(42)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Close": close, "Volume": 1_000_000.0}, index=dates)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)


def _write_jsonl(entries: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e) for e in entries),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# V1: Filename Convention
# ---------------------------------------------------------------------------

class TestV1FilenameConvention:
    def test_pass_when_ticker_in_filename(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v1_filename_convention
        df = _make_price_df()
        (tmp_path / "AAPL_data_extraction_test.parquet").write_bytes(
            _make_price_df().to_parquet(None) or b""
        )
        _write_parquet(df, tmp_path / "AAPL_data_extraction_test.parquet")
        results = check_v1_filename_convention(["AAPL"], tmp_path)
        assert len(results) == 1
        assert results[0].status == "PASS"
        assert "AAPL" in results[0].message

    def test_warn_when_unnamed_fallback_exists(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v1_filename_convention
        _write_parquet(_make_price_df(), tmp_path / "pipeline_20260101_data_extraction_test.parquet")
        results = check_v1_filename_convention(["MSFT"], tmp_path)
        assert len(results) == 1
        assert results[0].status == "WARN"

    def test_fail_when_no_parquet_at_all(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v1_filename_convention
        results = check_v1_filename_convention(["NVDA"], tmp_path)
        assert len(results) == 1
        assert results[0].status == "FAIL"

    def test_multiple_tickers_independent(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v1_filename_convention
        _write_parquet(_make_price_df(), tmp_path / "AAPL_data_extraction.parquet")
        results = check_v1_filename_convention(["AAPL", "MSFT"], tmp_path)
        statuses = {r.check_id.split(".")[1]: r.status for r in results}
        assert statuses["AAPL"] == "PASS"
        assert statuses["MSFT"] == "WARN"  # unnamed fallback (AAPL file exists)


# ---------------------------------------------------------------------------
# V2: Parquet Coverage Map
# ---------------------------------------------------------------------------

class TestV2ParquetCoverage:
    def test_pass_for_good_parquet(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v2_parquet_coverage
        _write_parquet(_make_price_df(n=200), tmp_path / "AAPL_data_extraction.parquet")
        results = check_v2_parquet_coverage(tmp_path)
        assert any(r.status == "PASS" for r in results)

    def test_fail_no_parquets(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v2_parquet_coverage
        results = check_v2_parquet_coverage(tmp_path)
        assert len(results) == 1
        assert results[0].status == "FAIL"

    def test_fail_no_close_column(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v2_parquet_coverage
        df = pd.DataFrame({"Volume": range(200)})
        df.to_parquet(tmp_path / "nclose_data_extraction.parquet")
        results = check_v2_parquet_coverage(tmp_path)
        assert any(r.status == "FAIL" for r in results)

    def test_fail_too_short(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v2_parquet_coverage
        _write_parquet(_make_price_df(n=10), tmp_path / "short_data_extraction.parquet")
        results = check_v2_parquet_coverage(tmp_path)
        assert any(r.status == "FAIL" for r in results)

    def test_fail_constant_prices_synthetic(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v2_parquet_coverage
        _write_parquet(_make_price_df(n=200, constant=True),
                       tmp_path / "synth_data_extraction.parquet")
        results = check_v2_parquet_coverage(tmp_path)
        assert any(r.status == "FAIL" for r in results)
        msg = " ".join(r.message for r in results)
        assert "constant" in msg.lower() or "synthetic" in msg.lower() or "degenerate" in msg.lower()


# ---------------------------------------------------------------------------
# V3: JSONL Timestamp Alignment
# ---------------------------------------------------------------------------

class TestV3JsonlAlignment:
    def test_skip_when_jsonl_missing(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v3_jsonl_alignment
        result = check_v3_jsonl_alignment(tmp_path / "missing.jsonl", tmp_path)
        assert result.status == "SKIP"

    def test_fail_when_no_readable_parquet_coverage(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v3_jsonl_alignment
        jsonl = tmp_path / "quant.jsonl"
        _write_jsonl([
            {"timestamp": "2020-06-01T12:00:00Z",
             "classifier_features": {"realized_vol": 0.2}},
        ], jsonl)
        result = check_v3_jsonl_alignment(jsonl, tmp_path)
        assert result.status == "FAIL"
        assert "No readable parquet coverage" in result.message

    def test_warn_when_zero_alignable(self, tmp_path):
        # 0% alignable = WARN (not FAIL): generate_classifier_training_labels.py
        # is unaffected by JSONL timestamps; blocking would be a false positive.
        from scripts.validate_pipeline_inputs import check_v3_jsonl_alignment
        _write_parquet(_make_price_df(n=200, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        jsonl = tmp_path / "quant.jsonl"
        _write_jsonl([
            {"timestamp": "2026-03-18T12:00:00Z",
             "classifier_features": {"realized_vol": 0.2}},
        ], jsonl)
        result = check_v3_jsonl_alignment(jsonl, tmp_path)
        assert result.status == "WARN"
        assert "0 of" in result.message

    def test_pass_when_timestamps_align(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v3_jsonl_alignment
        # Parquet covers 2020; JSONL entries are within that range
        _write_parquet(_make_price_df(n=200, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        jsonl = tmp_path / "quant.jsonl"
        entries = [
            {"timestamp": f"2020-{month:02d}-01T12:00:00Z",
             "classifier_features": {"realized_vol": 0.2}}
            for month in range(2, 8)
        ]
        _write_jsonl(entries, jsonl)
        result = check_v3_jsonl_alignment(jsonl, tmp_path)
        assert result.status == "PASS"

    def test_alignment_is_ticker_scoped(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v3_jsonl_alignment
        _write_parquet(_make_price_df(n=200, start="2020-01-01"),
                       tmp_path / "MSFT_data_extraction.parquet")
        jsonl = tmp_path / "quant.jsonl"
        _write_jsonl([
            {"ticker": "AAPL",
             "timestamp": "2020-03-02T12:00:00Z",
             "classifier_features": {"realized_vol": 0.2}},
        ], jsonl)
        result = check_v3_jsonl_alignment(jsonl, tmp_path, tickers=["AAPL", "MSFT"])
        assert result.status == "WARN"
        assert result.details["n_alignable"] == 0
        assert result.details["n_missing_ticker_coverage"] == 1

    def test_signal_id_can_supply_ticker_scope(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v3_jsonl_alignment
        _write_parquet(_make_price_df(n=200, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        jsonl = tmp_path / "quant.jsonl"
        _write_jsonl([
            {"signal_id": "ts_AAPL_20200302_0001",
             "timestamp": "2020-03-02T12:00:00Z",
             "classifier_features": {"realized_vol": 0.2}},
        ], jsonl)
        result = check_v3_jsonl_alignment(jsonl, tmp_path, tickers=["AAPL"])
        assert result.status == "PASS"
        assert result.details["n_alignable"] == 1


# ---------------------------------------------------------------------------
# V4: Eval Date Coverage
# ---------------------------------------------------------------------------

class TestV4EvalDateCoverage:
    def test_pass_when_date_within_parquet(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v4_eval_date_coverage
        _write_parquet(_make_price_df(n=400, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        results = check_v4_eval_date_coverage(
            ["2021-06-01"], ["AAPL"], tmp_path
        )
        assert results[0].status == "PASS"

    def test_fail_when_date_outside_parquet(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v4_eval_date_coverage
        # Parquet covers 2020 only; eval date is 2025
        _write_parquet(_make_price_df(n=200, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        results = check_v4_eval_date_coverage(
            ["2025-01-06"], ["AAPL"], tmp_path
        )
        assert results[0].status == "FAIL"
        assert "outside" in results[0].message.lower()

    def test_warn_partial_ticker_coverage(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v4_eval_date_coverage
        # AAPL parquet covers 2020; no MSFT parquet → partial coverage
        _write_parquet(_make_price_df(n=300, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        results = check_v4_eval_date_coverage(
            ["2020-06-01"], ["AAPL", "MSFT"], tmp_path
        )
        assert results[0].status == "WARN"
        assert "MSFT" in results[0].message


# ---------------------------------------------------------------------------
# V5: Duplicate-Parquet Multi-Ticker
# ---------------------------------------------------------------------------

class TestV5DuplicateParquet:
    def test_pass_when_distinct_parquets(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v5_duplicate_parquet
        # Two tickers with distinct parquets at different price levels
        rng = np.random.default_rng(0)
        df_aapl = pd.DataFrame({
            "Close": 150.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        }, index=pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC"))
        df_msft = pd.DataFrame({
            "Close": 300.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        }, index=pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC"))
        _write_parquet(df_aapl, tmp_path / "AAPL_data_extraction.parquet")
        _write_parquet(df_msft, tmp_path / "MSFT_data_extraction.parquet")
        result = check_v5_duplicate_parquet(["AAPL", "MSFT"], tmp_path)
        assert result.status == "PASS"

    def test_fail_on_identical_close0_synthetic_collision(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v5_duplicate_parquet
        # Single parquet, both tickers resolve to it → identical Close[0] = FAIL
        df = _make_price_df(n=200, start="2020-01-01")  # Close[0] ≈ 100
        pq = tmp_path / "pipeline_data_extraction.parquet"
        _write_parquet(df, pq)
        # No ticker-named parquets → both tickers fall back to same unnamed parquet
        result = check_v5_duplicate_parquet(["AAPL", "MSFT"], tmp_path)
        assert result.status == "FAIL"
        assert "synthetic" in result.message.lower() or "identical" in result.message.lower()


# ---------------------------------------------------------------------------
# V6: Edge Cases
# ---------------------------------------------------------------------------

class TestV6EdgeCases:
    def test_pass_when_no_anomalies(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v6_edge_cases
        _write_parquet(_make_price_df(n=200), tmp_path / "AAPL_data_extraction.parquet")
        jsonl = tmp_path / "quant.jsonl"
        _write_jsonl([{"timestamp": "2020-06-01T00:00:00Z", "action": "BUY"}], jsonl)
        results = check_v6_edge_cases(tmp_path, jsonl, tmp_path / "missing.parquet")
        assert all(r.status in ("PASS", "SKIP") for r in results)

    def test_fail_on_empty_parquet(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v6_edge_cases
        df = pd.DataFrame({"Close": []})
        df.to_parquet(tmp_path / "empty_data_extraction.parquet")
        results = check_v6_edge_cases(tmp_path, tmp_path / "missing.jsonl", tmp_path / "missing.parquet")
        assert any(r.status == "FAIL" for r in results)

    def test_warn_on_stale_training_dataset(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v6_edge_cases
        import time
        training = tmp_path / "directional_dataset.parquet"
        _write_parquet(_make_price_df(n=10), training)
        # Force mtime to 30 days ago
        old_time = time.time() - 30 * 86400
        import os
        os.utime(training, (old_time, old_time))
        results = check_v6_edge_cases(tmp_path, tmp_path / "missing.jsonl", training, stale_days=7)
        assert any(r.status == "WARN" and "stale" in r.check_id for r in results)

    def test_fail_when_checkpoint_dir_missing(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v6_edge_cases
        missing_dir = tmp_path / "no_such_dir"
        results = check_v6_edge_cases(missing_dir, tmp_path / "missing.jsonl", tmp_path / "missing.parquet")
        assert any(r.status == "FAIL" and "missing_checkpoint_dir" in r.check_id for r in results)

    def test_warn_on_malformed_jsonl_timestamp(self, tmp_path):
        from scripts.validate_pipeline_inputs import check_v6_edge_cases
        _write_parquet(_make_price_df(n=200), tmp_path / "AAPL_data_extraction.parquet")
        jsonl = tmp_path / "quant.jsonl"
        _write_jsonl([
            {"timestamp": "not-a-timestamp",
             "classifier_features": {"realized_vol": 0.2}},
        ], jsonl)
        results = check_v6_edge_cases(tmp_path, jsonl, tmp_path / "missing.parquet")
        malformed = [r for r in results if r.check_id == "V6.malformed_timestamps"]
        assert malformed
        assert malformed[0].status == "WARN"
        assert malformed[0].details["malformed_count"] == 1

    def test_warn_on_malformed_timestamps(self, tmp_path):
        """V6 must detect malformed timestamp strings, not just None/missing."""
        from scripts.validate_pipeline_inputs import check_v6_edge_cases
        import json
        jsonl = tmp_path / "quant.jsonl"
        entries = [
            {"signal_timestamp": "not-a-timestamp", "classifier_features": {"a": 1}},
            {"signal_timestamp": "2020-06-01T00:00:00Z", "classifier_features": {"a": 1}},
        ]
        jsonl.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        _write_parquet(_make_price_df(n=200), tmp_path / "AAPL_data_extraction.parquet")
        results = check_v6_edge_cases(tmp_path, jsonl, tmp_path / "missing.parquet")
        warn = next((r for r in results if "null_timestamp" in r.check_id), None)
        assert warn is not None, "Expected V6.null_timestamps warning for malformed timestamp"
        assert warn.status == "WARN"
        assert warn.details.get("malformed_count", 0) == 1

    def test_warn_counts_both_null_and_malformed(self, tmp_path):
        """V6 null_timestamps warning counts null + malformed separately in details."""
        from scripts.validate_pipeline_inputs import check_v6_edge_cases
        import json
        jsonl = tmp_path / "quant.jsonl"
        entries = [
            {"signal_timestamp": None, "classifier_features": {"a": 1}},
            {"signal_timestamp": "bad-ts", "classifier_features": {"a": 1}},
            {"signal_timestamp": "2020-06-01T00:00:00Z", "classifier_features": {"a": 1}},
        ]
        jsonl.write_text("\n".join(json.dumps(e) for e in entries), encoding="utf-8")
        _write_parquet(_make_price_df(n=200), tmp_path / "AAPL_data_extraction.parquet")
        results = check_v6_edge_cases(tmp_path, jsonl, tmp_path / "missing.parquet")
        warn = next((r for r in results if "null_timestamp" in r.check_id), None)
        assert warn is not None
        assert warn.details.get("null_count", 0) == 1
        assert warn.details.get("malformed_count", 0) == 1


# ---------------------------------------------------------------------------
# run_all_checks integration
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_exit_code_0_clean_setup(self, tmp_path):
        from scripts.validate_pipeline_inputs import run_all_checks
        df = _make_price_df(n=300, start="2020-01-01")
        _write_parquet(df, tmp_path / "AAPL_data_extraction.parquet")
        _results, code = run_all_checks(
            tickers=["AAPL"],
            eval_dates=["2020-06-01"],
            checkpoint_dir=tmp_path,
            jsonl_path=tmp_path / "missing.jsonl",
            training_path=tmp_path / "missing.parquet",
        )
        assert code == 0

    def test_exit_code_1_when_fail_present(self, tmp_path):
        from scripts.validate_pipeline_inputs import run_all_checks
        # No parquets at all → V1 FAIL + V2 FAIL
        _results, code = run_all_checks(
            tickers=["AAPL"],
            eval_dates=["2020-06-01"],
            checkpoint_dir=tmp_path,
            jsonl_path=tmp_path / "missing.jsonl",
            training_path=tmp_path / "missing.parquet",
        )
        assert code == 1

    def test_json_output_is_valid(self, tmp_path):
        from scripts.validate_pipeline_inputs import main
        _write_parquet(_make_price_df(n=300, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        main([
            "--tickers", "AAPL",
            "--eval-dates", "2020-06-01",
            "--checkpoint-dir", str(tmp_path),
            "--jsonl-path", str(tmp_path / "missing.jsonl"),
            "--training-path", str(tmp_path / "missing.parquet"),
            "--json",
        ])
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        payload = json.loads(output)
        assert "checks" in payload
        assert "exit_code" in payload
        assert isinstance(payload["checks"], list)

    def test_cli_returns_0_for_clean_setup(self, tmp_path):
        from scripts.validate_pipeline_inputs import main
        _write_parquet(_make_price_df(n=300, start="2020-01-01"),
                       tmp_path / "AAPL_data_extraction.parquet")
        rc = main([
            "--tickers", "AAPL",
            "--eval-dates", "2020-06-01",
            "--checkpoint-dir", str(tmp_path),
            "--jsonl-path", str(tmp_path / "missing.jsonl"),
            "--training-path", str(tmp_path / "missing.parquet"),
        ])
        assert rc == 0

    def test_cli_returns_1_when_fail(self, tmp_path):
        from scripts.validate_pipeline_inputs import main
        # No parquets → FAIL
        rc = main([
            "--tickers", "AAPL",
            "--eval-dates", "2025-01-01",
            "--checkpoint-dir", str(tmp_path),
            "--jsonl-path", str(tmp_path / "missing.jsonl"),
            "--training-path", str(tmp_path / "missing.parquet"),
        ])
        assert rc == 1
