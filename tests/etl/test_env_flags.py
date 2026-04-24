"""Unit tests for etl/env_flags.py.

Verifies that parse_env_bool and is_synthetic_mode never treat "0", "false",
"no", or an unset variable as truthy — the exact silent failure mode that was
causing all live trades to be classified as synthetic.
"""
import os
from unittest.mock import patch

import pytest

from etl.env_flags import is_synthetic_mode, parse_env_bool


# ---------------------------------------------------------------------------
# parse_env_bool
# ---------------------------------------------------------------------------

class TestParseEnvBool:
    """parse_env_bool(name) must return a proper Python bool, not a truthy string."""

    def _set(self, name: str, value: str):
        """Context manager that sets env var for the duration of a test."""
        return patch.dict(os.environ, {name: value})

    def _unset(self, name: str):
        """Context manager that removes env var for the duration of a test."""
        env = {k: v for k, v in os.environ.items() if k != name}
        return patch.dict(os.environ, env, clear=True)

    # --- falsy values -------------------------------------------------------

    def test_zero_string_is_false(self):
        with self._set("TEST_FLAG", "0"):
            assert parse_env_bool("TEST_FLAG") is False

    def test_false_string_is_false(self):
        with self._set("TEST_FLAG", "false"):
            assert parse_env_bool("TEST_FLAG") is False

    def test_False_titlecase_is_false(self):
        with self._set("TEST_FLAG", "False"):
            assert parse_env_bool("TEST_FLAG") is False

    def test_no_string_is_false(self):
        with self._set("TEST_FLAG", "no"):
            assert parse_env_bool("TEST_FLAG") is False

    def test_off_string_is_false(self):
        with self._set("TEST_FLAG", "off"):
            assert parse_env_bool("TEST_FLAG") is False

    def test_empty_string_is_false(self):
        with self._set("TEST_FLAG", ""):
            assert parse_env_bool("TEST_FLAG") is False

    def test_unset_returns_default_false(self):
        with self._unset("TEST_FLAG"):
            assert parse_env_bool("TEST_FLAG") is False

    def test_unset_returns_custom_default_true(self):
        with self._unset("TEST_FLAG"):
            assert parse_env_bool("TEST_FLAG", default=True) is True

    # --- truthy values ------------------------------------------------------

    def test_one_string_is_true(self):
        with self._set("TEST_FLAG", "1"):
            assert parse_env_bool("TEST_FLAG") is True

    def test_true_string_is_true(self):
        with self._set("TEST_FLAG", "true"):
            assert parse_env_bool("TEST_FLAG") is True

    def test_True_titlecase_is_true(self):
        with self._set("TEST_FLAG", "True"):
            assert parse_env_bool("TEST_FLAG") is True

    def test_yes_string_is_true(self):
        with self._set("TEST_FLAG", "yes"):
            assert parse_env_bool("TEST_FLAG") is True

    def test_on_string_is_true(self):
        with self._set("TEST_FLAG", "on"):
            assert parse_env_bool("TEST_FLAG") is True

    def test_whitespace_padded_true(self):
        with self._set("TEST_FLAG", "  1  "):
            assert parse_env_bool("TEST_FLAG") is True

    def test_whitespace_padded_false(self):
        with self._set("TEST_FLAG", "  0  "):
            assert parse_env_bool("TEST_FLAG") is False

    # --- return type --------------------------------------------------------

    def test_returns_bool_not_string(self):
        with self._set("TEST_FLAG", "1"):
            result = parse_env_bool("TEST_FLAG")
            assert type(result) is bool  # noqa: E721 — must be bool, not truthy

    def test_false_return_is_bool_not_string(self):
        with self._set("TEST_FLAG", "0"):
            result = parse_env_bool("TEST_FLAG")
            assert type(result) is bool


# ---------------------------------------------------------------------------
# is_synthetic_mode
# ---------------------------------------------------------------------------

class TestIsSyntheticMode:
    """is_synthetic_mode() must return False in all live-run configurations
    and True only when explicitly enabled."""

    def _env(self, **kwargs):
        """Patch os.environ with exactly the given keys; clear everything else."""
        return patch.dict(os.environ, kwargs, clear=True)

    # --- must be False in live runs -----------------------------------------

    def test_no_flags_set_is_real(self):
        with self._env():
            assert is_synthetic_mode() is False

    def test_synthetic_only_zero_is_real(self):
        """SYNTHETIC_ONLY=0 must NOT force synthetic mode — the original bug."""
        with self._env(SYNTHETIC_ONLY="0"):
            assert is_synthetic_mode() is False

    def test_synthetic_only_false_string_is_real(self):
        with self._env(SYNTHETIC_ONLY="false"):
            assert is_synthetic_mode() is False

    def test_synthetic_only_no_string_is_real(self):
        with self._env(SYNTHETIC_ONLY="no"):
            assert is_synthetic_mode() is False

    def test_execution_mode_live_is_real(self):
        with self._env(EXECUTION_MODE="live"):
            assert is_synthetic_mode(execution_mode="live") is False

    def test_execution_mode_auto_is_real(self):
        with self._env():
            assert is_synthetic_mode(execution_mode="auto") is False

    # --- must be True when explicitly enabled --------------------------------

    def test_synthetic_only_one_is_synthetic(self):
        with self._env(SYNTHETIC_ONLY="1"):
            assert is_synthetic_mode() is True

    def test_synthetic_only_true_string_is_synthetic(self):
        with self._env(SYNTHETIC_ONLY="true"):
            assert is_synthetic_mode() is True

    def test_execution_mode_arg_synthetic(self):
        with self._env():
            assert is_synthetic_mode(execution_mode="synthetic") is True

    def test_execution_mode_env_synthetic(self):
        with self._env(EXECUTION_MODE="synthetic"):
            assert is_synthetic_mode() is True

    def test_pmx_execution_mode_env_synthetic(self):
        with self._env(PMX_EXECUTION_MODE="synthetic"):
            assert is_synthetic_mode() is True

    def test_data_source_env_synthetic(self):
        with self._env(DATA_SOURCE="synthetic"):
            assert is_synthetic_mode() is True

    def test_preferred_source_env_synthetic(self):
        with self._env(PMX_PREFERRED_DATA_SOURCE="synthetic"):
            assert is_synthetic_mode() is True

    def test_data_source_argument_synthetic_is_real_signal(self):
        with self._env():
            assert is_synthetic_mode(data_source="synthetic") is True

    # --- priority: SYNTHETIC_ONLY wins over execution_mode arg --------------

    def test_synthetic_only_overrides_execution_mode_live(self):
        """SYNTHETIC_ONLY=1 beats execution_mode=live."""
        with self._env(SYNTHETIC_ONLY="1"):
            assert is_synthetic_mode(execution_mode="live") is True

    def test_synthetic_only_zero_does_not_block_execution_mode_synthetic(self):
        """SYNTHETIC_ONLY=0 + execution_mode=synthetic → synthetic (execution_mode wins)."""
        with self._env(SYNTHETIC_ONLY="0"):
            assert is_synthetic_mode(execution_mode="synthetic") is True
