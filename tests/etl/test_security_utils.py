"""
Security utilities tests.

Tests for error sanitization and log message sanitization to prevent
information leakage in production environments.
"""

import os
import pytest
import logging
from unittest.mock import patch, MagicMock

from etl.security_utils import sanitize_error, sanitize_log_message


class TestErrorSanitization:
    """Test error message sanitization."""

    def test_sanitize_error_production_mode_generic(self):
        """In production, errors should return generic message."""
        exc = ValueError("Database connection failed: user=admin, password=secret123")
        result = sanitize_error(exc, is_production=True)

        assert result == "An error occurred. Please contact support."
        assert "Database connection failed" not in result
        assert "password=secret123" not in result

    def test_sanitize_error_development_mode_detailed(self):
        """In development, errors should return detailed message."""
        exc = ValueError("Database connection failed: timeout after 30s")
        result = sanitize_error(exc, is_production=False)

        assert "Database connection failed" in result
        assert "timeout after 30s" in result

    def test_sanitize_error_auto_detects_production(self):
        """Should auto-detect production mode from PORTFOLIO_ENV."""
        exc = ValueError("Test error")

        with patch.dict(os.environ, {'PORTFOLIO_ENV': 'production'}):
            result = sanitize_error(exc)
            assert result == "An error occurred. Please contact support."

        with patch.dict(os.environ, {'PORTFOLIO_ENV': 'development'}):
            result = sanitize_error(exc)
            assert "Test error" in result

    def test_sanitize_error_logs_internally(self):
        """Production mode should log detailed error internally."""
        exc = ValueError("Sensitive error: api_key=abc123")

        with patch('etl.security_utils.logger') as mock_logger:
            result = sanitize_error(exc, is_production=True)

            # Should log detailed error
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Sensitive error" in str(call_args)
            # Verify exc_info=True was passed
            assert call_args[1].get('exc_info') is True

            # But return generic message
            assert result == "An error occurred. Please contact support."

    def test_sanitize_error_various_exception_types(self):
        """Should handle various exception types."""
        exceptions = [
            ValueError("Value error"),
            KeyError("Missing key"),
            ConnectionError("Connection failed"),
            RuntimeError("Runtime error"),
        ]

        for exc in exceptions:
            result = sanitize_error(exc, is_production=True)
            assert result == "An error occurred. Please contact support."

            result = sanitize_error(exc, is_production=False)
            assert str(exc) in result or type(exc).__name__ in result


class TestLogMessageSanitization:
    """Test log message sanitization."""

    def test_sanitize_log_message_redacts_api_keys(self):
        """Should redact API keys from log messages."""
        message = "API call failed: api_key=ALPHA_VANTAGE_KEY_1234567890"
        result = sanitize_log_message(message)

        assert "api_key=***REDACTED***" in result
        assert "ALPHA_VANTAGE_KEY_1234567890" not in result

    def test_sanitize_log_message_redacts_passwords(self):
        """Should redact passwords from log messages."""
        message = "Login failed: password=MySecretPassword123"
        result = sanitize_log_message(message)

        assert "password=***REDACTED***" in result
        assert "MySecretPassword123" not in result

    def test_sanitize_log_message_redacts_tokens(self):
        """Should redact tokens from log messages."""
        message = "Auth failed: token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = sanitize_log_message(message)

        assert "token=***REDACTED***" in result
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result

    def test_sanitize_log_message_redacts_secrets(self):
        """Should redact secrets from log messages."""
        message = "Config error: secret=my_secret_value_12345"
        result = sanitize_log_message(message)

        assert "secret=***REDACTED***" in result
        assert "my_secret_value_12345" not in result

    def test_sanitize_log_message_case_insensitive(self):
        """Should handle case-insensitive matching."""
        messages = [
            "API_KEY=test123",
            "Api_Key=test123",
            "API_key=test123",
        ]

        for message in messages:
            result = sanitize_log_message(message)
            assert "***REDACTED***" in result

    def test_sanitize_log_message_multiple_sensitive_data(self):
        """Should redact multiple sensitive values in one message."""
        message = "Error: api_key=test123 password=secret456 token=token789"
        result = sanitize_log_message(message)

        assert "api_key=***REDACTED***" in result
        assert "password=***REDACTED***" in result
        assert "token=***REDACTED***" in result
        assert "test123" not in result
        assert "secret456" not in result
        assert "token789" not in result

    def test_sanitize_log_message_safe_content_preserved(self):
        """Should preserve safe content in log messages."""
        message = "User logged in successfully at 2025-01-27 10:30:00"
        result = sanitize_log_message(message)

        assert result == message  # No changes needed

    def test_sanitize_log_message_custom_patterns(self):
        """Should support custom sensitive patterns."""
        message = "Custom secret: my_secret_field=secret_value"
        custom_patterns = [
            (r'my_secret_field["\s:=]+([^\s]+)', r'my_secret_field=***REDACTED***'),
        ]

        result = sanitize_log_message(message, sensitive_patterns=custom_patterns)

        assert "my_secret_field=***REDACTED***" in result
        assert "secret_value" not in result


@pytest.mark.security
class TestSecurityIntegration:
    """Integration tests for security utilities."""

    def test_production_error_handling_flow(self):
        """Test complete error handling flow in production mode."""
        try:
            raise ValueError("Database error: connection string contains sensitive data")
        except Exception as e:
            safe_msg = sanitize_error(e, is_production=True)
            sanitized_log = sanitize_log_message(f"Error occurred: {safe_msg}")

            assert safe_msg == "An error occurred. Please contact support."
            assert "sensitive data" not in sanitized_log

    def test_development_error_handling_flow(self):
        """Test complete error handling flow in development mode."""
        try:
            raise ValueError("Database error: connection timeout")
        except Exception as e:
            safe_msg = sanitize_error(e, is_production=False)
            sanitized_log = sanitize_log_message(f"Error occurred: {safe_msg}")

            assert "Database error" in safe_msg
            assert "connection timeout" in safe_msg
            # Log message should still be sanitized even in dev
            assert "***REDACTED***" not in sanitized_log  # No sensitive data in this message
