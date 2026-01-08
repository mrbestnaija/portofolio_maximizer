"""
Secret loader tests.

Tests for secure secret loading from Docker secrets or environment variables.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile

from etl.secret_loader import (
    load_secret,
    load_api_key,
    load_alpha_vantage_key,
    load_finnhub_key,
    load_xtb_username,
    load_xtb_password,
)


class TestSecretLoader:
    """Test secret loading functionality."""
    
    def test_load_secret_from_environment_variable(self):
        """Should load secret from environment variable."""
        with patch.dict(os.environ, {'TEST_API_KEY': 'test_key_123'}):
            result = load_secret('TEST_API_KEY')
            assert result == 'test_key_123'
    
    def test_load_secret_from_docker_secret_file(self, tmp_path):
        """Should load secret from Docker secret file."""
        secret_file = tmp_path / "test_secret.txt"
        secret_file.write_text("docker_secret_key_456")
        
        with patch.dict(os.environ, {
            'TEST_API_KEY_FILE': str(secret_file),
            'TEST_API_KEY': 'env_key_123'  # Should be ignored
        }):
            result = load_secret('TEST_API_KEY', 'TEST_API_KEY_FILE')
            # Docker secret takes priority
            assert result == 'docker_secret_key_456'
    
    def test_load_secret_fallback_to_env_when_file_missing(self):
        """Should fallback to environment variable when secret file doesn't exist."""
        with patch.dict(os.environ, {
            'TEST_API_KEY_FILE': '/nonexistent/path/secret.txt',
            'TEST_API_KEY': 'env_fallback_key'
        }):
            result = load_secret('TEST_API_KEY', 'TEST_API_KEY_FILE')
            assert result == 'env_fallback_key'
    
    def test_load_secret_returns_none_when_not_found(self):
        """Should return None when secret is not found."""
        with patch.dict(os.environ, {}, clear=True):
            result = load_secret('NONEXISTENT_KEY')
            assert result is None
    
    def test_load_secret_ignores_comments_in_secret_file(self, tmp_path):
        """Should ignore comment lines in secret files."""
        secret_file = tmp_path / "test_secret.txt"
        secret_file.write_text("# This is a comment\nactual_secret_value")
        
        with patch.dict(os.environ, {'TEST_API_KEY_FILE': str(secret_file)}):
            result = load_secret('TEST_API_KEY', 'TEST_API_KEY_FILE')
            assert result == 'actual_secret_value'
    
    def test_load_secret_handles_empty_secret_file(self, tmp_path):
        """Should fallback when secret file is empty."""
        secret_file = tmp_path / "empty_secret.txt"
        secret_file.write_text("")
        
        with patch.dict(os.environ, {
            'TEST_API_KEY_FILE': str(secret_file),
            'TEST_API_KEY': 'env_key'
        }):
            result = load_secret('TEST_API_KEY', 'TEST_API_KEY_FILE')
            # Should fallback to env var
            assert result == 'env_key'
    
    def test_load_secret_handles_file_read_error(self, tmp_path):
        """Should fallback when secret file cannot be read."""
        # Create a directory path instead of file path
        secret_dir = tmp_path / "secret_dir"
        secret_dir.mkdir()
        
        with patch.dict(os.environ, {
            'TEST_API_KEY_FILE': str(secret_dir),
            'TEST_API_KEY': 'env_key_fallback'
        }):
            result = load_secret('TEST_API_KEY', 'TEST_API_KEY_FILE')
            # Should fallback to env var
            assert result == 'env_key_fallback'
    
    def test_load_secret_auto_detects_file_env_var(self):
        """Should auto-detect _FILE suffix for secret file path."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("auto_detected_secret")
            secret_file_path = f.name
        
        try:
            with patch.dict(os.environ, {'TEST_KEY_FILE': secret_file_path}):
                result = load_secret('TEST_KEY')
                assert result == 'auto_detected_secret'
        finally:
            os.unlink(secret_file_path)


class TestConvenienceFunctions:
    """Test convenience functions for common API keys."""
    
    def test_load_api_key(self):
        """Test generic API key loader."""
        with patch.dict(os.environ, {'MY_API_KEY': 'test_key'}):
            result = load_api_key('MY_API_KEY')
            assert result == 'test_key'
    
    def test_load_alpha_vantage_key(self):
        """Test Alpha Vantage key loader."""
        with patch.dict(os.environ, {'ALPHA_VANTAGE_API_KEY': 'alpha_key_123'}):
            result = load_alpha_vantage_key()
            assert result == 'alpha_key_123'
    
    def test_load_finnhub_key(self):
        """Test Finnhub key loader."""
        with patch.dict(os.environ, {'FINNHUB_API_KEY': 'finnhub_key_456'}):
            result = load_finnhub_key()
            assert result == 'finnhub_key_456'
    
    def test_load_xtb_username(self):
        """Test XTB username loader."""
        with patch.dict(os.environ, {'XTB_USERNAME': 'xtb_user'}):
            result = load_xtb_username()
            assert result == 'xtb_user'
    
    def test_load_xtb_password(self):
        """Test XTB password loader."""
        with patch.dict(os.environ, {'XTB_PASSWORD': 'xtb_pass_123'}):
            result = load_xtb_password()
            assert result == 'xtb_pass_123'


@pytest.mark.security
class TestSecretLoaderIntegration:
    """Integration tests for secret loading."""
    
    def test_docker_secret_priority_over_env_var(self, tmp_path):
        """Docker secrets should take priority over environment variables."""
        # Create Docker secret file
        secret_file = tmp_path / "alpha_vantage_api_key.txt"
        secret_file.write_text("docker_secret_value")
        
        with patch.dict(os.environ, {
            'ALPHA_VANTAGE_API_KEY_FILE': str(secret_file),
            'ALPHA_VANTAGE_API_KEY': 'env_var_value'
        }):
            result = load_alpha_vantage_key()
            # Docker secret should be used
            assert result == 'docker_secret_value'
    
    def test_secret_loader_backward_compatibility(self):
        """Secret loader should work with existing .env file approach."""
        with patch.dict(os.environ, {'ALPHA_VANTAGE_API_KEY': 'env_key'}):
            result = load_alpha_vantage_key()
            assert result == 'env_key'
    
    def test_multiple_secrets_loading(self):
        """Should load multiple secrets independently."""
        with patch.dict(os.environ, {
            'ALPHA_VANTAGE_API_KEY': 'alpha_key',
            'FINNHUB_API_KEY': 'finnhub_key',
            'XTB_USERNAME': 'xtb_user',
            'XTB_PASSWORD': 'xtb_pass'
        }):
            assert load_alpha_vantage_key() == 'alpha_key'
            assert load_finnhub_key() == 'finnhub_key'
            assert load_xtb_username() == 'xtb_user'
            assert load_xtb_password() == 'xtb_pass'

