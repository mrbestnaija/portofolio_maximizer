"""
Unit tests for Ollama Client
Line Count: ~180 lines (within 500-line test budget)

Per AGENT_INSTRUCTION.md:
- Test only critical business logic
- Validate connection failure handling
- Test LLM response parsing
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
import json

from ai_llm.ollama_client import OllamaClient, OllamaConnectionError


class TestOllamaClientInitialization:
    """Test Ollama client initialization and validation"""
    
    def test_init_validates_connection(self):
        """Test that initialization validates Ollama availability"""
        with patch('requests.get') as mock_get:
            # Mock successful connection
            mock_response = Mock()
            mock_response.json.return_value = {
                'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            client = OllamaClient()
            assert client.model == 'deepseek-coder:6.7b-instruct-q4_K_M'
            assert client.host == 'http://localhost:11434'
    
    def test_init_fails_when_server_unavailable(self):
        """Test fail-fast when Ollama server not running"""
        with patch('requests.get', side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(OllamaConnectionError) as exc_info:
                OllamaClient()
            
            assert 'not running' in str(exc_info.value).lower()
    
    def test_init_fails_when_model_not_found(self):
        """Test fail-fast when requested model not available"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'models': [{'name': 'different-model'}]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with pytest.raises(OllamaConnectionError) as exc_info:
                OllamaClient(model='missing-model')
            
            assert 'not found' in str(exc_info.value).lower()
    
    def test_init_handles_timeout(self):
        """Test timeout handling during initialization"""
        with patch('requests.get', side_effect=requests.exceptions.Timeout()):
            with pytest.raises(OllamaConnectionError) as exc_info:
                OllamaClient()
            
            assert 'timeout' in str(exc_info.value).lower()


class TestOllamaGeneration:
    """Test LLM text generation"""
    
    @patch('requests.post')
    @patch('requests.get')
    def test_generate_returns_text(self, mock_get, mock_post):
        """Test successful text generation"""
        # Mock init validation
        mock_response = Mock()
        mock_response.json.return_value = {
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock generation response
        mock_post.return_value.json.return_value = {
            'response': 'Generated text response'
        }
        
        client = OllamaClient()
        result = client.generate('Test prompt')
        
        assert result == 'Generated text response'
        assert mock_post.called
    
    @patch('requests.post')
    @patch('requests.get')
    def test_generate_handles_timeout(self, mock_get, mock_post):
        """Test generation timeout handling"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        mock_post.side_effect = requests.exceptions.Timeout()
        
        client = OllamaClient()
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate('Test prompt')
        
        assert 'timeout' in str(exc_info.value).lower()
    
    @patch('requests.post')
    @patch('requests.get')
    def test_generate_rejects_empty_response(self, mock_get, mock_post):
        """Test that empty responses raise error"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        mock_post.return_value.json.return_value = {'response': ''}
        
        client = OllamaClient()
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate('Test prompt')
        
        assert 'empty' in str(exc_info.value).lower()
    
    @patch('requests.post')
    @patch('requests.get')
    def test_generate_with_system_prompt(self, mock_get, mock_post):
        """Test generation with system instructions"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        mock_post.return_value.json.return_value = {'response': 'Test response'}
        
        client = OllamaClient()
        result = client.generate('Test', system='You are a helpful assistant')
        
        # Verify system prompt was included in request
        call_args = mock_post.call_args[1]['json']
        assert 'system' in call_args
        assert call_args['system'] == 'You are a helpful assistant'

    @patch('requests.post')
    @patch('requests.get')
    def test_generate_uses_cache_for_repeat_prompts(self, mock_get, mock_post):
        """Repeated prompts should return cached response without new HTTP call."""
        init_response = Mock()
        init_response.json.return_value = {
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        }
        init_response.raise_for_status.return_value = None
        mock_get.return_value = init_response

        generation_response = Mock()
        generation_response.json.return_value = {'response': 'Cached output'}
        generation_response.raise_for_status.return_value = None
        mock_post.return_value = generation_response

        client = OllamaClient()
        first = client.generate('Repeatable prompt', system='system')
        assert first == 'Cached output'

        mock_post.reset_mock()
        second = client.generate('Repeatable prompt', system='system')
        assert second == 'Cached output'
        mock_post.assert_not_called()


class TestModelInfo:
    """Test model information retrieval"""
    
    @patch('requests.get')
    def test_get_model_info_returns_details(self, mock_get):
        """Test successful model info retrieval"""
        mock_responses = [
            # First call: init validation
            Mock(json=lambda: {
                'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
            }),
            # Second call: get_model_info
            Mock(json=lambda: {
                'models': [{
                    'name': 'deepseek-coder:6.7b-instruct-q4_K_M',
                    'size': '4.1GB',
                    'modified_at': '2024-01-01',
                    'details': {'family': 'deepseek'}
                }]
            })
        ]
        mock_get.side_effect = mock_responses
        
        client = OllamaClient()
        info = client.get_model_info()
        
        assert info['name'] == 'deepseek-coder:6.7b-instruct-q4_K_M'
        assert info['size'] == '4.1GB'
        assert info['family'] == 'deepseek'
    
    @patch('requests.get')
    def test_get_model_info_handles_failure_gracefully(self, mock_get):
        """Test graceful handling of info retrieval failure"""
        mock_responses = [
            Mock(json=lambda: {
                'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
            }),
            Mock(side_effect=Exception('API error'))
        ]
        mock_get.side_effect = mock_responses
        
        client = OllamaClient()
        info = client.get_model_info()
        
        # Should return empty dict on failure, not crash
        assert info == {}


class TestConnectionValidation:
    """Test connection validation logic"""
    
    def test_validation_checks_server_health(self):
        """Test that validation checks server endpoint"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {
                'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
            }
            
            client = OllamaClient()
            
            # Verify /api/tags endpoint was called
            assert mock_get.called
            call_url = mock_get.call_args[0][0]
            assert '/api/tags' in call_url
    
    def test_validation_runs_on_every_init(self):
        """Test that validation runs each time client is created"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {
                'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
            }
            
            # Create multiple instances
            client1 = OllamaClient()
            client2 = OllamaClient()
            
            # Validation should run twice
            assert mock_get.call_count >= 2


# Performance validation
def test_ollama_client_has_docstrings():
    """Validate documentation exists"""
    assert OllamaClient.__doc__ is not None
    assert OllamaClient.__init__.__doc__ is not None
    assert OllamaClient.generate.__doc__ is not None
    assert len(OllamaClient.__doc__) > 50


# Line count: ~190 lines (within 500-line budget)
