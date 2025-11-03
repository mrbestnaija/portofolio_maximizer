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
from unittest.mock import Mock

from ai_llm.ollama_client import OllamaClient, OllamaConnectionError


def _mock_response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _http_client():
    client = Mock()
    client.get = Mock()
    client.post = Mock()
    client.close = Mock()
    return client


class TestOllamaClientInitialization:
    """Test Ollama client initialization and validation"""

    def test_init_validates_connection(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })

        client = OllamaClient(http_client=http_client)
        assert client.model == 'deepseek-coder:6.7b-instruct-q4_K_M'
        assert client.host == 'http://localhost:11434'

    def test_init_fails_when_server_unavailable(self):
        http_client = _http_client()
        http_client.get.side_effect = requests.exceptions.ConnectionError()

        with pytest.raises(OllamaConnectionError) as exc_info:
            OllamaClient(http_client=http_client)

        assert 'not running' in str(exc_info.value).lower()

    def test_init_fails_when_model_not_found(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'different-model'}]
        })

        with pytest.raises(OllamaConnectionError) as exc_info:
            OllamaClient(model='missing-model', http_client=http_client)

        assert 'not found' in str(exc_info.value).lower()

    def test_init_handles_timeout(self):
        http_client = _http_client()
        http_client.get.side_effect = requests.exceptions.Timeout()

        with pytest.raises(OllamaConnectionError) as exc_info:
            OllamaClient(http_client=http_client)

        assert 'timeout' in str(exc_info.value).lower()


class TestOllamaGeneration:
    """Test LLM text generation"""

    def test_generate_returns_text(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })
        http_client.post.return_value = _mock_response({'response': 'Generated text response'})

        client = OllamaClient(http_client=http_client)
        result = client.generate('Test prompt')

        assert result == 'Generated text response'
        http_client.post.assert_called_once()

    def test_generate_handles_timeout(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })
        http_client.post.side_effect = requests.exceptions.Timeout()

        client = OllamaClient(http_client=http_client, latency_failover_threshold=1.0)
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate('Test prompt')

        assert 'timeout' in str(exc_info.value).lower()

    def test_generate_rejects_empty_response(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })
        http_client.post.return_value = _mock_response({'response': ''})

        client = OllamaClient(http_client=http_client)
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate('Test prompt')

        assert 'empty' in str(exc_info.value).lower()

    def test_generate_with_system_prompt(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })
        http_client.post.return_value = _mock_response({'response': 'Test response'})

        client = OllamaClient(http_client=http_client)
        result = client.generate('Test', system='You are a helpful assistant')

        assert result == 'Test response'
        call_args = http_client.post.call_args[1]['json']
        assert call_args['system'] == 'You are a helpful assistant'

    def test_generate_uses_cache_for_repeat_prompts(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })
        http_client.post.return_value = _mock_response({'response': 'Cached output'})

        client = OllamaClient(http_client=http_client, cache_ttl_seconds=120)
        first = client.generate('Repeatable prompt', system='system')
        assert first == 'Cached output'

        http_client.post.reset_mock()
        second = client.generate('Repeatable prompt', system='system')
        assert second == 'Cached output'
        http_client.post.assert_not_called()

    def test_prompt_is_optimised_before_generation(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })
        http_client.post.return_value = _mock_response({'response': 'OK'})

        noisy_prompt = """
        BUY signal
        BUY signal

        BUY signal with   extra   spaces
        """

        client = OllamaClient(http_client=http_client)
        client.generate(noisy_prompt)

        sent_prompt = http_client.post.call_args[1]['json']['prompt']
        assert 'extra   spaces' not in sent_prompt
        assert sent_prompt.count('BUY signal') == 1


class TestModelInfo:
    """Test model information retrieval"""

    def test_get_model_info_returns_details(self):
        http_client = _http_client()
        http_client.get.side_effect = [
            _mock_response({
                'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
            }),
            _mock_response({
                'models': [{
                    'name': 'deepseek-coder:6.7b-instruct-q4_K_M',
                    'size': '4.1GB',
                    'modified_at': '2024-01-01',
                    'details': {'family': 'deepseek'}
                }]
            })
        ]

        client = OllamaClient(http_client=http_client)
        info = client.get_model_info()

        assert info['name'] == 'deepseek-coder:6.7b-instruct-q4_K_M'
        assert info['size'] == '4.1GB'
        assert info['family'] == 'deepseek'

    def test_get_model_info_handles_failure_gracefully(self):
        http_client = _http_client()
        http_client.get.side_effect = [
            _mock_response({
                'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
            }),
            Exception('API error')
        ]

        client = OllamaClient(http_client=http_client)
        info = client.get_model_info()

        assert info == {}


class TestConnectionValidation:
    """Test connection validation logic"""

    def test_validation_checks_server_health(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })

        OllamaClient(http_client=http_client)

        assert http_client.get.called
        call_url = http_client.get.call_args[0][0]
        assert '/api/tags' in call_url

    def test_validation_runs_on_every_init(self):
        http_client = _http_client()
        http_client.get.return_value = _mock_response({
            'models': [{'name': 'deepseek-coder:6.7b-instruct-q4_K_M'}]
        })

        OllamaClient(http_client=http_client)
        OllamaClient(http_client=http_client)

        assert http_client.get.call_count >= 2


# Performance validation

def test_ollama_client_has_docstrings():
    """Validate documentation exists"""
    assert OllamaClient.__doc__ is not None
    assert OllamaClient.__init__.__doc__ is not None
    assert OllamaClient.generate.__doc__ is not None
    assert len(OllamaClient.__doc__) > 50


# Line count: ~190 lines (within 500-line budget)
