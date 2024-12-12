import pytest
from pathlib import Path

@pytest.fixture
def test_config_path():
    return str(Path(__file__).parent.parent / 'fixtures' / 'test_config.json')

@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test_key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test_key')
    monkeypatch.setenv('GOOGLE_API_KEY', 'test_key')
