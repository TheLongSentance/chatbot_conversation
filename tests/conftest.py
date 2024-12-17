import os
import sys
import pytest
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from src.utils.env import APIConfig

# Add project root to path for imports in tests
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()

@pytest.fixture
def test_config_path() -> str:
    return str(Path(__file__).parent / 'fixtures' / 'test_config.json')

@pytest.fixture
def test_config_empty_path() -> str:
    return str(Path(__file__).parent / 'fixtures' / 'test_config_empty.json')

@pytest.fixture
def setup_valid_env():
    """Fixture to set up environment with valid keys."""
    APIConfig.setup_env()

@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    env_vars = {
        "OPENAI_API_KEY": "invalid-test-key",
        "ANTHROPIC_API_KEY": "invalid-test-key",
        "GOOGLE_API_KEY": "invalid-test-key"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars
