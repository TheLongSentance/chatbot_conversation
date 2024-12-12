import os
import sys
import pytest
from typing import Dict

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

@pytest.fixture
def test_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'test_config.json')

@pytest.fixture
def mock_env() -> Dict[str, str]:
    return {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key"
    }
