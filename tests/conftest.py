"""
This module contains pytest fixtures for setting up the test environment.

The fixtures handle:
- Providing paths to test configuration files
- Setting up environment variables
- Mocking environment variables for tests
- Resetting bot count before each test
"""

from pathlib import Path
from typing import Dict, Generator

import pytest

from chatbot_conversation.models.base import ChatbotBase
from chatbot_conversation.utils.env import APIConfig

APIConfig.setup_env()


@pytest.fixture
def test_config_path() -> str:
    """Fixture to provide the path to the test configuration file.

    Returns:
        str: Path to the test configuration file
    """
    return str(Path(__file__).parent / "fixtures" / "test_config.json")


@pytest.fixture
def test_config_empty_path() -> str:
    """Fixture to provide the path to an empty test configuration file.

    Returns:
        str: Path to the empty test configuration file
    """
    return str(Path(__file__).parent / "fixtures" / "test_config_empty.json")


@pytest.fixture
def setup_valid_env() -> None:
    """Fixture to set up environment with valid keys."""
    APIConfig.setup_env()


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    """Fixture to mock environment variables for tests.

    Args:
        monkeypatch: pytest's monkeypatch fixture

    Returns:
        Dict[str, str]: Dictionary of mocked environment variables
    """
    env_vars = {
        "OPENAI_API_KEY": "invalid-test-key",
        "ANTHROPIC_API_KEY": "invalid-test-key",
        "GOOGLE_API_KEY": "invalid-test-key",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture(autouse=True)
def reset_bot_count() -> Generator[None, None, None]:
    """Fixture to reset the bot count before each test."""
    ChatbotBase.reset_total_count()
    yield
