""" top level module for the tests defined in the tests directory """

from pathlib import Path

import pytest

from chatbot_conversation.models import ChatbotBase
from chatbot_conversation.utils import APIConfig

TEST_ROOT = "./newtests"


@pytest.fixture
def config_dir() -> Path:
    """returns the path to the test config data directory"""
    return Path(__file__) / "config"


@pytest.fixture
def output_dir() -> Path:
    """returns the path to the test data directory"""
    return Path(__file__) / "output"


@pytest.fixture(autouse=True)
def reset_chatbot_base() -> None:
    """Reset ChatbotBase class variables before each test.

    This ensures each test starts with a clean state for:
    - _total_count: Number of bot instances
    - _used_names: Set of used bot names
    """
    # Intentionally accessing protected members for testing purposes
    ChatbotBase._total_count = 0  # pyright: ignore[reportPrivateUsage]
    ChatbotBase._used_names.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def setup_valid_env() -> None:
    """Fixture to set up environment with valid keys."""
    APIConfig.setup_env()
