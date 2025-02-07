"""Unit tests for the APIConfig class."""

import logging
import os
from pathlib import Path
from typing import Dict

from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from chatbot_conversation.utils.env import APIConfig
from chatbot_conversation.utils.logging_util import LOGNAME_CONFIGURATION


def test_setup_env_missing_file(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    mock_logging_config: None,
    mock_config_dir: Path,
) -> None:
    """Test setup_env handles missing .env files gracefully.

    Args:
        monkeypatch: Fixture for mocking
        caplog: Fixture for capturing logs
        mock_logging_config: Fixture for configuring logging
        mock_config_dir: Fixture providing mock config directory
    """
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIGURATION)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Mock get_config_dir to return our mock directory
    monkeypatch.setattr(
        "chatbot_conversation.utils.env.get_config_dir", lambda: mock_config_dir
    )

    APIConfig.setup_env()

    assert "No .env file found in config directory" in caplog.text


def test_load_config_with_env_file(
    monkeypatch: MonkeyPatch,
    mock_config_dir: Path,
    tmp_path: Path,
) -> None:
    """Test loading configuration from .env file.

    Args:
        monkeypatch: Fixture for mocking
        mock_config_dir: Fixture providing mock config directory
        tmp_path: PyTest's temporary path fixture
    """
    # Create test .env file in mock config directory
    env_file = mock_config_dir / ".env"
    env_content = """
OPENAI_API_KEY=mock-openai-key-12345678
ANTHROPIC_API_KEY=mock-anthropic-key-12345678
GOOGLE_API_KEY=mock-google-key-12345678
"""
    env_file.write_text(env_content.strip())

    # Mock get_config_dir to return our mock directory
    monkeypatch.setattr(
        "chatbot_conversation.utils.env.get_config_dir", lambda: mock_config_dir
    )

    # Clear any existing environment variables
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    APIConfig._load_config()  # pyright: ignore[reportPrivateUsage]

    # Verify the environment variables were loaded
    assert os.getenv("OPENAI_API_KEY") == "mock-openai-key-12345678"
    assert os.getenv("ANTHROPIC_API_KEY") == "mock-anthropic-key-12345678"
    assert os.getenv("GOOGLE_API_KEY") == "mock-google-key-12345678"


def test_load_config_logs_api_keys(
    mock_env_keys: Dict[str, str],
    caplog: LogCaptureFixture,
    mock_logging_config: None,
    monkeypatch: MonkeyPatch,
    mock_config_dir: Path,
) -> None:
    """Test that API keys are properly logged when present.

    Args:
        mock_env_keys: Fixture providing mock API keys
        caplog: Fixture for capturing logs
        mock_logging_config: Fixture for configuring logging
        monkeypatch: Fixture for mocking
        mock_config_dir: Fixture providing mock config directory
    """
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIGURATION)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Mock get_config_dir to return our mock directory
    monkeypatch.setattr(
        "chatbot_conversation.utils.env.get_config_dir", lambda: mock_config_dir
    )

    APIConfig._load_config()  # pyright: ignore[reportPrivateUsage]

    for key in mock_env_keys.keys():
        assert f"{key} is set in environment" in caplog.text
