"""Unit tests for the APIConfig class."""

import logging
import os
from pathlib import Path

from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from chatbot_conversation.utils.env import APIConfig
from chatbot_conversation.utils.logging_util import LOGNAME_CONFIGURATION


def test_setup_env_missing_file(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    mock_logging_config: None,
    tmp_path: Path,
) -> None:
    """Test setup_env handles missing .env files gracefully.

    Args:
        monkeypatch: Fixture for mocking
        caplog: Fixture for capturing logs
        mock_logging_config: Fixture for configuring logging
        tmp_path: Temporary directory path
    """
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIGURATION)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Mock current working directory to a clean temporary directory
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

    APIConfig.setup_env()

    assert "No .env file found in current directory" in caplog.text


def test_load_config_with_env_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test loading configuration from .env file.

    Args:
        monkeypatch: Fixture for mocking
        tmp_path: PyTest's temporary path fixture
    """
    # Create test .env file in temporary directory
    env_file = tmp_path / ".env"
    env_content = """
OPENAI_API_KEY=mock-openai-key-12345678
ANTHROPIC_API_KEY=mock-anthropic-key-12345678
GOOGLE_API_KEY=mock-google-key-12345678
"""
    env_file.write_text(env_content.strip())

    # Mock current working directory
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

    # Clear any existing environment variables
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    APIConfig._load_config()  # pyright: ignore[reportPrivateUsage]

    # Verify the environment variables were loaded
    assert os.getenv("OPENAI_API_KEY") == "mock-openai-key-12345678"
    assert os.getenv("ANTHROPIC_API_KEY") == "mock-anthropic-key-12345678"
    assert os.getenv("GOOGLE_API_KEY") == "mock-google-key-12345678"


def test_env_precedence(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: LogCaptureFixture,
    mock_logging_config: None,
) -> None:
    """Test that environment variables take precedence over .env files."""
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIGURATION)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Create .env file with different value
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=env-file-value")

    # Set environment variable directly
    direct_key = "direct-key-value"
    monkeypatch.setenv("OPENAI_API_KEY", direct_key)

    # Mock current working directory
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

    APIConfig.setup_env()

    assert os.getenv("OPENAI_API_KEY") == direct_key
    assert "OPENAI_API_KEY is set in environment" in caplog.text


def test_custom_api_key(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    mock_logging_config: None,
    tmp_path: Path,
) -> None:
    """Test handling of custom API key environment variables."""
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIGURATION)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Create .env file with custom API key
    env_file = tmp_path / ".env"
    env_file.write_text("CUSTOM_API_KEY=test-value")

    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)

    APIConfig.setup_env()

    assert os.getenv("CUSTOM_API_KEY") == "test-value"
    assert "CUSTOM_API_KEY is set in environment" in caplog.text
