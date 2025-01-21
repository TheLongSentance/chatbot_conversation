"""Unit tests for the APIConfig class."""

import os
from typing import Dict
import logging
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from chatbot_conversation.utils.env import APIConfig


def test_setup_env_missing_file(monkeypatch: MonkeyPatch) -> None:
    """Test setup_env raises FileNotFoundError when .env file is missing."""
    # Mock the path to the .env file to a non-existent location
    monkeypatch.setattr(os.path, "join", lambda *args: "non_existent_path/.env")  # type: ignore

    with pytest.raises(FileNotFoundError):
        APIConfig.setup_env()


def test_validate_keys_all_present(mock_env_keys: Dict[str, str],
                                 caplog: LogCaptureFixture) -> None:
    """Test key validation when all keys are present.

    Args:
        mock_env_keys: Fixture providing mock API keys
        caplog: Fixture for capturing log output
    """
    with caplog.at_level(logging.INFO):
        keys = [
            ("OpenAI", os.getenv("OPENAI_API_KEY")),
            ("Anthropic", os.getenv("ANTHROPIC_API_KEY")),
            ("Google", os.getenv("GOOGLE_API_KEY")),
        ]
        APIConfig._validate_keys(keys) # type: ignore

    assert "OpenAI API Key exists" in caplog.text
    assert "Anthropic API Key exists" in caplog.text
    assert "Google API Key exists" in caplog.text


def test_validate_keys_missing(caplog: LogCaptureFixture) -> None:
    """Test key validation when keys are missing.

    Args:
        caplog: Fixture for capturing log output
    """
    with caplog.at_level(logging.WARNING):
        keys = [
            ("OpenAI", None),
            ("Anthropic", None),
            ("Google", None),
        ]
        APIConfig._validate_keys(keys)  # type: ignore

    assert "OpenAI API Key not set" in caplog.text
    assert "Anthropic API Key not set" in caplog.text
    assert "Google API Key not set" in caplog.text


def test_load_config_with_env_file(
    monkeypatch: MonkeyPatch, temp_env_file: str
) -> None:
    """Test loading configuration from .env file.

    Args:
        monkeypatch: Fixture for modifying environment
        temp_env_file: Fixture providing temporary .env file
    """

    # Mock the path resolution to use our temporary env file
    def mock_join(*args: str) -> str:
        return temp_env_file

    monkeypatch.setattr(os.path, "join", mock_join)

    # Clear any existing environment variables
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    # Call the actual load_config method
    APIConfig._load_config()  # type: ignore

    # Verify the environment variables were loaded from our mock file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key is not None and openai_api_key.startswith("mock-openai-key")
    anthopic_api_key = os.getenv("ANTHROPIC_API_KEY")
    assert anthopic_api_key is not None and anthopic_api_key.startswith(
        "mock-anthropic-key"
    )
    google_api_key = os.getenv("GOOGLE_API_KEY")
    assert google_api_key is not None and google_api_key.startswith("mock-google-key")


def test_log_key_status(caplog: LogCaptureFixture) -> None:
    """Test logging of key status.

    Args:
        caplog: Fixture for capturing log output
    """
    with caplog.at_level(logging.INFO):
        APIConfig._log_key_status("Test", "test-key-12345678")  # type: ignore
        assert "Test API Key exists and begins test-key" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        APIConfig._log_key_status("Test", None)  # type: ignore
        assert "Test API Key not set" in caplog.text
