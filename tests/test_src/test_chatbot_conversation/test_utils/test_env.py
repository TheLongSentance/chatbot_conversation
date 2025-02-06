"""Unit tests for the APIConfig class."""

import logging
import os
from pathlib import Path
from typing import List

from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from chatbot_conversation.utils.env import (
    CONFIG_DIR_ENV_VAR,
    DEFAULT_CONFIG_DIR,
    FILE_IN_PROJECT_ROOT,
    APIConfig,
)
from chatbot_conversation.utils.logging_util import LOGNAME_CONFIG


def test_get_config_dir_from_env(monkeypatch: MonkeyPatch) -> None:
    """Test get_config_dir_from_env returns expected values."""
    # Test when env var is not set
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR, raising=False)
    assert APIConfig.get_config_dir_from_env() is None

    # Test when env var is set
    test_path = "/test/config/path"
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, test_path)
    assert APIConfig.get_config_dir_from_env() == test_path


def test_get_default_config_dir(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Test get_default_config_dir returns expected values."""
    # Create a mock project structure
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / FILE_IN_PROJECT_ROOT).touch()
    config_dir = project_root / DEFAULT_CONFIG_DIR
    config_dir.mkdir()

    # Mock the current file location
    mock_file_path = project_root / "src" / "module" / "env.py"
    mock_file_path.parent.mkdir(parents=True)
    mock_file_path.touch()

    monkeypatch.setattr('chatbot_conversation.utils.env.__file__', str(mock_file_path))

    # Test finding config dir
    result = APIConfig.get_default_config_dir()
    assert result is not None
    assert result.name == DEFAULT_CONFIG_DIR
    assert result.parent == project_root

    # Test when project root marker is not found
    monkeypatch.setattr(
        'chatbot_conversation.utils.env.__file__',
        str(tmp_path / "nowhere" / "env.py")
    )
    assert APIConfig.get_default_config_dir() is None


def test_setup_env_missing_file(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture, mock_logging_config: None
) -> None:
    """Test setup_env handles missing .env files gracefully."""
    # First set up caplog
    caplog.set_level(logging.DEBUG)

    # Then create and set up the logger
    test_logger = logging.getLogger(LOGNAME_CONFIG)
    # Clear any existing handlers
    test_logger.handlers = []
    # Add the caplog handler
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Mock path operations
    monkeypatch.setattr(os.path, "join", lambda *args: "non_existent_path/.env")  # type: ignore
    monkeypatch.setattr(os.path, "exists", lambda x: False)  # type: ignore
    monkeypatch.setattr(APIConfig, "get_config_dir_from_env", lambda: None)
    monkeypatch.setattr(APIConfig, "get_default_config_dir", lambda: None)

    APIConfig.setup_env()

    assert "No .env file found in searched locations" in caplog.text


def test_load_config_no_env_files(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture, mock_logging_config: None
) -> None:
    """Test load_config when no .env files exist."""
    # First set up caplog
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIG)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Clear any existing environment variables
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR, raising=False)

    # Mock path.exists to return False for all paths
    monkeypatch.setattr(os.path, "exists", lambda x: False)  # type: ignore

    APIConfig._load_config()  # pyright: ignore[reportPrivateUsage]

    assert "No .env file found in searched locations" in caplog.text


def test_load_config_path_precedence(
    monkeypatch: MonkeyPatch,
    temp_env_files: List[str],
    caplog: LogCaptureFixture,
    mock_logging_config: None,
) -> None:
    """Test that .env files are loaded in the correct order of precedence."""
    # First set up caplog
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIG)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    # Set custom config dir to first temp file location
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, os.path.dirname(temp_env_files[0]))

    APIConfig._load_config()  # pyright: ignore[reportPrivateUsage]

    # Should load the first .env file (highest precedence)
    assert f"Loaded environment from: {temp_env_files[0]}" in caplog.text


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
    APIConfig._load_config()  # pyright: ignore[reportPrivateUsage]

    # Verify the environment variables were loaded from our mock file
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key is not None and openai_api_key.startswith("mock-openai-key")
    anthopic_api_key = os.getenv("ANTHROPIC_API_KEY")
    assert anthopic_api_key is not None and anthopic_api_key.startswith(
        "mock-anthropic-key"
    )
    google_api_key = os.getenv("GOOGLE_API_KEY")
    assert google_api_key is not None and google_api_key.startswith("mock-google-key")


def test_load_config_logs_api_keys(
    mock_env_keys: dict[str, str],
    caplog: LogCaptureFixture,
    mock_logging_config: None,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that API keys are properly logged when present."""
    # First set up caplog
    caplog.set_level(logging.DEBUG)

    # Set up the logger with caplog handler
    test_logger = logging.getLogger(LOGNAME_CONFIG)
    test_logger.handlers = []
    test_logger.addHandler(caplog.handler)
    monkeypatch.setattr("chatbot_conversation.utils.env.logger", test_logger)

    APIConfig._load_config()  # pyright: ignore[reportPrivateUsage]

    for key in mock_env_keys.keys():
        assert f"{key} is set in environment" in caplog.text
