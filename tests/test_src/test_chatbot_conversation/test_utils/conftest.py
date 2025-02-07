"""Fixtures for testing environment configuration."""

import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Generator, List

import pytest
from _pytest.monkeypatch import MonkeyPatch

from chatbot_conversation.utils.dir_util import FILE_IN_PROJECT_ROOT


@pytest.fixture
def mock_env_keys(monkeypatch: MonkeyPatch) -> Dict[str, str]:
    """Fixture to provide mock API keys.

    Args:
        monkeypatch: PyTest's monkeypatch fixture

    Returns:
        Dict containing mock API keys
    """
    mock_keys = {
        "OPENAI_API_KEY": "mock-openai-key-12345678",
        "ANTHROPIC_API_KEY": "mock-anthropic-key-12345678",
        "GOOGLE_API_KEY": "mock-google-key-12345678",
    }
    for key, value in mock_keys.items():
        monkeypatch.setenv(key, value)
    return mock_keys


@pytest.fixture
def temp_env_file(tmp_path: Path) -> Generator[str, None, None]:
    """Fixture to create a temporary .env file.

    Args:
        tmp_path: PyTest's temporary path fixture providing a temporary directory

    Yields:
        str: Path to temporary .env file as a string
    """
    config_dir: Path = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    env_file: Path = config_dir / ".env"

    env_content = """
OPENAI_API_KEY=mock-openai-key-12345678
ANTHROPIC_API_KEY=mock-anthropic-key-12345678
GOOGLE_API_KEY=mock-google-key-12345678
"""
    env_file.write_text(env_content.strip())
    try:
        yield str(env_file)
    finally:
        if env_file.exists():
            env_file.unlink()


@pytest.fixture
def temp_env_files(tmp_path: Path) -> Generator[List[str], None, None]:
    """Create multiple temporary .env files for testing path precedence.

    Args:
        tmp_path: PyTest's temporary path fixture

    Returns:
        List of paths to temporary .env files
    """
    env_files: List[str] = []

    # Create three different config directories
    for i in range(3):
        config_dir = tmp_path / f"config_{i}"
        config_dir.mkdir(exist_ok=True)
        env_file = config_dir / ".env"

        env_content = f"""
OPENAI_API_KEY=mock-openai-key-{i}
ANTHROPIC_API_KEY=mock-anthropic-key-{i}
GOOGLE_API_KEY=mock-google-key-{i}
"""
        env_file.write_text(env_content.strip())
        env_files.append(str(env_file))

    try:
        yield env_files
    finally:
        for file_path in env_files:
            if os.path.exists(file_path):
                os.unlink(file_path)


@pytest.fixture
def mock_logging_config() -> Generator[None, None, None]:
    """Setup mock logging configuration using dictConfig.

    Yields:
        None
    """
    # Reset logging config
    logging.shutdown()

    # Configure test logging
    test_config: Dict[str, Any] = {
        "version": 1,
        "formatters": {
            "testFormatter": {"format": "%(name)s - %(levelname)s - %(message)s"}
        },
        "handlers": {
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "testFormatter",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "root": {"level": "DEBUG", "handlers": ["consoleHandler"]},
            "api": {
                "level": "DEBUG",
                "handlers": ["consoleHandler"],
                "propagate": False,
            },
            "configuration": {
                "level": "DEBUG",
                "handlers": ["consoleHandler"],
                "propagate": False,
            },
            "conversation": {
                "level": "DEBUG",
                "handlers": ["consoleHandler"],
                "propagate": False,
            },
            "models": {
                "level": "DEBUG",
                "handlers": ["consoleHandler"],
                "propagate": False,
            },
            "system": {
                "level": "DEBUG",
                "handlers": ["consoleHandler"],
                "propagate": False,
            },
            "utils": {
                "level": "DEBUG",
                "handlers": ["consoleHandler"],
                "propagate": False,
            },
            "validation": {
                "level": "DEBUG",
                "handlers": ["consoleHandler"],
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(test_config)
    yield


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory.

    Args:
        tmp_path: PyTest's temporary path fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def temp_project_root(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary project root structure with pyproject.toml.

    Creates a mock project root directory containing a pyproject.toml file
    for testing directory location logic. Changes the working directory to
    the project root during the test.

    Args:
        temp_dir: Fixture providing temporary directory

    Yields:
        Path: Path to temporary project root containing pyproject.toml
    """
    # Create mock project root with pyproject.toml
    project_root = temp_dir / "project_root"
    project_root.mkdir(parents=True)
    (project_root / FILE_IN_PROJECT_ROOT).touch()

    # Change to project directory during test
    original_dir = os.getcwd()
    os.chdir(project_root)

    try:
        yield project_root
    finally:
        os.chdir(original_dir)


@pytest.fixture
def mock_config_dir(tmp_path: Path) -> Path:
    """Fixture providing a mock config directory.

    Args:
        tmp_path: PyTest's temporary path fixture

    Returns:
        Path: Path to mock config directory
    """
    config_dir = tmp_path / "mock_config"
    config_dir.mkdir(parents=True)
    return config_dir
