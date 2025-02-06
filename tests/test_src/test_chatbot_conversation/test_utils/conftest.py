"""Fixtures for testing environment configuration."""

import logging.config
import os
from pathlib import Path
from typing import Dict, Generator, List

import pytest
from _pytest.monkeypatch import MonkeyPatch


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
def temp_logging_conf(tmp_path: Path) -> Generator[str, None, None]:
    """Fixture to create a temporary logging.conf file."""
    config_dir: Path = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    log_conf: Path = config_dir / "logging.conf"

    conf_content = """
[loggers]
keys=root,api,config,model,system,validation

[handlers]
keys=consoleHandler

[formatters]
keys=testFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_api]
level=DEBUG
handlers=consoleHandler
qualname=api
propagate=0

[logger_config]
level=DEBUG
handlers=consoleHandler
qualname=config
propagate=0

[logger_model]
level=DEBUG
handlers=consoleHandler
qualname=model
propagate=0

[logger_system]
level=DEBUG
handlers=consoleHandler
qualname=system
propagate=0

[logger_validation]
level=DEBUG
handlers=consoleHandler
qualname=validation
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=testFormatter
args=(sys.stdout,)

[formatter_testFormatter]
format=%(name)s - %(levelname)s - %(message)s
"""
    log_conf.write_text(conf_content.strip())
    try:
        yield str(log_conf)
    finally:
        if log_conf.exists():
            log_conf.unlink()


@pytest.fixture
def mock_logging_config(
    temp_logging_conf: str, monkeypatch: MonkeyPatch
) -> Generator[None, None, None]:
    """Setup mock logging configuration.

    Args:
        temp_logging_conf: Fixture providing path to temporary logging.conf
        monkeypatch: PyTest's monkeypatch fixture

    Yields:
        None
    """
    # Store original config path
    original_path = os.environ.get("LOGGING_CONFIG_PATH")

    # Set up mock config path
    monkeypatch.setenv("LOGGING_CONFIG_PATH", temp_logging_conf)

    # Reset logging config
    logging.shutdown()
    logging.config.fileConfig(temp_logging_conf)

    yield

    # Restore original config path
    if original_path:
        monkeypatch.setenv("LOGGING_CONFIG_PATH", original_path)
    else:
        monkeypatch.delenv("LOGGING_CONFIG_PATH", raising=False)
