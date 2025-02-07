"""Unit tests for directory utility functions.

Tests functionality for locating and creating configuration and output directories
based on environment variables, project structure, and fallback options.
"""

from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch

from chatbot_conversation.utils.dir_util import (
    CONFIG_DIR_ENV_VAR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_DIR_ENV_VAR,
    get_config_dir,
    get_output_dir,
)


def test_get_config_dir_from_env(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test configuration directory retrieval from environment variable.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_dir: Fixture providing temporary directory path
    """
    # Create the config directory first since get_config_dir() won't create it
    config_path = temp_dir / "custom_config"
    config_path.mkdir(parents=True)
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(config_path))

    result = get_config_dir()

    assert result == Path(config_path)
    assert result.exists()
    assert result.is_dir()


def test_get_config_dir_from_project_root(temp_project_root: Path) -> None:
    """Test configuration directory retrieval from project root.

    Args:
        temp_project_root: Fixture providing temporary project structure
    """
    # For config directories, if directory doesn't exist under project root,
    # return local directory which happens to be the project root in this case
    result = get_config_dir()

    assert result == temp_project_root
    assert result.exists()
    assert result.is_dir()


def test_get_config_dir_fallback(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test configuration directory fallback to current directory.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_dir: Temporary directory to use as current working directory
    """
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(temp_dir)

    result = get_config_dir()

    assert result == temp_dir


def test_get_config_dir_precedence(
    monkeypatch: MonkeyPatch, temp_project_root: Path, temp_dir: Path
) -> None:
    """Test configuration directory source precedence.

    Verifies that environment variable takes precedence over project root
    only when the environment variable directory exists.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_project_root: Fixture providing temporary project structure
        temp_dir: Fixture providing temporary directory
    """
    # Setup and create env var config path
    env_config = temp_dir / "env_config"
    env_config.mkdir(parents=True)  # Create the directory first
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_config))

    result = get_config_dir()

    # Should prefer existing env var over project root
    assert result == Path(env_config)
    assert result != temp_project_root / DEFAULT_CONFIG_DIR


def test_get_config_dir_no_create(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test configuration directory retrieval without directory creation.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_dir: Fixture providing temporary directory path
    """
    config_path = temp_dir / "nonexistent_config"
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(config_path))
    # Change working directory to temp_dir to avoid finding project root
    monkeypatch.chdir(temp_dir)

    result = (
        get_config_dir()
    )  # Should fall back to cwd since no env dir or project root

    assert result == Path(temp_dir)
    assert not config_path.exists()


# Output directory tests
def test_get_output_dir_from_env(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test output directory retrieval from environment variable.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_dir: Fixture providing temporary directory path
    """
    output_path = temp_dir / "custom_output"
    monkeypatch.setenv(OUTPUT_DIR_ENV_VAR, str(output_path))

    result = get_output_dir()

    assert result == output_path
    assert result.exists()
    assert result.is_dir()


def test_get_output_dir_from_project_root(temp_project_root: Path) -> None:
    """Test output directory retrieval from project root.

    Args:
        temp_project_root: Fixture providing temporary project structure
    """
    expected_output = temp_project_root / DEFAULT_OUTPUT_DIR

    result = get_output_dir()

    assert result == expected_output
    assert result.exists()
    assert result.is_dir()


def test_get_output_dir_fallback(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test output directory fallback to current directory.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_dir: Temporary directory to use as current working directory
    """
    monkeypatch.delenv(OUTPUT_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(temp_dir)

    result = get_output_dir()

    assert result == temp_dir


def test_get_output_dir_precedence(
    monkeypatch: MonkeyPatch, temp_project_root: Path, temp_dir: Path
) -> None:
    """Test output directory source precedence.

    Verifies that environment variable takes precedence over project root.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_project_root: Fixture providing temporary project structure
        temp_dir: Fixture providing temporary directory
    """
    env_output = temp_dir / "env_output"
    monkeypatch.setenv(OUTPUT_DIR_ENV_VAR, str(env_output))

    result = get_output_dir()

    assert result == env_output
    assert result != temp_project_root / DEFAULT_OUTPUT_DIR


def test_get_output_dir_creates_dirs(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test that output directory is created when it doesn't exist.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_dir: Fixture providing temporary directory path
    """
    output_path = temp_dir / "new_output_dir"
    monkeypatch.setenv(OUTPUT_DIR_ENV_VAR, str(output_path))

    result = get_output_dir()

    assert result == output_path
    assert output_path.exists()
    assert output_path.is_dir()


@pytest.mark.parametrize("dir_exists", [True, False])
def test_directory_creation(
    monkeypatch: MonkeyPatch, temp_dir: Path, dir_exists: bool
) -> None:
    """Test directory creation behavior for both config and output paths.

    Args:
        monkeypatch: Fixture for modifying environment variables
        temp_dir: Fixture providing temporary directory
        dir_exists: Parameter indicating if directory should exist before test
    """
    config_path = temp_dir / "test_config"
    output_path = temp_dir / "test_output"

    if dir_exists:
        config_path.mkdir(parents=True)
        output_path.mkdir(parents=True)

    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(config_path))
    monkeypatch.setenv(OUTPUT_DIR_ENV_VAR, str(output_path))

    assert get_config_dir().exists()
    assert get_output_dir().exists()
