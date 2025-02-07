"""Unit tests for dir_util module."""

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from chatbot_conversation.utils.dir_util import (
    CONFIG_DIR_ENV_VAR,
    DEFAULT_CONFIG_DIR,
    get_config_dir,
)


def test_get_config_dir_from_env(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test get_config_dir when environment variable is set.

    Args:
        monkeypatch: PyTest fixture for modifying environment
        temp_dir: PyTest fixture providing temporary directory
    """
    config_path = temp_dir / "custom_config"
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(config_path))

    result = get_config_dir()

    assert result == config_path
    assert result.exists()
    assert result.is_dir()


def test_get_config_dir_from_project_root(temp_project_root: Path) -> None:
    """Test get_config_dir finding config dir under project root.

    Args:
        temp_project_root: Fixture providing temporary project structure
    """
    expected_config = temp_project_root / DEFAULT_CONFIG_DIR

    result = get_config_dir()

    assert result == expected_config
    assert result.exists()
    assert result.is_dir()


def test_get_config_dir_fallback(monkeypatch: MonkeyPatch, temp_dir: Path) -> None:
    """Test get_config_dir fallback to current directory.

    Args:
        monkeypatch: PyTest fixture for modifying environment
        temp_dir: Temporary directory to use as current working directory
    """
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(temp_dir)

    result = get_config_dir()

    assert result == temp_dir


def test_get_config_dir_precedence(
    monkeypatch: MonkeyPatch, temp_project_root: Path, temp_dir: Path
) -> None:
    """Test get_config_dir respects precedence of config sources.

    Args:
        monkeypatch: PyTest fixture for modifying environment
        temp_project_root: Fixture providing temporary project structure
        temp_dir: PyTest fixture providing temporary directory
    """
    # Setup env var config path
    env_config = temp_dir / "env_config"
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_config))

    result = get_config_dir()

    # Should prefer env var over project root
    assert result == env_config
    assert result != temp_project_root / DEFAULT_CONFIG_DIR
