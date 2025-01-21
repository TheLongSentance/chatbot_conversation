from typing import Any, Dict, cast

import toml

from chatbot_conversation.version import __version__


def test_version() -> None:
    """Test that the version string is correctly defined."""
    assert __version__ == "1.0.0"


def test_version_consistency() -> None:
    """Test that the version in pyproject.toml matches the version in version.py."""
    with open("pyproject.toml", "r") as f:
        pyproject_data = cast(Dict[str, Any], toml.load(f))  # type: ignore
    toml_version = pyproject_data["project"]["version"]
    assert (
        toml_version == __version__
    ), f"Version mismatch: {toml_version} (pyproject.toml) != {__version__} (version.py)"
