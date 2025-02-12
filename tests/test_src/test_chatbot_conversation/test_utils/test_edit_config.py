"""Unit tests for the edit_config module."""

import json
from pathlib import Path
from typing import List

from chatbot_conversation.utils.edit_config import update_bot_config


def test_update_bot_config_success(temp_bot_config: str, capture_stdout: List[str]) -> None:
    """Test successful update of bot configurations."""
    new_type = "new_type"
    new_version = "new_version"
    
    update_bot_config(temp_bot_config, new_type, new_version)
    
    # Verify the changes in file
    with open(temp_bot_config, "r") as f:
        updated_config = json.load(f)
    
    for bot in updated_config["bots"]:
        assert bot["bot_type"] == new_type
        assert bot["bot_version"] == new_version
    
    # Verify success messages
    assert f"Successfully updated {temp_bot_config}" in capture_stdout
    assert f"All bots now have type: {new_type} and version: {new_version}" in capture_stdout


def test_update_bot_config_file_not_found(capture_stdout: List[str]) -> None:
    """Test handling of non-existent config file."""
    update_bot_config("nonexistent.json", "new_type", "new_version")
    assert "Error: Config file 'nonexistent.json' not found" in capture_stdout


def test_update_bot_config_invalid_json(tmp_path: Path, capture_stdout: List[str]) -> None:
    """Test handling of invalid JSON in config file."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{invalid json")
    
    update_bot_config(str(invalid_file), "new_type", "new_version")
    assert f"Error: Invalid JSON in '{str(invalid_file)}'" in capture_stdout


def test_update_bot_config_missing_bots_key(tmp_path: Path, capture_stdout: List[str]) -> None:
    """Test handling of config file missing 'bots' key."""
    invalid_config = tmp_path / "no_bots.json"
    with open(invalid_config, "w") as f:
        json.dump({"author": "test"}, f)
    
    update_bot_config(str(invalid_config), "new_type", "new_version")
    assert "Error: Missing required 'bots' key in config file" in capture_stdout
