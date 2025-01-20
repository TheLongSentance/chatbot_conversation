"""Unit tests for the configuration loader module.

This module contains test cases for the configuration loading and validation
functionality, including testing for valid configurations, error cases, and
parameter validation.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from chatbot_conversation.conversation.loader import (
    ChatbotConfigData,
    ChatbotParamsOptData,
    ConfigurationLoader,
    ConversationConfig,
)


def test_load_valid_config(test_config_path: str) -> None:
    """Test loading a valid configuration file.

    Args:
        test_config_path: Path to a valid test configuration file.
    """
    config: ConversationConfig = ConfigurationLoader.load_config(test_config_path)
    assert isinstance(config, ConversationConfig)
    assert config.author == "Brian Sentance"
    assert config.rounds == 3
    assert len(config.bots) == 3


def test_load_nonexistent_config(invalid_config_path: str) -> None:
    """Test loading a nonexistent configuration file raises FileNotFoundError.

    Args:
        invalid_config_path: Path to a nonexistent configuration file.
    """
    with pytest.raises(FileNotFoundError):
        ConfigurationLoader.load_config(invalid_config_path)


def test_empty_config_validation(test_config_empty_path: str) -> None:
    """Test validation of an empty configuration file.

    Args:
        test_config_empty_path: Path to an empty configuration file.
    """
    with pytest.raises(ValueError):
        ConfigurationLoader.load_config(test_config_empty_path)


def test_duplicate_bot_names() -> None:
    """Test that configuration with duplicate bot names raises ValueError."""
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 1,
        "shared_prefix": "",
        "first_round_postfix": "",
        "last_round_postfix": "",
        "bots": [
            {
                "bot_name": "same_name",
                "bot_prompt": "prompt1",
                "bot_type": "type1",
                "bot_version": "v1",
            },
            {
                "bot_name": "same_name",
                "bot_prompt": "prompt2",
                "bot_type": "type2",
                "bot_version": "v2",
            },
        ],
        "bot_params_opt": ChatbotParamsOptData(temperature=0.5, max_tokens=100),
    }
    with pytest.raises(ValueError, match="Duplicate bot names"):
        ConversationConfig(**config_data)


def test_invalid_temperature() -> None:
    """Test that invalid temperature values in bot parameters raise ValueError."""
    params: Dict[str, Any] = {"temperature": 2.5, "max_tokens": 100}  # Should be <= 2.0
    with pytest.raises(ValueError):
        ChatbotParamsOptData(**params)


def test_invalid_max_tokens() -> None:
    """Test validation of invalid max_tokens and empty bot name values."""
    params: Dict[str, Any] = {"temperature": 1.0, "max_tokens": -100}  # Should be > 0
    with pytest.raises(ValueError):
        ChatbotParamsOptData(**params)

    bot_config: Dict[str, Any] = {
        "bot_name": "",  # Should not be empty
        "bot_prompt": "test prompt",
        "bot_type": "test type",
        "bot_version": "v1",
    }
    with pytest.raises(ValueError):
        ChatbotConfigData(**bot_config)


def test_valid_bot_params() -> None:
    """Test creation of valid bot parameters."""
    params: ChatbotParamsOptData = ChatbotParamsOptData(temperature=0.5, max_tokens=100)
    assert params.temperature == 0.5
    assert params.max_tokens == 100


def test_invalid_json_format(tmp_path: Path) -> None:
    """Test handling of malformed JSON configuration files.

    Args:
        tmp_path: Temporary directory path for creating test files.
    """
    invalid_json_path: Path = tmp_path / "invalid.json"
    with open(invalid_json_path, "w", encoding="utf-8") as f:
        f.write("{invalid json")

    with pytest.raises(json.JSONDecodeError):
        ConfigurationLoader.load_config(str(invalid_json_path))


def test_zero_rounds() -> None:
    """Test that configuration with zero rounds raises ValueError."""
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 0,  # Should be > 0
        "shared_prefix": "",
        "first_round_postfix": "",
        "last_round_postfix": "",
        "bots": [
            {
                "bot_name": "bot1",
                "bot_prompt": "prompt1",
                "bot_type": "type1",
                "bot_version": "v1",
            }
        ],
    }
    with pytest.raises(ValueError):
        ConversationConfig(**config_data)


def test_empty_bots_list() -> None:
    """Test that configuration with empty bots list raises ValueError."""
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 1,
        "shared_prefix": "",
        "first_round_postfix": "",
        "last_round_postfix": "",
        "bots": [],  # Should not be empty
    }
    with pytest.raises(ValueError):
        ConversationConfig(**config_data)
