"""Unit tests for the configuration loader module.

This module contains test cases for the configuration loading and validation
functionality, including testing for valid configurations, error cases, and
parameter validation. It covers template variable validation, moderator message
validation, and bot configuration validation.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from chatbot_conversation.conversation.loader import (
    ChatbotConfigData,
    ChatbotParamsOptData,
    ConversationConfig,
    ModeratorMessage,
    load_conversation_config,
)
from chatbot_conversation.utils import (
    ConfigurationException,
    ValidationException,
    handle_pydantic_validation_errors,
)


def test_load_valid_config(test_config_path: str) -> None:
    """Test loading a valid configuration file.

    Args:
        test_config_path: Path to a valid test configuration file
    """
    config: ConversationConfig = load_conversation_config(test_config_path)
    assert isinstance(config, ConversationConfig)
    assert config.author == "Brian Sentance"
    assert config.rounds == 5
    assert len(config.bots) == 3
    assert config.core_prompt is not None
    assert len(config.moderator_messages_opt) > 0


def test_load_nonexistent_config(invalid_config_path: str) -> None:
    """Test loading a nonexistent configuration file raises FileNotFoundError.

    Args:
        invalid_config_path: Path to a nonexistent configuration file
    """
    with pytest.raises(ConfigurationException):
        load_conversation_config(invalid_config_path)


def test_empty_config_validation(test_config_empty_path: str) -> None:
    """Test validation of an empty configuration file.

    Args:
        test_config_empty_path: Path to an empty configuration file
    """
    with pytest.raises(ValidationException):
        load_conversation_config(test_config_empty_path)


def test_duplicate_bot_names() -> None:
    """Test that configuration with duplicate bot names raises ValueError."""
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 1,
        "core_prompt": "Test prompt {bot_name}",
        "moderator_messages": [],
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
    }
    with pytest.raises(
        ValidationException,
        match="Duplicate bot names found in configuration: same_name",
    ):
        ConversationConfig(**config_data)


def test_template_variables_validation() -> None:
    """Test validation of template variables in core_prompt and bot_prompt."""
    # Test invalid variable in core_prompt
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 1,
        "core_prompt": "Invalid {variable} here",
        "moderator_messages": [],
        "bots": [
            {
                "bot_name": "bot1",
                "bot_prompt": "Valid prompt",
                "bot_type": "type1",
                "bot_version": "v1",
            }
        ],
    }
    with pytest.raises(
        ValidationException, match="Invalid template variables found: {'variable'}"
    ):
        ConversationConfig(**config_data)

    # Test invalid variable in bot_prompt
    config_data["core_prompt"] = "Valid {bot_name} prompt"
    config_data["bots"][0]["bot_prompt"] = "Invalid {unknown_var} here"
    with pytest.raises(
        ValidationException,
        match="Invalid template variables in bot_prompt: {'unknown_var'}",
    ):
        ConversationConfig(**config_data)


def test_optional_moderator_messages() -> None:
    """Test configuration with and without moderator messages."""
    # Test with no moderator messages
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 2,
        "core_prompt": "Test {bot_name}",
        "bots": [
            {
                "bot_name": "bot1",
                "bot_prompt": "Valid prompt",
                "bot_type": "type1",
                "bot_version": "v1",
            }
        ],
    }
    config = ConversationConfig(**config_data)
    assert len(config.moderator_messages_opt) == 0

    # Test with moderator messages
    config_data["moderator_messages_opt"] = [
        {"round_number": 1, "content": "Message 1"},
        {"round_number": 2, "content": "Message 2"},
    ]
    config = ConversationConfig(**config_data)
    assert len(config.moderator_messages_opt) == 2


def test_moderator_messages_validation() -> None:
    """Test validation of moderator messages configuration."""
    # Test duplicate round numbers
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 2,
        "core_prompt": "Test {bot_name}",
        "moderator_messages_opt": [
            {"round_number": 1, "content": "Message 1"},
            {"round_number": 1, "content": "Message 2"},  # Duplicate round
        ],
        "bots": [
            {
                "bot_name": "bot1",
                "bot_prompt": "Valid prompt",
                "bot_type": "type1",
                "bot_version": "v1",
            }
        ],
    }
    with pytest.raises(ValidationException, match="Duplicate round numbers"):
        ConversationConfig(**config_data)

    # Test round number exceeding total rounds
    config_data["moderator_messages_opt"] = [
        {"round_number": 3, "content": "Message 1"}  # Exceeds total rounds (2)
    ]
    with pytest.raises(ValidationException, match="Round numbers exceed total rounds"):
        ConversationConfig(**config_data)


def test_moderator_message_display_opt() -> None:
    """Test moderator message display_opt configurations."""
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 2,
        "core_prompt": "Test {bot_name}",
        "moderator_messages_opt": [
            {"round_number": 1, "content": "Message 1"},  # No display_opt
            {"round_number": 2, "content": "Message 2", "display_opt": True},
        ],
        "bots": [
            {
                "bot_name": "bot1",
                "bot_prompt": "Valid prompt",
                "bot_type": "type1",
                "bot_version": "v1",
            }
        ],
    }

    config = ConversationConfig(**config_data)
    assert len(config.moderator_messages_opt) == 2
    assert config.moderator_messages_opt[0].display_opt is False  # Default value
    assert config.moderator_messages_opt[1].display_opt is True  # Explicit value


def test_bot_name_format_validation() -> None:
    """Test validation of bot name format."""
    invalid_names: List[str] = ["_invalid", "invalid_", "invalid@name"]

    for name in invalid_names:
        config_data: Dict[str, Any] = {
            "author": "Test Author",
            "conversation_seed": "Test seed",
            "rounds": 1,
            "core_prompt": "Test {bot_name}",
            "moderator_messages": [],
            "bots": [
                {
                    "bot_name": name,
                    "bot_prompt": "Valid prompt",
                    "bot_type": "type1",
                    "bot_version": "v1",
                }
            ],
        }
        with pytest.raises(ValidationException, match="Invalid bot names"):
            ConversationConfig(**config_data)


def test_invalid_temperature() -> None:
    """Test that invalid temperature values in bot parameters raise ValueError."""
    params: Dict[str, Any] = {"temperature": 2.5, "max_tokens": 100}  # Should be <= 2.0
    with pytest.raises(ValueError):
        ChatbotParamsOptData(**params)


def test_invalid_max_tokens() -> None:
    """Test validation of invalid max_tokens and empty bot name values."""
    params: Dict[str, Any] = {"temperature": 1.0, "max_tokens": -100}  # Should be > 0
    with pytest.raises(
        ValidationException,
        match="Validation failed: max_tokens: Input should be greater than 0",
    ):
        handle_pydantic_validation_errors(ChatbotParamsOptData)(**params)


def test_valid_bot_params() -> None:
    """Test creation of valid bot parameters."""
    params: ChatbotParamsOptData = ChatbotParamsOptData(temperature=0.5, max_tokens=100)
    assert params.temperature == 0.5
    assert params.max_tokens == 100


def test_invalid_json_format(tmp_path: Path) -> None:
    """Test handling of malformed JSON configuration files.

    Args:
        tmp_path: Temporary directory path for creating test files
    """
    invalid_json_path: Path = tmp_path / "invalid.json"
    with open(invalid_json_path, "w", encoding="utf-8") as f:
        f.write("{invalid json")

    with pytest.raises(
        ConfigurationException, match="Invalid JSON in configuration file:"
    ):
        load_conversation_config(str(invalid_json_path))


def test_zero_rounds() -> None:
    """Test that configuration with zero rounds raises ValueError."""
    config_data: Dict[str, Any] = {
        "author": "Test Author",
        "conversation_seed": "Test seed",
        "rounds": 0,  # Should be > 0
        "core_prompt": "Test {bot_name}",
        "moderator_messages": [],
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
        "core_prompt": "Test {bot_name}",
        "moderator_messages": [],
        "bots": [],  # Should not be empty
    }
    with pytest.raises(ValueError):
        ConversationConfig(**config_data)


def test_reject_unknown_fields() -> None:
    """Test that unknown fields in configurations are rejected.

    Tests the 'extra=forbid' setting in BaseConfigModel.
    """
    unknown_bot_params: Dict[str, Any] = {"temperature": 1.0, "unknown_field": "value"}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ChatbotParamsOptData(**unknown_bot_params)

    # Test ModeratorMessage unknown field rejection
    unknown_moderator_msg: Dict[str, Any] = {
        "round_number": 1,
        "content": "test message",
        "unknown_field": "should fail",
    }
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ModeratorMessage(**unknown_moderator_msg)

    unknown_bot_config: Dict[str, Any] = {
        "bot_name": "test_bot",
        "bot_prompt": "test",
        "bot_type": "test",
        "bot_version": "v1",
        "invalid_field": True,
    }
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ChatbotConfigData(**unknown_bot_config)


def test_string_whitespace_stripping() -> None:
    """Test that whitespace is stripped from string fields.

    Tests the 'str_strip_whitespace=True' setting in BaseConfigModel.
    """
    bot: ChatbotConfigData = ChatbotConfigData(
        bot_name="  test_bot  ",
        bot_prompt=" test prompt ",
        bot_type=" test ",
        bot_version=" v1 ",
    )

    assert bot.bot_name == "test_bot"
    assert bot.bot_prompt == "test prompt"
    assert bot.bot_type == "test"
    assert bot.bot_version == "v1"


def test_config_immutability() -> None:
    """Test that configuration objects are immutable after creation.

    Tests the 'frozen=True' setting in BaseConfigModel.
    """
    # Test ChatbotParamsOptData immutability
    bot_params: ChatbotParamsOptData = ChatbotParamsOptData(
        temperature=1.0, max_tokens=100
    )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        bot_params.temperature = 0.5

    # Test ModeratorMessage immutability
    moderator_msg: ModeratorMessage = ModeratorMessage(round_number=1, content="test")
    with pytest.raises(ValidationError, match="Instance is frozen"):
        moderator_msg.content = "new content"

    # Test ChatbotConfigData immutability
    bot: ChatbotConfigData = ChatbotConfigData(
        bot_name="test_bot", bot_prompt="test", bot_type="test", bot_version="v1"
    )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        bot.bot_name = "new_name"

    # Test ConversationConfig immutability
    config: ConversationConfig = ConversationConfig(
        author="Test Author",
        conversation_seed="Test seed",
        rounds=1,
        core_prompt="Test prompt",
        bots=[bot],
    )
    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.author = "New Author"
