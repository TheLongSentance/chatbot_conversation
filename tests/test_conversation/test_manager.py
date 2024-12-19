"""
This module contains tests for the ConversationManager class.

The tests handle:
- Initialization of the ConversationManager from a configuration file
- Running conversation rounds
- Handling invalid configuration paths
- Running multiple conversation rounds
- Handling empty conversation seeds in configuration
"""

import pytest

from src.conversation.manager import ERROR_EMPTY_CONVERSATION_SEED, ConversationManager


def test_conversation_manager_initialization(test_config_path: str) -> None:
    """Test initialization of ConversationManager from a configuration file.

    Args:
        test_config_path: Path to the test configuration file
        setup_valid_env: Fixture to set up environment with valid keys
    """
    manager = ConversationManager.from_config(test_config_path)
    assert manager is not None
    assert len(manager.bots) == 2  # Based on test_config.json

    # Verify initial conversation state
    assert len(manager.conversation) == 1
    assert manager.conversation[0]["bot_index"] == 0
    assert manager.conversation[0]["content"] == "This is a test conversation"

    # Verify bot configurations
    assert manager.config["rounds"] == 2
    assert len(manager.config["bots"]) == 2

    # Verify first bot configuration
    assert manager.bots[0].name == "TestBot1"
    assert (
        manager.bots[0].bot_index == 1
    )  # 1-indexed since 0 is reserved for seed message
    assert manager.bots[0].system_prompt.endswith("You are a test bot.")

    # Verify second bot configuration
    assert manager.bots[1].name == "TestBot2"
    assert manager.bots[1].bot_index == 2
    assert manager.bots[1].system_prompt.endswith("You are another test bot.")

    # Verify bot types
    assert str(manager.bots[0].__class__.__name__).startswith("OpenAI")
    assert str(manager.bots[1].__class__.__name__).startswith("Claude")


def test_run_round(test_config_path: str) -> None:
    """Test running a single round of conversation.

    Args:
        test_config_path: Path to the test configuration file
        setup_valid_env: Fixture to set up environment with valid keys
    """
    manager = ConversationManager.from_config(test_config_path)
    manager.run_round()
    assert len(manager.conversation) > 1  # Initial message + at least one response


def test_invalid_config_path() -> None:
    """Test handling of invalid configuration file path."""
    with pytest.raises(FileNotFoundError):
        ConversationManager.from_config("nonexistent.json")


def test_multiple_rounds(test_config_path: str) -> None:
    """Test running multiple rounds of conversation.

    Args:
        test_config_path: Path to the test configuration file
        setup_valid_env: Fixture to set up environment with valid keys
    """
    manager = ConversationManager.from_config(test_config_path)
    initial_length = len(manager.conversation)
    num_rounds = 3
    for _ in range(num_rounds):
        manager.run_round()
    # At least 2 messages per round
    assert len(manager.conversation) >= initial_length + num_rounds * 2


def test_invalid_config_empty_seed(test_config_empty_path: str) -> None:
    """Test handling of empty conversation seed in configuration.

    Args:
        test_config_empty_path: Path to the empty test configuration file
    """
    with pytest.raises(ValueError, match=ERROR_EMPTY_CONVERSATION_SEED):
        ConversationManager.from_config(test_config_empty_path)
