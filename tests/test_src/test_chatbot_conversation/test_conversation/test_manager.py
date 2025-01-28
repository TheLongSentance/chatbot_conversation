"""
Test module for the ConversationManager class.

This module contains unit tests that verify the functionality of the ConversationManager class,
including initialization, bot management, system prompt modifications, and conversation handling.

Test cases cover:
- Manager initialization
- Bot addition and management
- System prompt modifications
- Text formatting and bot name insertion
- File I/O operations
- Display functionality
"""

import pytest

from chatbot_conversation.conversation.manager import ConversationManager


def test_initialization(test_config_path: str) -> None:
    """
    Test ConversationManager initialization with valid configuration.

    Args:
        test_config_path (str): Path to test configuration file.

    Verifies:
        - Manager instance is created correctly
        - Bots are initialized
        - Initial conversation state is set up properly
    """
    manager = ConversationManager(test_config_path)
    assert isinstance(manager, ConversationManager)
    assert len(manager.bots) > 0
    assert isinstance(manager.conversation[0], dict)
    assert "bot_index" in manager.conversation[0]
    assert "content" in manager.conversation[0]

def test_invalid_config_loading(invalid_config_path: str) -> None:
    """
    Test manager behavior with invalid configuration.

    Args:
        invalid_config_path (str): Path to non-existent or invalid config file.

    Verifies:
        - Appropriate exception raising
        - Error handling for invalid configurations
    """
    with pytest.raises(FileNotFoundError):
        ConversationManager(invalid_config_path)
