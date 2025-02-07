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

from pathlib import Path

import pytest

from chatbot_conversation.conversation.manager import ConversationManager
from chatbot_conversation.utils import ConfigurationException


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
    manager = ConversationManager(Path(test_config_path))
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
    with pytest.raises(ConfigurationException):
        ConversationManager(Path(invalid_config_path))


def test_clean_truncated_response(manager: ConversationManager) -> None:
    """
    Test the clean_truncated_response method for various edge cases.

    Args:
        manager (ConversationManager): Instance of ConversationManager.

    Verifies:
        - Proper truncation at the last complete sentence.
        - Handling of ellipses.
        - Leaving valid responses unchanged.
    """
    # Test case: valid response with complete sentences
    response = "This is a complete sentence. This is another one."
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == response

    # Test case: response ending with an incomplete sentence
    response = "This is a complete sentence. This is another one. Incomplete"
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == "This is a complete sentence. This is another one."

    # Test case: response with ellipses
    response = "This is a complete sentence... But this is incomplete"
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == "This is a complete sentence..."

    # Test case: response with question marks and exclamation marks
    response = "Is this a question? Yes! It is."
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == response

    # Test case: response with question marks and exclamation marks
    response = "Is this a question? Yes!"
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == response

    # Test case: response with question marks and exclamation marks
    response = "Is this a question?"
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == response

    # Test case: response with no complete sentences
    response = "Incomplete sentence without ending"
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == response

    # Test case: response with one complete sentence
    response = "Complete sentence! Incomplete sentence without ending"
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == "Complete sentence!"

    # Test case: response with multiple sentence endings
    response = "First sentence. Second sentence! Third sentence?"
    cleaned_response = manager.clean_truncated_response(response)
    assert cleaned_response == response
