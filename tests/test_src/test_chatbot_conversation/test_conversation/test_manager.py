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
from typing import List

import pytest

from chatbot_conversation.conversation.manager import ConversationManager
from chatbot_conversation.utils import ConfigurationException
from chatbot_conversation.models import ConversationMessage


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


def test_filter_private_content(
    manager: ConversationManager, sample_private_messages: List[ConversationMessage]
) -> None:
    """
    Test private content filtering for individual messages.

    Args:
        manager (ConversationManager): Instance of ConversationManager
        sample_private_messages (List[ConversationMessage]): Sample messages with private content

    Verifies:
        - Private content is preserved for matching bot index
        - Private content is removed for non-matching bot index
        - Messages without private content remain unchanged
        - Private content is removed when no bot index is specified
    """
    # Test message with no private content
    public_msg = sample_private_messages[0]
    assert manager.filter_private_content(public_msg) == "Public message from bot 1"
    assert manager.filter_private_content(public_msg, 1) == "Public message from bot 1"

    # Test message with private content for matching bot
    private_msg = sample_private_messages[1]
    assert (
        manager.filter_private_content(private_msg, 1)
        == "Public part PR1V4T3: Private part for bot 1"
    )

    # Test message with private content for non-matching bot
    assert manager.filter_private_content(private_msg, 2) == "Public part"

    # Test message with private content and no bot specified
    assert manager.filter_private_content(private_msg) == "Public part"


def test_get_filtered_conversation(
    manager: ConversationManager, sample_private_messages: List[ConversationMessage]
) -> None:
    """
    Test filtering of entire conversation history.

    Args:
        manager (ConversationManager): Instance of ConversationManager
        sample_private_messages (List[ConversationMessage]): Sample messages with private content

    Verifies:
        - Conversation history is correctly filtered for specific bot
        - Bot indices are preserved
        - Private content is only included for specified bot
        - Message structure remains intact
    """
    # Replace manager's conversation with our test data
    manager.conversation = sample_private_messages

    # Get filtered conversation for bot 1
    filtered_conv = manager.get_filtered_conversation(1)

    assert len(filtered_conv) == len(sample_private_messages)

    # Check first message (public only)
    assert filtered_conv[0]["content"] == "Public message from bot 1"
    assert filtered_conv[0]["bot_index"] == 1

    # Check second message (has private content for bot 1)
    assert filtered_conv[1]["content"] == "Public part PR1V4T3: Private part for bot 1"
    assert filtered_conv[1]["bot_index"] == 1

    # Check third message (has private content for bot 2)
    assert filtered_conv[2]["content"] == "Another public message"
    assert filtered_conv[2]["bot_index"] == 2

    # Check fourth message (public only)
    assert filtered_conv[3]["content"] == "Just public content"
    assert filtered_conv[3]["bot_index"] == 3
