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

from unittest.mock import Mock, patch

import pytest

from chatbot_conversation.conversation.manager import ConversationManager
from chatbot_conversation.models import ChatbotBase, ConversationMessage


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


def test_add_bot(test_config_path: str, mock_bot: ChatbotBase) -> None:
    """
    Test the addition of a bot to the conversation manager.

    Args:
        test_config_path (str): Path to test configuration file.
        mock_bot (ChatbotBase): Mock bot instance for testing.

    Verifies:
        - Bot count increases after addition
        - Added bot is correctly stored in manager
    """
    manager = ConversationManager(test_config_path)
    initial_bot_count = len(manager.bots)
    manager.add_bot(mock_bot)
    assert len(manager.bots) == initial_bot_count + 1
    assert manager.bots[-1] == mock_bot


def test_system_prompt_modifications(
    test_config_path: str, mock_bot: ChatbotBase
) -> None:
    """
    Test system prompt modification methods.

    Args:
        test_config_path (str): Path to test configuration file.
        mock_bot (ChatbotBase): Mock bot instance for testing.

    Verifies:
        - System prompt suffix addition
        - System prompt suffix removal
        - Prompt state after modifications
    """
    manager = ConversationManager(test_config_path)

    # Test adding suffix
    test_suffix = " TEST SUFFIX"
    initial_prompt = mock_bot.system_prompt
    manager.system_prompt_add_suffix(mock_bot, test_suffix)
    assert mock_bot.system_prompt == initial_prompt + test_suffix

    # Test removing suffix
    manager.system_prompt_remove_suffix(mock_bot, test_suffix)
    assert mock_bot.system_prompt == initial_prompt


@pytest.mark.parametrize(
    "text,bot_name,expected",
    [
        ("Hello {bot_name}!", "TestBot", "Hello TestBot!"),
        ("{bot_name} says hi", "Bot2", "Bot2 says hi"),
        ("No placeholder text", "Bot3", "No placeholder text"),
    ],
)
def test_insert_bot_name(
    test_config_path: str, text: str, bot_name: str, expected: str
) -> None:
    """
    Test bot name insertion into text templates.

    Args:
        test_config_path (str): Path to test configuration file.
        text (str): Template text with bot name placeholder.
        bot_name (str): Bot name to insert.
        expected (str): Expected result after insertion.

    Verifies:
        - Correct replacement of bot name placeholder
        - Handling of text without placeholders
        - Multiple placeholder cases
    """
    manager = ConversationManager(test_config_path)
    result = manager.insert_bot_name(text, bot_name)
    assert result == expected


@patch("chatbot_conversation.conversation.manager.os")
def test_write_conversation_to_file(
    mock_os: Mock,
    test_config_path: str,
    sample_conversation_data: list[ConversationMessage],
) -> None:
    """
    Test conversation transcript writing functionality.

    Args:
        mock_os (Mock): Mocked os module.
        test_config_path (str): Path to test configuration file.
        sample_conversation_data (list[dict[str, Any]]): Sample conversation data.

    Verifies:
        - Directory creation
        - File opening and writing
        - Proper handling of conversation data
    """
    manager = ConversationManager(test_config_path)
    manager.conversation = [
        ConversationMessage(**data) for data in sample_conversation_data
    ]

    with patch("builtins.open", create=True) as mock_open:
        manager.write_conversation_to_file("test_output")

    mock_os.makedirs.assert_called_once()
    mock_open.assert_called_once()


def test_invalid_config_loading(invalid_config_path: str) -> None:
    """
    Test manager behavior with invalid configuration.

    Args:
        invalid_config_path (str): Path to non-existent or invalid config file.

    Verifies:
        - Appropriate exception raising
        - Error handling for invalid configurations
    """
    with pytest.raises(Exception):  # Replace with specific exception if known
        ConversationManager(invalid_config_path)


@patch("chatbot_conversation.conversation.manager.Console")
def test_display_methods(mock_console: Mock, test_config_path: str) -> None:
    """
    Test console display functionality.

    Args:
        mock_console (Mock): Mocked Console class.
        test_config_path (str): Path to test configuration file.

    Verifies:
        - Text display through console
        - Proper console initialization and usage
    """
    manager = ConversationManager(test_config_path)
    test_text = "Test message"

    manager.display_text(test_text)
    mock_console.return_value.print.assert_called_once()
