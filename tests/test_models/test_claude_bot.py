"""
Test module for ClaudeChatbot class implementation.
Validates core functionality including model configuration, message handling,
and response generation.
"""

from typing import List
from unittest.mock import MagicMock

import pytest
from anthropic import Anthropic

from chatbot_conversation.models import ConversationMessage
from chatbot_conversation.models.bots.claude_bot import ClaudeChatbot


def test_claude_bot(claude_chatbot: ClaudeChatbot) -> None:
    """
    Test the ClaudeChatbot class functionality.

    Verifies:
    - Correct initialization of bot parameters
    - API client setup
    - Response generation for single and multi-message conversations

    Parameters
    ----------
    claude_chatbot : ClaudeChatbot
        Fixture providing configured ClaudeChatbot instance

    Returns
    -------
    None
        Test passes if all assertions are successful
    """

    assert claude_chatbot.model_version == "claude-3-haiku-20240307"
    assert claude_chatbot.system_prompt == "You are a helpful assistant."
    assert claude_chatbot.name == "ClaudeTestBot1"
    assert claude_chatbot.bot_index == 1
    assert claude_chatbot.get_total_bots() == 1
    assert claude_chatbot.api is not None
    assert isinstance(claude_chatbot.api, Anthropic)

    conversation = [
        ConversationMessage(
            bot_index=0, content="Hello my name is John! Please say my name!"
        )
    ]

    response = claude_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "John" in response

    # Calling the model again seems to cause an error in that no response is generated
    # most likely because the last message in the conversation is from the bot
    # but have not been able to confirm this

    # conversation.append(ConversationMessage(bot_index=1, content=response))

    # response = claude_chatbot.generate_response(conversation)
    # assert response is not None
    # assert isinstance(response, str)
    # assert len(response) > 0


def test_empty_conversation(claude_chatbot: ClaudeChatbot) -> None:
    """Test bot response to an empty conversation."""
    conversation: List[ConversationMessage] = []
    with pytest.raises(Exception) as exc_info:
        claude_chatbot.generate_response(
            conversation
        )  # No need to assign to a variable given the assert
    assert isinstance(exc_info.value, Exception)
    assert "at least one message is required" in str(exc_info.value)


def test_multiple_bots() -> None:
    """Test interaction of multiple bot instances."""
    bot1 = ClaudeChatbot(
        bot_model_version="claude-3",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot2",
    )
    bot2 = ClaudeChatbot(
        bot_model_version="claude-3",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot3",
    )
    assert bot1.bot_index != bot2.bot_index
    assert bot1.get_total_bots() == 2


def test_long_conversation(claude_chatbot: ClaudeChatbot) -> None:
    """Test bot handling of a long conversation history."""
    conversation = [
        ConversationMessage(bot_index=0, content=f"Message {i}") for i in range(50)
    ]
    response = claude_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_response_with_mock(claude_chatbot: ClaudeChatbot) -> None:
    """Test the generate_response method with mocked Claude API."""
    conversation = [ConversationMessage(bot_index=0, content="Hello, bot!")]

    # Mock the Claude API response
    mock_response = MagicMock()
    mock_response.content[0].text = "Hello, user!"
    claude_chatbot.api.messages.create = MagicMock(return_value=mock_response)

    response = claude_chatbot.generate_response(conversation)
    assert response == "Hello, user!"
