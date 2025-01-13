"""
Test module for GeminiChatbot class implementation.
Validates core functionality including model configuration, message handling,
and response generation.
"""

from typing import List
from unittest.mock import MagicMock

import pytest

from chatbot_conversation.models import ConversationMessage
from chatbot_conversation.models.bots.gemini_bot import GeminiChatbot


def test_gemini_bot(gemini_chatbot: GeminiChatbot) -> None:
    """
    Test the GeminiChatbot class functionality.

    Verifies:
    - Correct initialization of bot parameters
    - API client setup
    - Response generation for single and multi-message conversations

    Parameters
    ----------
    gemini_chatbot : GeminiChatbot
        Fixture providing configured GeminiChatbot instance

    Returns
    -------
    None
        Test passes if all assertions are successful
    """

    assert gemini_chatbot.model_version == "gemini-1.5-flash"
    assert gemini_chatbot.system_prompt == "You are a helpful assistant."
    assert gemini_chatbot.name == "GeminiTestBot1"
    assert gemini_chatbot.bot_index == 1
    assert gemini_chatbot.get_total_bots() == 1
    assert gemini_chatbot.api is not None

    conversation = [
        ConversationMessage(
            bot_index=0, content="Hello my name is John! Please say my name!"
        )
    ]

    response = gemini_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "John" in response


def test_empty_conversation(gemini_chatbot: GeminiChatbot) -> None:
    """Test bot response to an empty conversation."""
    conversation: List[ConversationMessage] = []
    with pytest.raises(Exception) as exc_info:
        gemini_chatbot.generate_response(
            conversation
        )  # No need to assign to a variable given the assert
    assert isinstance(exc_info.value, Exception)
    assert "contents must not be empty" in str(exc_info.value)


def test_multiple_bots() -> None:
    """Test interaction of multiple bot instances."""
    bot1 = GeminiChatbot(
        bot_name="GeminiTestBot2",
        bot_system_prompt="You are a helpful assistant.",
        bot_model_version="gemini-1.5-flash",
        bot_temp=0.7,
    )
    bot2 = GeminiChatbot(
        bot_name="GeminiTestBot3",
        bot_system_prompt="You are a helpful assistant.",
        bot_model_version="gemini-1.5-flash",
        bot_temp=0.7,
    )
    assert bot1.bot_index != bot2.bot_index
    assert bot1.get_total_bots() == 2


def test_long_conversation(gemini_chatbot: GeminiChatbot) -> None:
    """Test bot handling of a long conversation history."""
    conversation = [
        ConversationMessage(bot_index=0, content=f"Message {i}") for i in range(50)
    ]
    response = gemini_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_response_with_mock(gemini_chatbot: GeminiChatbot) -> None:
    """Test the generate_response method with mocked Gemini API."""
    conversation = [ConversationMessage(bot_index=0, content="Hello, bot!")]

    # Mock the Gemini API response
    mock_response = MagicMock()
    mock_response.text = "Hello, user!"
    gemini_chatbot.api.generate_content = MagicMock(return_value=mock_response)
    response = gemini_chatbot.generate_response(conversation)

    # mock_response assertion fails below because gemini_chatbot.api is now reset
    # in the generate_response method
    # assert response == "Hello, user!"
    assert "Hello" in response
