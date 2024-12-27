"""
Test module for OpenAIChatbot class implementation.
Validates core functionality including model configuration, message handling,
and response generation.
"""

from typing import List
from unittest.mock import MagicMock

from openai import OpenAI

from chatbot_conversation.models import ConversationMessage, OpenAIChatbot


def test_openai_bot(openai_chatbot: OpenAIChatbot) -> None:
    """
    Test the OpenAIChatbot class functionality.

    Verifies:
    - Correct initialization of bot parameters
    - API client setup
    - Response generation for single and multi-message conversations

    Parameters
    ----------
    openai_chatbot : OpenAIChatbot
        Fixture providing configured OpenAIChatbot instance

    Returns
    -------
    None
        Test passes if all assertions are successful
    """

    assert openai_chatbot.model_version == "gpt-4o-mini"
    assert openai_chatbot.system_prompt == "You are a helpful assistant."
    assert openai_chatbot.name == "OpenAITestBot1"
    assert openai_chatbot.bot_index == 1
    assert openai_chatbot.get_total_bots() == 1
    assert openai_chatbot.api is not None
    assert isinstance(openai_chatbot.api, OpenAI)

    conversation = [
        ConversationMessage(
            bot_index=0, content="Hello my name is John! Please say my name!"
        )
    ]

    response = openai_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "John" in response


def test_empty_conversation(openai_chatbot: OpenAIChatbot) -> None:
    """Test bot response to an empty conversation."""
    conversation: List[ConversationMessage] = []
    response = openai_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_multiple_bots() -> None:
    """Test interaction of multiple bot instances."""
    bot1 = OpenAIChatbot(
        bot_model_version="gpt-4o-mini",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="Bot1",
    )
    bot2 = OpenAIChatbot(
        bot_model_version="gpt-4o-mini",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="Bot2",
    )
    assert bot1.bot_index != bot2.bot_index
    assert bot1.get_total_bots() == 2


def test_long_conversation(openai_chatbot: OpenAIChatbot) -> None:
    """Test bot handling of a long conversation history."""
    conversation = [
        ConversationMessage(bot_index=0, content=f"Message {i}") for i in range(50)
    ]
    response = openai_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_response_with_mock(openai_chatbot: OpenAIChatbot) -> None:
    """Test the generate_response method with mocked OpenAI API."""
    conversation = [ConversationMessage(bot_index=0, content="Hello, bot!")]

    # Mock the OpenAI API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello, user!"
    openai_chatbot.api.chat.completions.create = MagicMock(return_value=mock_response)

    response = openai_chatbot.generate_response(conversation)
    assert response == "Hello, user!"
