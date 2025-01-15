"""
Test module for GPTChatbot class implementation.
Validates core functionality including model configuration, message handling,
and response generation.
"""

from typing import List
from unittest.mock import MagicMock

from openai import OpenAI

from chatbot_conversation.models import (
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
    ConversationMessage,
)
from chatbot_conversation.models.bots.gpt_bot import GPTChatbot


def test_gpt_bot(gpt_chatbot: GPTChatbot) -> None:
    """
    Test the GPTChatbot class functionality.

    Verifies:
    - Correct initialization of bot parameters
    - API client setup
    - Response generation for single and multi-message conversations

    Parameters
    ----------
    gpt_chatbot : GPTChatbot
        Fixture providing configured GPTChatbot instance

    Returns
    -------
    None
        Test passes if all assertions are successful
    """

    assert gpt_chatbot.model_version == "gpt-4o-mini"
    assert gpt_chatbot.system_prompt == "You are a helpful assistant."
    assert gpt_chatbot.name == "GPTTestBot1"
    assert gpt_chatbot.bot_index == 1
    assert gpt_chatbot.get_total_bots() == 1
    assert gpt_chatbot.model_api is not None
    assert isinstance(gpt_chatbot.model_api, OpenAI)

    conversation = [
        ConversationMessage(
            bot_index=0, content="Hello my name is John! Please say my name!"
        )
    ]

    response = gpt_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    assert "John" in response


def test_empty_conversation(gpt_chatbot: GPTChatbot) -> None:
    """Test bot response to an empty conversation."""
    conversation: List[ConversationMessage] = []
    response = gpt_chatbot.generate_response(conversation)
    # in this case, OpenAI returns a response even if the conversation is empty
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_multiple_bots() -> None:
    """Test interaction of multiple bot instances."""

    config1: ChatbotConfig = ChatbotConfig(
        name="TestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="GPT",
            version="gpt-4o-mini",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    config2: ChatbotConfig = ChatbotConfig(
        name="TestBot2",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="GPT",
            version="gpt-4o-mini",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    bot1 = GPTChatbot(config1)

    bot2 = GPTChatbot(config2)

    assert bot1.bot_index != bot2.bot_index
    assert bot1.get_total_bots() == 2


def test_long_conversation(gpt_chatbot: GPTChatbot) -> None:
    """Test bot handling of a long conversation history."""
    conversation = [
        ConversationMessage(bot_index=0, content=f"Message {i}") for i in range(50)
    ]
    response = gpt_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_response_with_mock(gpt_chatbot: GPTChatbot) -> None:
    """Test the generate_response method with mocked OpenAI API."""
    conversation = [ConversationMessage(bot_index=0, content="Hello, bot!")]

    # Mock the OpenAI API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello, user!"
    gpt_chatbot.model_api.chat.completions.create = MagicMock(
        return_value=mock_response
    )

    response = gpt_chatbot.generate_response(conversation)
    assert response == "Hello, user!"
