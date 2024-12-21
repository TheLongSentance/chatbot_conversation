"""
Test module for ClaudeChatbot class implementation.
Validates core functionality including model configuration, message handling,
and response generation.
"""

from typing import List

from anthropic import Anthropic

from src.models import ClaudeChatbot, ConversationMessage


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
    assert (
        claude_chatbot.system_prompt
        == "You are in a test program and you are called ClaudeTestBot1 - You are a helpful assistant."
    )
    assert claude_chatbot.name == "ClaudeTestBot1"
    assert claude_chatbot.bot_index == 1
    assert claude_chatbot.get_total_bots() == 1
    assert claude_chatbot.api is not None
    assert isinstance(claude_chatbot.api, Anthropic)

    conversation = [ConversationMessage(bot_index=0, content="Hello, bot!")]

    response = claude_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

    conversation.append(ConversationMessage(bot_index=1, content=response))

    response = claude_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_empty_conversation(claude_chatbot: ClaudeChatbot) -> None:
    """Test bot response to an empty conversation."""
    conversation: List[ConversationMessage] = []
    response = claude_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_multiple_bots() -> None:
    """Test interaction of multiple bot instances."""
    bot1 = ClaudeChatbot(
        bot_model_version="claude-3",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot2",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
    bot2 = ClaudeChatbot(
        bot_model_version="claude-3",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot3",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
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
