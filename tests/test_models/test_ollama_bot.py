"""
Test module for OllamaChatbot class implementation.
Validates core functionality including model configuration, message handling,
and response generation.
"""

from typing import List

from src.models import ConversationMessage, OllamaChatbot


def test_ollama_bot(ollama_chatbot: OllamaChatbot) -> None:
    """
    Test the OllamaChatbot class functionality.

    Verifies:
    - Correct initialization of bot parameters
    - API client setup
    - Response generation for single and multi-message conversations

    Parameters
    ----------
    ollama_chatbot : OllamaChatbot
        Fixture providing configured OllamaChatbot instance

    Returns
    -------
    None
        Test passes if all assertions are successful
    """

    assert ollama_chatbot.model_version == "llama3.2"
    assert (
        ollama_chatbot.system_prompt
        == "You are in a test program and you are called OllamaTestBot1 - You are a helpful assistant."
    )
    assert ollama_chatbot.name == "OllamaTestBot1"
    assert ollama_chatbot.bot_index == 1
    assert ollama_chatbot.get_total_bots() == 1
    assert ollama_chatbot.api is None  # Ollama doesn't need initialization

    conversation = [ConversationMessage(bot_index=0, content="Hello, bot!")]

    response = ollama_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

    conversation.append(ConversationMessage(bot_index=1, content=response))

    response = ollama_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_empty_conversation(ollama_chatbot: OllamaChatbot) -> None:
    """Test bot response to an empty conversation."""
    conversation: List[ConversationMessage] = []
    response = ollama_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_multiple_bots() -> None:
    """Test interaction of multiple bot instances."""
    bot1 = OllamaChatbot(
        bot_model_version="llama3.2",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OllamaTestBot2",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
    bot2 = OllamaChatbot(
        bot_model_version="llama3.2",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OllamaTestBot3",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
    assert bot1.bot_index != bot2.bot_index
    assert bot1.get_total_bots() == 2


def test_long_conversation(ollama_chatbot: OllamaChatbot) -> None:
    """Test bot handling of a long conversation history."""
    conversation = [
        ConversationMessage(bot_index=0, content=f"Message {i}") for i in range(50)
    ]
    response = ollama_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
