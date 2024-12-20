"""
Test module for OpenAIChatbot class implementation.
Validates core functionality including model configuration, message handling,
and response generation.
"""

from openai import OpenAI
from src.models import OpenAIChatbot, ConversationMessage

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
    assert openai_chatbot.system_prompt == "You are in a test program and you are called OpenAITestBot - You are a helpful assistant."
    assert openai_chatbot.name == "OpenAITestBot"
    assert openai_chatbot.bot_index == 1
    assert openai_chatbot.get_total_bots() == 1
    assert openai_chatbot.api is not None
    assert isinstance(openai_chatbot.api, OpenAI)

    conversation = [
        ConversationMessage(bot_index=0, content="Hello, bot!")
    ]

    response = openai_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

    conversation.append(ConversationMessage(bot_index=1, content=response))

    response = openai_chatbot.generate_response(conversation)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0