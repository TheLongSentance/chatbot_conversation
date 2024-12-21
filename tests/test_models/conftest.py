import pytest
from src.models import OpenAIChatbot, ClaudeChatbot, OllamaChatbot, GeminiChatbot
from src.models.factory import ChatbotFactory

@pytest.fixture
def openai_chatbot():
    """Fixture to create an instance of OpenAIChatbot."""
    return OpenAIChatbot(
        bot_model_version="gpt-4o-mini",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OpenAITestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )

@pytest.fixture
def claude_chatbot():
    """Fixture to create an instance of ClaudeChatbot."""
    return ClaudeChatbot(
        bot_model_version="claude-3-haiku-20240307",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )

@pytest.fixture
def ollama_chatbot():
    """Fixture to create an instance of OllamaChatbot."""
    return OllamaChatbot(
        bot_model_version="llama3.2",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OllamaTestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )

@pytest.fixture
def gemini_chatbot():
    """Fixture to create an instance of GeminiChatbot."""
    return GeminiChatbot(
        bot_model_version="gemini-1.5-flash",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="GeminiTestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )

@pytest.fixture
def chatbot_factory():
    """Fixture to create an instance of ChatbotFactory."""
    return ChatbotFactory()
