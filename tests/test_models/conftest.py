import pytest
from src.models import OpenAIChatbot


@pytest.fixture
def openai_chatbot():
    """Fixture to create an instance of OpenAIChatbot."""
    return OpenAIChatbot(
        bot_model_version="gpt-4o-mini",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OpenAITestBot",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
