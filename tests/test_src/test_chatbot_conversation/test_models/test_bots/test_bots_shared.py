"""Tests shared across concrete child classes of ChatbotBase"""

import pytest

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
)


@pytest.mark.parametrize(
    "bot_fixture", ["gpt_chatbot", "claude_chatbot", "ollama_chatbot", "gemini_chatbot"]
)
class TestSharedBotBehavior:
    """Test behavior common to all bot implementations"""

    def test_system_prompt_update(
        self, bot_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """Test that system prompt updates are handled correctly"""
        bot = request.getfixturevalue(bot_fixture)
        original_prompt = bot.system_prompt

        # Update prompt
        new_prompt = "New test prompt"
        bot.system_prompt = new_prompt

        assert bot.system_prompt == new_prompt
        assert bot.system_prompt != original_prompt
        assert bot.model_system_prompt_needs_update

        # Mark as updated
        bot.model_system_prompt_updated()
        assert not bot.model_system_prompt_needs_update


@pytest.mark.parametrize(
    "bot_class", ["GPTChatbot", "ClaudeChatbot", "OllamaChatbot", "GeminiChatbot"]
)
class TestSharedBotParameters:
    """Test parameter handling common to all bot implementations"""

    def test_temperature_initialization(
        self, bot_class: str, real_bot_classes: list[type[ChatbotBase]]
    ) -> None:
        """Test that each bot initializes with its default temperature"""
        # Find the bot class from the list
        bot_class_obj = next(bc for bc in real_bot_classes if bc.__name__ == bot_class)

        # Create instance with no temperature specified
        config = ChatbotConfig(
            name=f"TestBot_{bot_class}",
            system_prompt="Test prompt",
            model=ChatbotModel(
                type=bot_class_obj._get_class_model_type(),  # pyright: ignore[reportPrivateUsage]
                version="test-version",
            ),
        )
        bot = bot_class_obj(config)

        # Check temperature is set to default
        assert (
            bot.model_temperature
            == bot._default_temperature  # pyright: ignore[reportPrivateUsage]
        )

    def test_max_tokens_initialization(
        self, bot_class: str, real_bot_classes: list[type[ChatbotBase]]
    ) -> None:
        """Test that each bot initializes with correct max tokens"""
        # Find the bot class from the list
        bot_class_obj = next(bc for bc in real_bot_classes if bc.__name__ == bot_class)

        # Test with specified max tokens
        test_tokens = 500
        config = ChatbotConfig(
            name=f"TestBot_{bot_class}_Tokens",
            system_prompt="Test prompt",
            model=ChatbotModel(
                type=bot_class_obj._get_class_model_type(),  # pyright: ignore[reportPrivateUsage]
                version="test-version",
                params_opt=ChatbotParamsOpt(max_tokens=test_tokens),
            ),
        )
        bot = bot_class_obj(config)
        assert bot.model_max_tokens == test_tokens


@pytest.mark.live_api  # Skip these tests unless explicitly running live API tests
@pytest.mark.parametrize(
    "bot_fixture", ["gpt_chatbot", "claude_chatbot", "ollama_chatbot", "gemini_chatbot"]
)
class TestLiveAPIResponses:
    """Test actual API responses from each bot implementation"""

    async def test_live_response(
        self, bot_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        """Test that each bot can generate a real response via API"""
        bot = request.getfixturevalue(bot_fixture)
        
        prompt = "What is 2+2? Reply with just the number."
        response = await bot.get_response(prompt)
        
        assert response is not None
        assert "4" in response
