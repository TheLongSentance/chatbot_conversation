"""Tests shared across concrete child classes of ChatbotBase"""

import pytest

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
    ConversationMessage,
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


@pytest.mark.parametrize(
    "bot_fixture", ["gpt_chatbot", "claude_chatbot", "ollama_chatbot", "gemini_chatbot"]
)
class TestLiveAPIResponses:
    """Test actual API responses from each bot implementation"""

    def test_generate_response(self, bot_fixture: str, request: pytest.FixtureRequest) -> None:

        bot = request.getfixturevalue(bot_fixture)
        conversation = [
            ConversationMessage(
                bot_index=0, content="What is 2+2? Reply with just the number."
            )
        ]
        response = bot.generate_response(conversation)

        assert response is not None
        assert "4" in response


@pytest.mark.live_api  # Skip these tests unless explicitly running live API tests
@pytest.mark.parametrize(
    "bot_fixture", ["gpt_chatbot", "claude_chatbot", "ollama_chatbot", "gemini_chatbot", "dummy_chatbot"]
)
class TestLiveAPIStreamingResponses:
    """Test actual API streaming responses from each bot implementation"""

    def test_generate_streaming_response(
        self, bot_fixture: str, request: pytest.FixtureRequest
    ) -> None:
        bot = request.getfixturevalue(bot_fixture)
        conversation = [
            ConversationMessage(
                bot_index=0,
                content="Give me a 100 token response on any subject, please.",
            )
        ]

        response_chunks = list(bot.stream_response(conversation))

        # Ensure response is not empty
        assert response_chunks, "The response should not be empty"

        # Ensure response is streamed in chunks
        assert (
            len(response_chunks) > 1
        ), "The response should be streamed in multiple chunks"

        # Print each chunk (for debugging purposes)
        for chunk in response_chunks:
            print(chunk)
