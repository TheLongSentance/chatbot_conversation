"""Tests shared across concrete child classes of ChatbotBase"""

import time
from typing import Optional

import pytest

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
    ConversationMessage,
)


@pytest.mark.parametrize(
    "bot_class, model_version",
    [
        ("GPTChatbot", "gpt-4o-mini"),
        ("ClaudeChatbot", "claude-3-haiku-20240307"),
        ("OllamaChatbot", "llama3.2"),
        ("GeminiChatbot", "gemini-1.5-flash"),
    ],
)
class TestSharedBotParameters:
    """Test parameter handling common to all bot implementations"""

    def test_temperature_initialization(
        self,
        bot_class: str,
        model_version: str,
        real_bot_classes: list[type[ChatbotBase]],
    ) -> None:
        """Test that each bot initializes with its default temperature"""
        # Find the bot class from the list
        bot_class_obj = next(bc for bc in real_bot_classes if bc.__name__ == bot_class)

        # Create instance with no temperature specified
        config = ChatbotConfig(
            name=f"TestBot_{bot_class}",
            system_prompt="Test prompt",
            model=ChatbotModel(
                type=bot_class_obj._get_class_model_type(),
                version=model_version,
            ),
        )
        bot = bot_class_obj(config)

        # Check temperature is set to default
        assert bot.model_temperature == bot.model_default_temperature

    def test_max_tokens_initialization(
        self,
        bot_class: str,
        model_version: str,
        real_bot_classes: list[type[ChatbotBase]],
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
                type=bot_class_obj._get_class_model_type(),
                version=model_version,
                params_opt=ChatbotParamsOpt(max_tokens=test_tokens),
            ),
        )
        bot = bot_class_obj(config)
        assert bot.model_max_tokens == test_tokens


@pytest.mark.live_api
@pytest.mark.parametrize(
    "bot_fixture", ["gpt_chatbot", "claude_chatbot", "ollama_chatbot", "gemini_chatbot"]
)
class TestLiveAPIResponses:
    """Test actual API responses from each bot implementation"""

    def test_generate_response(self, bot_fixture: str, request: pytest.FixtureRequest) -> None:

        bot = request.getfixturevalue(bot_fixture)
        conversation = [
            ConversationMessage(bot_index=0, content="What is 2+2? Reply with just the number.")
        ]
        response = bot.generate_response(conversation)

        assert response is not None
        assert "4" in response


@pytest.mark.live_api
@pytest.mark.parametrize(
    "bot_fixture",
    [
        "gpt_chatbot",
        "claude_chatbot",
        "ollama_chatbot",
        "gemini_chatbot",
    ],
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

        assert response_chunks, "The response should not be empty"
        assert len(response_chunks) > 1, "The response should be streamed in multiple chunks"

        for chunk in response_chunks:
            print(chunk)

    def test_streaming_max_tokens(self, bot_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test that streaming responses respect various token limits."""
        # Test cases: very small, medium, large, and default (None) limits
        test_cases: list[tuple[Optional[int], str]] = [
            (50, "small"),
            (150, "medium"),
            (500, "large"),
            (None, "default"),
        ]

        system_prompt = (
            "You are a test assistant, involved in testing max token limits for your api. "
        )

        for max_tokens, size in test_cases:
            time.sleep(1)  # Try to avoid rate limiting
            # Create bot with specific token limit
            config = ChatbotConfig(
                name=f"{size.capitalize()}Limit_{bot_fixture}",
                system_prompt=system_prompt,
                model=ChatbotModel(
                    type=request.getfixturevalue(bot_fixture).model_type,
                    version=request.getfixturevalue(bot_fixture).model_version,
                    params_opt=ChatbotParamsOpt(max_tokens=max_tokens),
                ),
            )
            test_bot = type(request.getfixturevalue(bot_fixture))(config)

            # Generate response
            conversation = [
                ConversationMessage(
                    bot_index=0,
                    content=(
                        f"Write a really long response that is several pages long on the role of AI in society."
                    ),
                )
            ]

            chunks = list(test_bot.stream_response(conversation))
            response = "".join(chunks)
            word_count = len(response.split())
            estimated_tokens = int(word_count * 1.25)

            # Check against expected limit
            expected_limit = (
                max_tokens if max_tokens is not None else test_bot.model_default_max_tokens
            )
            assert (
                estimated_tokens <= expected_limit * 1.25
            ), f"Token limit exceeded for {size} test"
            assert estimated_tokens >= expected_limit * 0.75, f"Response too short for {size} test"

    def test_streaming_temperature(self, bot_fixture: str, request: pytest.FixtureRequest) -> None:
        """Test that streaming responses reflect different temperature settings."""
        # Test with different temperatures
        test_temps = [0.0, 0.5, 1.0]  # Low, medium, high temperatures
        responses_per_temp = 5  # Number of responses to generate per temperature

        # Simple prompt that should generate variable responses
        prompt = "List 5 random words. Just the words, separated by spaces."

        for temp in test_temps:
            # Create bot with specific temperature
            config = ChatbotConfig(
                name=f"Temp{temp}_{bot_fixture}".replace(".", "_"),
                system_prompt="You are a test assistant.",
                model=ChatbotModel(
                    type=request.getfixturevalue(bot_fixture).model_type,
                    version=request.getfixturevalue(bot_fixture).model_version,
                    params_opt=ChatbotParamsOpt(temperature=temp),
                ),
            )
            test_bot = type(request.getfixturevalue(bot_fixture))(config)

            # Generate multiple responses at this temperature
            responses: list[str] = []
            for _ in range(responses_per_temp):
                time.sleep(1)  # Try to avoid rate limiting
                conversation = [
                    ConversationMessage(
                        bot_index=0,
                        content=prompt,
                    )
                ]
                chunks = list(test_bot.stream_response(conversation))
                responses.append("".join(chunks).strip())

            # For very low temperature (0.0), responses should be more similar
            if temp == 0.0:
                # At least 2 responses should be identical at low temperature
                assert any(
                    responses.count(r) >= 2 for r in responses
                ), f"Expected some identical responses at temperature {temp}"

            # For high temperature (1.0), responses should be more varied
            if temp == 1.0:
                # Almost all responses should be different at higher temperature
                assert len(set(responses)) >= (
                    len(responses) - 1
                ), f"Expected almost all different responses at temperature {temp}"

            # Verify temperature was set correctly in bot
            assert test_bot.model_temperature == temp
