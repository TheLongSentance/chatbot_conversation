"""Tests specific to OllamaChatbot implementation"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from chatbot_conversation.models.base import (
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
    ConversationMessage,
)
from chatbot_conversation.models.bots.ollama_bot import (
    MODEL_TYPE,
    OLLAMA_DEFAULT_TEMP,
    OLLAMA_MAX_MODEL_TEMP,
    OLLAMA_MIN_MODEL_TEMP,
    OllamaChatbot,
)


class TestOllamaChatbot:
    """Test Ollama-specific chatbot functionality"""

    def test_model_type(self, ollama_config_for_tests: ChatbotConfig) -> None:
        """Test that Ollama model type constant is correctly used"""
        bot = OllamaChatbot(ollama_config_for_tests)
        assert bot.model_type == MODEL_TYPE

    def test_temperature_bounds(self) -> None:
        """Test that temperature settings respect Ollama's specific bounds"""
        # Test initialization with default temperature
        config = ChatbotConfig(
            name="TestBot1",
            system_prompt="Test prompt",
            model=ChatbotModel(type="OLLAMA", version="llama2"),
        )
        bot = OllamaChatbot(config)
        assert bot.model_temperature == OLLAMA_DEFAULT_TEMP

        # Test valid temperature initialization
        config_valid = ChatbotConfig(
            name="TestBot2",
            system_prompt="Test prompt",
            model=ChatbotModel(
                type="OLLAMA",
                version="llama2",
                params_opt=ChatbotParamsOpt(temperature=0.5),
            ),
        )
        bot_valid = OllamaChatbot(config_valid)
        assert bot_valid.model_temperature == 0.5

        # Test invalid temperature initialization
        with pytest.raises(ValueError, match="Temperature .* must be between"):
            invalid_config = ChatbotConfig(
                name="TestBot3",
                system_prompt="Test prompt",
                model=ChatbotModel(
                    type="OLLAMA",
                    version="llama2",
                    params_opt=ChatbotParamsOpt(
                        temperature=OLLAMA_MAX_MODEL_TEMP + 0.1
                    ),
                ),
            )
            OllamaChatbot(invalid_config)

        # Test invalid temperature initialization
        with pytest.raises(ValueError, match="Temperature .* must be between"):
            invalid_config = ChatbotConfig(
                name="TestBot4",
                system_prompt="Test prompt",
                model=ChatbotModel(
                    type="OLLAMA",
                    version="llama2",
                    params_opt=ChatbotParamsOpt(
                        temperature=OLLAMA_MIN_MODEL_TEMP - 0.1
                    ),
                ),
            )
            OllamaChatbot(invalid_config)

    @pytest.mark.parametrize(
        "exception,should_retry",
        [
            (httpx.TimeoutException("test"), True),
            (httpx.NetworkError("test"), True),
            (
                httpx.HTTPStatusError(
                    "test", request=MagicMock(), response=MagicMock()
                ),
                True,
            ),
            (ValueError("test"), False),
            (Exception("test"), False),
        ],
    )
    def test_should_retry_on_exception(
        self,
        ollama_config_for_tests: ChatbotConfig,
        exception: Exception,
        should_retry: bool,
    ) -> None:
        """Test retry logic for Ollama-specific exceptions"""
        bot = OllamaChatbot(ollama_config_for_tests)
        assert (
            bot._should_retry_on_exception(  # pyright: ignore[reportPrivateUsage]
                exception
            )
            == should_retry
        )

    @patch("chatbot_conversation.models.bots.ollama_bot.ollama")
    def test_api_call_parameters(
        self, mock_ollama: MagicMock, ollama_config_for_tests: ChatbotConfig
    ) -> None:
        """Test Ollama API call parameter formatting"""
        # Create a mock response
        mock_response = {"message": {"content": "Test response"}}
        mock_ollama.chat.return_value = mock_response

        bot = OllamaChatbot(ollama_config_for_tests)
        conversation: list[ConversationMessage] = [
            {"bot_index": 0, "content": "Test message"}
        ]

        # Generate response using the mock
        response = bot._generate_response(  # pyright: ignore[reportPrivateUsage]
            conversation
        )
        assert response == "Test response"

        # Verify chat was called with correct parameters
        chat_call = mock_ollama.chat.call_args
        assert chat_call is not None, "Chat method was not called"

        _, call_kwargs = chat_call
        assert call_kwargs["model"] == ollama_config_for_tests.model.version
        assert call_kwargs["options"]["temperature"] == bot.model_temperature
        assert call_kwargs["options"]["num_predict"] == bot.model_max_tokens
        assert isinstance(call_kwargs["messages"], list)

        # Check message formatting
        messages = call_kwargs["messages"]
        assert len(messages) == 2  # System prompt + user message
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == ollama_config_for_tests.system_prompt
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test message"

    @patch("chatbot_conversation.models.bots.ollama_bot.ollama")
    def test_empty_response_handling(
        self, mock_ollama: MagicMock, ollama_config_for_tests: ChatbotConfig
    ) -> None:
        """Test handling of empty responses from Ollama API"""
        mock_response = {"message": {"content": ""}}
        mock_ollama.chat.return_value = mock_response

        bot = OllamaChatbot(ollama_config_for_tests)
        conversation: list[ConversationMessage] = [{"bot_index": 1, "content": "Hello"}]

        with pytest.raises(ValueError, match="Model returned an empty response"):
            bot.generate_response(conversation)
