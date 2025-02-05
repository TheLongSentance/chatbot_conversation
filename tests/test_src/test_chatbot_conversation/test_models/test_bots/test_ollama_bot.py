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
    OLLAMA_MAXIMUM_TEMPERATURE,
    OLLAMA_MINIMUM_TEMPERATURE,
    OLLAMA_MODEL_TYPE,
    OllamaChatbot,
)
from chatbot_conversation.utils import ModelException, ValidationException


class TestOllamaChatbot:
    """Test Ollama-specific chatbot functionality"""

    def test_model_type(self, ollama_config_for_tests: ChatbotConfig) -> None:
        """Test that Ollama model type constant is correctly used"""
        bot = OllamaChatbot(ollama_config_for_tests)
        assert bot.model_type == OLLAMA_MODEL_TYPE

    def test_temperature_bounds(self) -> None:
        """Test that temperature settings respect Ollama's specific bounds"""
        # Test initialization with default temperature
        config = ChatbotConfig(
            name="TestBot1",
            system_prompt="Test prompt",
            model=ChatbotModel(type="OLLAMA", version="llama3.2"),
        )
        bot = OllamaChatbot(config)
        assert bot.model_temperature == bot.model_default_temperature

        # Test valid temperature initialization
        config_valid = ChatbotConfig(
            name="TestBot2",
            system_prompt="Test prompt",
            model=ChatbotModel(
                type="OLLAMA",
                version="llama3.2",
                params_opt=ChatbotParamsOpt(temperature=0.5),
            ),
        )
        bot_valid = OllamaChatbot(config_valid)
        assert bot_valid.model_temperature == 0.5

        # Test invalid temperature initialization
        with pytest.raises(ValidationException, match="Temperature .* must be between"):
            invalid_config = ChatbotConfig(
                name="TestBot3",
                system_prompt="Test prompt",
                model=ChatbotModel(
                    type="OLLAMA",
                    version="llama3.2",
                    params_opt=ChatbotParamsOpt(
                        temperature=OLLAMA_MAXIMUM_TEMPERATURE + 0.1
                    ),
                ),
            )
            OllamaChatbot(invalid_config)

        # Test invalid temperature initialization
        with pytest.raises(ValidationException, match="Temperature .* must be between"):
            invalid_config = ChatbotConfig(
                name="TestBot4",
                system_prompt="Test prompt",
                model=ChatbotModel(
                    type="OLLAMA",
                    version="llama3.2",
                    params_opt=ChatbotParamsOpt(
                        temperature=OLLAMA_MINIMUM_TEMPERATURE - 0.1
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
    @patch.object(OllamaChatbot, "available_versions", return_value=["llama3.2"])
    def test_empty_response_handling(
        self,
        mock_available_versions: MagicMock,
        mock_ollama: MagicMock,
        ollama_config_for_tests: ChatbotConfig,
    ) -> None:
        """Test handling of empty responses from Ollama API"""
        mock_response = {"message": {"content": ""}}
        mock_ollama.chat.return_value = mock_response

        bot = OllamaChatbot(ollama_config_for_tests)
        conversation: list[ConversationMessage] = [{"bot_index": 1, "content": "Hello"}]

        with pytest.raises(
            ModelException, match="Model returned an empty string response"
        ):
            bot.generate_response(conversation)

    def test_available_versions_returns_valid_list(self) -> None:
        """
        Test that available_versions returns non-empty list of model versions.

        Verifies that:
        1. Method returns a list
        2. List is not empty
        3. All entries are strings
        """
        versions = OllamaChatbot.available_versions()
        assert versions is not None
        assert len(versions) > 0
        assert isinstance(versions, list)
        assert all(isinstance(v, str) for v in versions)

    def test_bot_creation_with_valid_version(
        self, ollama_config_for_tests: ChatbotConfig
    ) -> None:
        """
        Test that bot creation with valid version succeeds.

        Uses first available version from API to ensure test uses valid version.
        """
        # Use the first available version from the API
        versions = OllamaChatbot.available_versions()
        assert versions is not None
        ollama_config_for_tests.model.version = versions[0]
        bot = OllamaChatbot(ollama_config_for_tests)
        assert bot.model_version == versions[0]

    def test_bot_creation_with_invalid_version(
        self, ollama_config_for_tests: ChatbotConfig
    ) -> None:
        """
        Test that bot creation with invalid version fails.

        Uses a known invalid version string to verify error handling.
        """
        ollama_config_for_tests.model.version = "invalid-model-version"
        with pytest.raises(ValidationException, match="Invalid model version"):
            OllamaChatbot(ollama_config_for_tests)

    def test_version_caching(self) -> None:
        """
        Test that available versions are cached.

        Verifies that:
        1. Cache is initially empty
        2. First call retrieves versions
        3. Second call uses cached versions
        4. Cache contains expected values
        """
        # Clear cache first
        OllamaChatbot._available_versions_cache = None  # pyright: ignore[reportPrivateUsage]

        # First call should hit API
        versions1 = OllamaChatbot.available_versions()

        # Second call should use cache
        versions2 = OllamaChatbot.available_versions()

        assert versions1 == versions2
        assert OllamaChatbot._available_versions_cache == versions1  # pyright: ignore[reportPrivateUsage]
