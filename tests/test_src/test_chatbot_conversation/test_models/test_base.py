"""Tests for ChatbotBase class"""

import json
from typing import List, cast
from unittest.mock import MagicMock

import pytest
import tenacity
from pytest_mock import MockFixture

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
    ChatMessage,
    ConversationMessage,
    _Model,
)

# For now, only DummyChatbot is used for testing base class
from chatbot_conversation.models.bots.dummy_bot import DummyChatbot
from chatbot_conversation.utils import APIException, ValidationException

# List of bot classes to test - for the moment, only DummyChatbot
bot_classes = [DummyChatbot]


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotConfig:
    """Test basic configuration of chatbot fixtures"""

    def test_chatbot_config(self, bot_class: type[ChatbotBase]) -> None:
        """Test that chatbot fixture has correct configuration"""
        config = ChatbotConfig(
            name="TestBot",
            system_prompt="You are a helpful assistant.",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="tpg-o4-mini",
                params_opt=ChatbotParamsOpt(temperature=0.7, max_tokens=100),
            ),
        )
        bot: ChatbotBase = bot_class(config)
        assert bot.name == "TestBot"
        assert bot.system_prompt == "You are a helpful assistant."
        assert bot.model_type == bot_class.__name__.replace("Chatbot", "").upper()
        assert bot.model_version == "tpg-o4-mini"
        assert bot.model_temperature == 0.7
        assert bot.model_max_tokens == 100
        assert bot.model_timeout == config.timeout
        assert bot.bot_index == ChatbotBase.get_total_bots()
        assert bot._model_api is not None


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseValidation:
    """Test validation logic in ChatbotBase"""

    def test_valid_name(self, bot_class: type[ChatbotBase]) -> None:
        """Test that valid names are accepted"""
        config = ChatbotConfig(
            name="ValidNameBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="tpg-o4-mini",
            ),
        )
        bot = bot_class(config)
        assert bot.name == "ValidNameBot"

    def test_empty_name(self, bot_class: type[ChatbotBase]) -> None:
        """Test that empty names are rejected"""
        with pytest.raises(ValidationException, match="Bot name must be"):
            config = ChatbotConfig(
                name="",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="tpg-o4-mini",
                ),
            )
            bot_class(config)

    def test_whitespace_name(self, bot_class: type[ChatbotBase]) -> None:
        """Test that whitespace-only names are rejected"""
        with pytest.raises(ValidationException, match="Bot name must be"):
            config = ChatbotConfig(
                name="   ",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="tpg-o4-mini",
                ),
            )
            bot_class(config)

    def test_invalid_chars_name(self, bot_class: type[ChatbotBase]) -> None:
        """Test that names with invalid characters are rejected"""
        invalid_names = ["test!", "test@bot", "test#", "test$", "test%"]
        for name in invalid_names:
            with pytest.raises(ValidationException, match="invalid characters"):
                config = ChatbotConfig(
                    name=name,
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                    ),
                )
                bot_class(config)

    def test_invalid_underscore_usage(self, bot_class: type[ChatbotBase]) -> None:
        """Test that names with invalid underscore placement are rejected"""
        invalid_names = ["_test", "test_", "_test_"]
        for name in invalid_names:
            with pytest.raises(ValidationException, match="invalid underscore usage"):
                config = ChatbotConfig(
                    name=name,
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                    ),
                )
                bot_class(config)

    def test_valid_underscore_usage(self, bot_class: type[ChatbotBase]) -> None:
        """Test that names with valid underscore placement are accepted"""
        valid_names = ["test_underscore", "test_more_underscore"]
        for name in valid_names:
            config = ChatbotConfig(
                name=name,
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="tpg-o4-mini",
                ),
            )
            bot = bot_class(config)
            assert bot.name == name

    def test_duplicate_name(self, bot_class: type[ChatbotBase]) -> None:
        """Test that duplicate names are rejected"""
        config = ChatbotConfig(
            name="DuplicateNameBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="tpg-o4-mini",
            ),
        )
        bot_class(config)
        with pytest.raises(ValidationException, match="already in use"):
            bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseMessageFormatting:
    """Test message formatting in ChatbotBase"""

    def test_api_message_formatting(
        self,
        bot_class: type[ChatbotBase],
        basic_conversation: List[ConversationMessage],
        request: pytest.FixtureRequest,
    ) -> None:
        """Test that the bot formats messages correctly for its API"""
        config = ChatbotConfig(
            name="TestBot",
            system_prompt="You are a helpful assistant.",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="tpg-o4-mini",
                params_opt=ChatbotParamsOpt(temperature=0.7, max_tokens=100),
            ),
        )
        bot: ChatbotBase = bot_class(config)

        messages: list[ChatMessage] = bot._format_conv_for_api_util(basic_conversation)

        # Common format validation
        assert isinstance(messages, list)
        assert len(messages) > 0
        assert all("role" in msg and "content" in msg for msg in messages)

        # Log formatted messages for debugging
        formatted = json.dumps(messages, indent=2)
        bot._log_debug(f"Formatted messages:\n{formatted}")


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseTemperature:
    """Test temperature handling in ChatbotBase"""

    def test_valid_temperature(self, bot_class: type[ChatbotBase]) -> None:
        """Test valid temperature values"""
        # Use appropriate temperature range based on bot type
        if (
            bot_class.__name__ == "OllamaChatbot"
            or bot_class.__name__ == "ClaudeChatbot"
            or bot_class.__name__ == "DummyChatbot"
        ):
            valid_temps = [0.0, 0.3, 0.7, 1.0]  # range 0.0-1.0
        else:
            valid_temps = [0.0, 0.7, 1.0, 1.5, 2.0]  # Standard range 0.0-2.0

        for temp in valid_temps:
            config = ChatbotConfig(
                name=f"TempBot{str(temp).replace('.', '_')}",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="tpg-o4-mini",
                    params_opt=ChatbotParamsOpt(temperature=temp),
                ),
            )
            bot = bot_class(config)
            assert bot.model_temperature == temp

    def test_invalid_temperature(self, bot_class: type[ChatbotBase]) -> None:
        """Test that invalid temperatures are rejected"""
        # Use appropriate invalid temperatures based on bot type
        if (
            bot_class.__name__ == "OllamaChatbot"
            or bot_class.__name__ == "ClaudeChatbot"
            or bot_class.__name__ == "DummyChatbot"
        ):
            invalid_temps = [-0.1, 1.1]  # Outside Ollama range
        else:
            invalid_temps = [-0.1, 2.1]  # Outside standard range

        for temp in invalid_temps:
            with pytest.raises(ValidationException, match="(?i).*temperature.*must be between.*"):
                config = ChatbotConfig(
                    name=f"TempBot{str(temp).replace('.', '_').replace('-', 'neg')}",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                        params_opt=ChatbotParamsOpt(temperature=temp),
                    ),
                )
                bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseMaxTokens:
    """Test max tokens handling in ChatbotBase"""

    def test_valid_max_tokens(self, bot_class: type[ChatbotBase]) -> None:
        """Test valid max token values"""
        valid_tokens = [1, 50, 100, 1000]
        for tokens in valid_tokens:
            config = ChatbotConfig(
                name=f"TokenBot{tokens}",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="tpg-o4-mini",
                    params_opt=ChatbotParamsOpt(max_tokens=tokens),
                ),
            )
            bot = bot_class(config)
            assert bot.model_max_tokens == tokens

    def test_invalid_max_tokens(self, bot_class: type[ChatbotBase]) -> None:
        """Test that invalid max token values are rejected"""
        invalid_tokens = [0, -1, -100]
        for tokens in invalid_tokens:
            with pytest.raises(ValidationException, match="Max tokens.*must be greater than 0"):
                config = ChatbotConfig(
                    name=f"TokenBot{str(tokens).replace('-', 'neg')}",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                        params_opt=ChatbotParamsOpt(max_tokens=tokens),
                    ),
                )
                bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseCounter:
    """Test bot instance counting functionality"""

    def test_bot_counter(self, bot_class: type[ChatbotBase]) -> None:
        """Test that bot counter increments correctly"""
        initial_count = 0
        bots: list[ChatbotBase] = []
        for i in range(3):
            config = ChatbotConfig(
                name=f"CountBot{i}",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="tpg-o4-mini",
                ),
            )
            bot = bot_class(config)
            bots.append(bot)
            initial_count += 1
            assert bot.bot_index == initial_count
            assert bot_class.get_total_bots() == initial_count


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseModelType:
    """Test model type validation"""

    def test_invalid_model_type(self, bot_class: type[ChatbotBase]) -> None:
        """Test that invalid model types are rejected"""
        with pytest.raises(ValidationException, match="Invalid model type"):
            config = ChatbotConfig(
                name="TestBot",
                system_prompt="test",
                model=ChatbotModel(type="INVALID", version="test"),
            )
            bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestModelValidation:
    """Test validation logic in _Model"""

    def test_empty_model_type(self, bot_class: type[ChatbotBase]) -> None:
        """Test that empty model types are rejected"""
        with pytest.raises(ValidationException, match="Model type cannot be empty"):
            _Model(
                type="",
                version="1.0",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_whitespace_model_type(self, bot_class: type[ChatbotBase]) -> None:
        """Test that whitespace-only model types are rejected"""
        with pytest.raises(ValidationException, match="Model type cannot be empty"):
            _Model(
                type="   ",
                version="1.0",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_empty_model_version(self, bot_class: type[ChatbotBase]) -> None:
        """Test that empty model versions are rejected"""
        with pytest.raises(ValidationException, match="Model version cannot be empty"):
            _Model(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_whitespace_model_version(self, bot_class: type[ChatbotBase]) -> None:
        """Test that whitespace-only model versions are rejected"""
        with pytest.raises(ValidationException, match="Model version cannot be empty"):
            _Model(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="   ",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="tpg-o4-mini",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )


@pytest.mark.parametrize("bot_class", bot_classes)
class TestRetryBehavior:
    """Test retry mechanism behavior"""

    def test_retry_on_transient_error(
        self, bot_class: type[ChatbotBase], mocker: MockFixture
    ) -> None:
        """Test that transient errors trigger retries"""
        config = ChatbotConfig(
            name="RetryBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="tpg-o4-mini",
            ),
        )
        bot = bot_class(config)

        error = ConnectionError("Transient error")  # fallback for DummyBot

        # Use patch with proper typing
        mock_generate = mocker.patch.object(
            bot,
            "_generate_response",
            side_effect=[error, error, "Success response"],
            autospec=True,  # This helps with type checking
        )
        mock_generate = cast(MagicMock, mock_generate)  # Explicitly cast to MagicMock

        conversation: list[ConversationMessage] = [{"bot_index": 0, "content": "test message"}]
        response = bot.generate_response(conversation)

        assert response == "Success response"
        assert mock_generate.call_count == 3

    def test_no_retry_on_permanent_error(
        self, bot_class: type[ChatbotBase], mocker: MockFixture
    ) -> None:
        """Test that permanent errors don't trigger retries"""
        config = ChatbotConfig(
            name="NoRetryBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="tpg-o4-mini",
            ),
        )
        bot = bot_class(config)

        # Mock _generate_response with proper typing
        mocker.patch.object(
            bot,
            "_generate_response",
            side_effect=ValueError("Permanent error"),
            autospec=True,
        )

        conversation: list[ConversationMessage] = [{"bot_index": 0, "content": "test message"}]
        with pytest.raises(APIException):
            bot.generate_response(conversation)

    def test_max_retries_exceeded(self, bot_class: type[ChatbotBase], mocker: MockFixture) -> None:
        """Test that max retries limit is enforced"""
        config = ChatbotConfig(
            name="MaxRetryBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="tpg-o4-mini",
            ),
        )
        bot = bot_class(config)

        # Mock with proper typing
        mock_generate = mocker.patch.object(
            bot,
            "_generate_response",
            side_effect=[ConnectionError("Transient error")] * (bot.model_timeout.max_retries + 1),
            autospec=True,
        )
        mock_generate = cast(MagicMock, mock_generate)

        conversation: list[ConversationMessage] = [{"bot_index": 0, "content": "test message"}]

        with pytest.raises(APIException) as exc_info:
            bot.generate_response(conversation)

        # Verify retry count matches max_retries
        assert mock_generate.call_count == bot.model_timeout.max_retries
        assert isinstance(exc_info.value.original_error, tenacity.RetryError)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseVersionValidation:
    """Test version validation in ChatbotBase"""

    def test_valid_model_version(self, bot_class: type[ChatbotBase]) -> None:
        """Test that valid model versions are accepted"""
        versions = bot_class.available_versions()
        if versions is not None:
            config = ChatbotConfig(
                name="ValidVersionBot",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version=versions[0],  # Use first available version
                ),
            )
            bot = bot_class(config)
            assert bot.model_version == versions[0]

    def test_invalid_model_version(self, bot_class: type[ChatbotBase]) -> None:
        """Test that invalid model versions are rejected"""
        versions = bot_class.available_versions()
        if versions is not None:
            with pytest.raises(ValidationException, match="Invalid model version"):
                config = ChatbotConfig(
                    name="InvalidVersionBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="invalid_version",
                    ),
                )
                bot_class(config)

    def test_none_available_versions(
        self, bot_class: type[ChatbotBase], mocker: MockFixture
    ) -> None:
        """Test that None available_versions skips version validation"""
        # Mock available_versions to return None
        mocker.patch.object(bot_class, "available_versions", return_value=None)

        config = ChatbotConfig(
            name="NoVersionValidationBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="any_version",  # Should accept any version
            ),
        )
        bot = bot_class(config)
        assert bot.model_version == "any_version"

    def test_empty_available_versions(
        self, bot_class: type[ChatbotBase], mocker: MockFixture
    ) -> None:
        """Test that empty available_versions list rejects all versions"""
        # Mock available_versions to return empty list
        mocker.patch.object(bot_class, "available_versions", return_value=[])

        with pytest.raises(ValidationException, match="Invalid model version"):
            config = ChatbotConfig(
                name="EmptyVersionsBot",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="any_version",
                ),
            )
            bot_class(config)
