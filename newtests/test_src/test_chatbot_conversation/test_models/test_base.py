"""Tests for ChatbotBase class"""

import pytest

from chatbot_conversation.models import (
    ChatbotBase,
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
)
from chatbot_conversation.models.base import (
    _Model,  # pyright: ignore[reportPrivateUsage]
)
from chatbot_conversation.models.bots.claude_bot import ClaudeChatbot
from chatbot_conversation.models.bots.dummy_bot import DummyChatbot
from chatbot_conversation.models.bots.gemini_bot import GeminiChatbot
from chatbot_conversation.models.bots.gpt_bot import GPTChatbot
from chatbot_conversation.models.bots.ollama_bot import OllamaChatbot

# List of bot classes to test
bot_classes = [DummyChatbot, GPTChatbot, ClaudeChatbot, OllamaChatbot, GeminiChatbot]


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotConfig:
    """Test basic configuration of chatbot fixtures"""

    def test_chatbot_config(self, bot_class: type[ChatbotBase]):
        """Test that chatbot fixture has correct configuration"""
        config = ChatbotConfig(
            name="TestBot",
            system_prompt="You are a helpful assistant.",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="None",
                params_opt=ChatbotParamsOpt(temperature=0.7, max_tokens=100),
            ),
        )
        bot: ChatbotBase = bot_class(config)
        assert bot.name == "TestBot"
        assert bot.system_prompt == "You are a helpful assistant."
        assert bot.model_type == bot_class.__name__.replace("Chatbot", "").upper()
        assert bot.model_version == "None"
        assert bot.model_temperature == 0.7
        assert bot.model_max_tokens == 100


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseValidation:
    """Test validation logic in ChatbotBase"""

    def test_valid_name(self, bot_class: type[ChatbotBase]):
        """Test that valid names are accepted"""
        config = ChatbotConfig(
            name="ValidNameBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(), version="test"
            ),
        )
        bot = bot_class(config)
        assert bot.name == "ValidNameBot"

    def test_empty_name(self, bot_class: type[ChatbotBase]):
        """Test that empty names are rejected"""
        with pytest.raises(ValueError, match="Bot name must be"):
            config = ChatbotConfig(
                name="",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="test",
                ),
            )
            bot_class(config)

    def test_whitespace_name(self, bot_class: type[ChatbotBase]):
        """Test that whitespace-only names are rejected"""
        with pytest.raises(ValueError, match="Bot name must be"):
            config = ChatbotConfig(
                name="   ",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="test",
                ),
            )
            bot_class(config)

    def test_invalid_chars_name(self, bot_class: type[ChatbotBase]):
        """Test that names with invalid characters are rejected"""
        invalid_names = ["test!", "test@bot", "test#", "test$", "test%"]
        for name in invalid_names:
            with pytest.raises(ValueError, match="invalid characters"):
                config = ChatbotConfig(
                    name=name,
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                    ),
                )
                bot_class(config)

    def test_invalid_underscore_usage(self, bot_class: type[ChatbotBase]):
        """Test that names with invalid underscore placement are rejected"""
        invalid_names = ["_test", "test_", "_test_"]
        for name in invalid_names:
            with pytest.raises(ValueError, match="invalid underscore usage"):
                config = ChatbotConfig(
                    name=name,
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                    ),
                )
                bot_class(config)

    def test_valid_underscore_usage(self, bot_class: type[ChatbotBase]):
        """Test that names with valid underscore placement are accepted"""
        valid_names = ["test_underscore", "test_more_underscore"]
        for name in valid_names:
            config = ChatbotConfig(
                name=name,
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="test",
                ),
            )
            bot = bot_class(config)
            assert bot.name == name

    def test_duplicate_name(self, bot_class: type[ChatbotBase]):
        """Test that duplicate names are rejected"""
        config = ChatbotConfig(
            name="DuplicateNameBot",
            system_prompt="test",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(), version="test"
            ),
        )
        bot_class(config)
        with pytest.raises(ValueError, match="already in use"):
            bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseTemperature:
    """Test temperature handling in ChatbotBase"""

    def test_valid_temperature(self, bot_class: type[ChatbotBase]):
        """Test valid temperature values"""
        # Use appropriate temperature range based on bot type
        if bot_class.__name__ == "OllamaChatbot":
            valid_temps = [0.0, 0.3, 0.7, 1.0]  # Ollama range 0.0-1.0
        else:
            valid_temps = [0.0, 0.7, 1.0, 1.5, 2.0]  # Standard range 0.0-2.0

        for temp in valid_temps:
            config = ChatbotConfig(
                name=f"TempBot{str(temp).replace('.', '_')}",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="test",
                    params_opt=ChatbotParamsOpt(temperature=temp),
                ),
            )
            bot = bot_class(config)
            assert bot.model_temperature == temp

    def test_invalid_temperature(self, bot_class: type[ChatbotBase]):
        """Test that invalid temperatures are rejected"""
        # Use appropriate invalid temperatures based on bot type
        if bot_class.__name__ == "OllamaChatbot":
            invalid_temps = [-0.1, 1.1]  # Outside Ollama range
        else:
            invalid_temps = [-0.1, 2.1]  # Outside standard range

        for temp in invalid_temps:
            with pytest.raises(
                ValueError, match="(?i).*temperature.*must be between.*"
            ):
                config = ChatbotConfig(
                    name=f"TempBot{str(temp).replace('.', '_').replace('-', 'neg')}",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                        params_opt=ChatbotParamsOpt(temperature=temp),
                    ),
                )
                bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseMaxTokens:
    """Test max tokens handling in ChatbotBase"""

    def test_valid_max_tokens(self, bot_class: type[ChatbotBase]):
        """Test valid max token values"""
        valid_tokens = [1, 50, 100, 1000]
        for tokens in valid_tokens:
            config = ChatbotConfig(
                name=f"TokenBot{tokens}",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="test",
                    params_opt=ChatbotParamsOpt(max_tokens=tokens),
                ),
            )
            bot = bot_class(config)
            assert bot.model_max_tokens == tokens

    def test_invalid_max_tokens(self, bot_class: type[ChatbotBase]):
        """Test that invalid max token values are rejected"""
        invalid_tokens = [0, -1, -100]
        for tokens in invalid_tokens:
            with pytest.raises(ValueError, match="Max tokens.*must be greater than 0"):
                config = ChatbotConfig(
                    name=f"TokenBot{str(tokens).replace('-', 'neg')}",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                        params_opt=ChatbotParamsOpt(max_tokens=tokens),
                    ),
                )
                bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseSystemPrompt:
    """Test system prompt handling in ChatbotBase"""

    def test_empty_system_prompt(self, bot_class: type[ChatbotBase]):
        """Test that empty system prompts are rejected"""
        with pytest.raises(ValueError, match="System prompt cannot be empty"):
            config = ChatbotConfig(
                name="TestBot",
                system_prompt="",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="test",
                ),
            )
            bot_class(config)

    def test_update_system_prompt(self, bot_class: type[ChatbotBase]):
        """Test system prompt update functionality"""
        config = ChatbotConfig(
            name="TestBot",
            system_prompt="Initial system prompt",
            model=ChatbotModel(
                type=bot_class.__name__.replace("Chatbot", "").upper(), version="test"
            ),
        )
        bot = bot_class(config)
        new_prompt = "New system prompt"
        bot.system_prompt = new_prompt
        assert bot.system_prompt == new_prompt
        assert bot.model_system_prompt_needs_update
        bot.model_system_prompt_updated()
        assert not bot.model_system_prompt_needs_update


@pytest.mark.parametrize("bot_class", bot_classes)
class TestChatbotBaseCounter:
    """Test bot instance counting functionality"""

    def test_bot_counter(self, bot_class: type[ChatbotBase]):
        """Test that bot counter increments correctly"""
        initial_count = 0
        bots: list[ChatbotBase] = []
        for i in range(3):
            config = ChatbotConfig(
                name=f"CountBot{i}",
                system_prompt="test",
                model=ChatbotModel(
                    type=bot_class.__name__.replace("Chatbot", "").upper(),
                    version="test",
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

    def test_invalid_model_type(self, bot_class: type[ChatbotBase]):
        """Test that invalid model types are rejected"""
        with pytest.raises(ValueError, match="Invalid model type"):
            config = ChatbotConfig(
                name="TestBot",
                system_prompt="test",
                model=ChatbotModel(type="INVALID", version="test"),
            )
            bot_class(config)


@pytest.mark.parametrize("bot_class", bot_classes)
class TestModelValidation:
    """Test validation logic in _Model"""

    def test_empty_model_type(self, bot_class: type[ChatbotBase]):
        """Test that empty model types are rejected"""
        with pytest.raises(ValueError, match="Model type cannot be empty"):
            _Model(
                type="",
                version="1.0",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_whitespace_model_type(self, bot_class: type[ChatbotBase]):
        """Test that whitespace-only model types are rejected"""
        with pytest.raises(ValueError, match="Model type cannot be empty"):
            _Model(
                type="   ",
                version="1.0",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_empty_model_version(self, bot_class: type[ChatbotBase]):
        """Test that empty model versions are rejected"""
        with pytest.raises(ValueError, match="Model version cannot be empty"):
            _Model(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_whitespace_model_version(self, bot_class: type[ChatbotBase]):
        """Test that whitespace-only model versions are rejected"""
        with pytest.raises(ValueError, match="Model version cannot be empty"):
            _Model(
                type=bot_class.__name__.replace("Chatbot", "").upper(),
                version="   ",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(
                        type=bot_class.__name__.replace("Chatbot", "").upper(),
                        version="test",
                    ),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )
