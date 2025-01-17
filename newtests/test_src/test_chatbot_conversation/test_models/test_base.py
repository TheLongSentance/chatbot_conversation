"""Tests for ChatbotBase class"""

import pytest

from chatbot_conversation.models import ChatbotConfig, ChatbotModel, ChatbotParamsOpt

# import of private class _Model for testing purposes
from chatbot_conversation.models.base import (
    _Model,  # pyright: ignore[reportPrivateUsage]
)
from chatbot_conversation.models.bots.dummy_bot import DummyChatbot


class TestDummyChatbotConfig:
    """Test basic configuration of dummy_chatbot fixture"""

    def test_dummy_chatbot_config(self, dummy_chatbot: DummyChatbot):
        """Test that dummy_chatbot fixture has correct configuration"""
        assert dummy_chatbot.name == "DummyTestBot"
        assert dummy_chatbot.system_prompt == "You are a helpful assistant."
        assert dummy_chatbot.model_type == "DUMMY"
        assert dummy_chatbot.model_version == "None"
        assert dummy_chatbot.model_temperature == 0.7
        assert dummy_chatbot.model_max_tokens == 100


class TestChatbotBaseValidation:
    """Test validation logic in ChatbotBase"""

    def test_valid_name(self, dummy_chatbot: DummyChatbot):
        """Test that valid names are accepted"""
        assert dummy_chatbot.name == "DummyTestBot"

    def test_empty_name(self):
        """Test that empty names are rejected"""
        with pytest.raises(ValueError, match="Bot name must be"):
            config = ChatbotConfig(
                name="",
                system_prompt="test",
                model=ChatbotModel(type="DUMMY", version="test"),
            )
            DummyChatbot(config)

    def test_whitespace_name(self):
        """Test that whitespace-only names are rejected"""
        with pytest.raises(ValueError, match="Bot name must be"):
            config = ChatbotConfig(
                name="   ",
                system_prompt="test",
                model=ChatbotModel(type="DUMMY", version="test"),
            )
            DummyChatbot(config)

    def test_invalid_chars_name(self):
        """Test that names with invalid characters are rejected"""
        invalid_names = ["test!", "test@bot", "test#", "test$", "test%"]
        for name in invalid_names:
            with pytest.raises(ValueError, match="invalid characters"):
                config = ChatbotConfig(
                    name=name,
                    system_prompt="test",
                    model=ChatbotModel(type="DUMMY", version="test"),
                )
                DummyChatbot(config)

    def test_invalid_underscore_usage(self):
        """Test that names with invalid underscore placement are rejected"""
        invalid_names = ["_test", "test_", "_test_"]
        for name in invalid_names:
            with pytest.raises(ValueError, match="invalid underscore usage"):
                config = ChatbotConfig(
                    name=name,
                    system_prompt="test",
                    model=ChatbotModel(type="DUMMY", version="test"),
                )
                DummyChatbot(config)

    def test_valid_underscore_usage(self):
        """Test that names with valid underscore placement are accepted"""
        valid_names = ["test_underscore", "test_more_underscore"]
        for name in valid_names:
            config = ChatbotConfig(
                name=name,
                system_prompt="test",
                model=ChatbotModel(type="DUMMY", version="test"),
            )
            bot = DummyChatbot(config)
            assert bot.name == name

    def test_duplicate_name(self, dummy_chatbot: DummyChatbot):
        """Test that duplicate names are rejected"""
        with pytest.raises(ValueError, match="already in use"):
            config = ChatbotConfig(
                name="DummyTestBot",  # Same name as fixture
                system_prompt="test",
                model=ChatbotModel(type="DUMMY", version="test"),
            )
            DummyChatbot(config)


class TestChatbotBaseTemperature:
    """Test temperature handling in ChatbotBase"""

    def test_valid_temperature(self):
        """Test valid temperature values"""
        valid_temps = [0.0, 0.7, 1.0, 1.5, 2.0]
        for temp in valid_temps:
            config = ChatbotConfig(
                name=f"TempBot{str(temp).replace('.', '_')}",
                system_prompt="test",
                model=ChatbotModel(
                    type="DUMMY",
                    version="test",
                    params_opt=ChatbotParamsOpt(temperature=temp),
                ),
            )
            bot = DummyChatbot(config)
            assert bot.model_temperature == temp

    def test_invalid_temperature(self):
        """Test that invalid temperatures are rejected"""
        invalid_temps = [-0.1, 2.1]
        for temp in invalid_temps:
            with pytest.raises(ValueError, match="Temperature.*must be between"):
                config = ChatbotConfig(
                    name=f"TempBot{str(temp).replace('.', '_').replace('-', 'neg')}",
                    system_prompt="test",
                    model=ChatbotModel(
                        type="DUMMY",
                        version="test",
                        params_opt=ChatbotParamsOpt(temperature=temp),
                    ),
                )
                DummyChatbot(config)


class TestChatbotBaseMaxTokens:
    """Test max tokens handling in ChatbotBase"""

    def test_valid_max_tokens(self):
        """Test valid max token values"""
        valid_tokens = [1, 50, 100, 1000]
        for tokens in valid_tokens:
            config = ChatbotConfig(
                name=f"TokenBot{tokens}",
                system_prompt="test",
                model=ChatbotModel(
                    type="DUMMY",
                    version="test",
                    params_opt=ChatbotParamsOpt(max_tokens=tokens),
                ),
            )
            bot = DummyChatbot(config)
            assert bot.model_max_tokens == tokens

    def test_invalid_max_tokens(self):
        """Test that invalid max token values are rejected"""
        invalid_tokens = [0, -1, -100]
        for tokens in invalid_tokens:
            with pytest.raises(ValueError, match="Max tokens.*must be greater than 0"):
                config = ChatbotConfig(
                    name=f"TokenBot{str(tokens).replace('-', 'neg')}",
                    system_prompt="test",
                    model=ChatbotModel(
                        type="DUMMY",
                        version="test",
                        params_opt=ChatbotParamsOpt(max_tokens=tokens),
                    ),
                )
                DummyChatbot(config)


class TestChatbotBaseSystemPrompt:
    """Test system prompt handling in ChatbotBase"""

    def test_empty_system_prompt(self):
        """Test that empty system prompts are rejected"""
        with pytest.raises(ValueError, match="System prompt cannot be empty"):
            config = ChatbotConfig(
                name="TestBot",
                system_prompt="",
                model=ChatbotModel(type="DUMMY", version="test"),
            )
            DummyChatbot(config)

    def test_update_system_prompt(self, dummy_chatbot: DummyChatbot):
        """Test system prompt update functionality"""
        new_prompt = "New system prompt"
        dummy_chatbot.system_prompt = new_prompt
        assert dummy_chatbot.system_prompt == new_prompt
        assert dummy_chatbot.model_system_prompt_needs_update
        dummy_chatbot.model_system_prompt_updated()
        assert not dummy_chatbot.model_system_prompt_needs_update


class TestChatbotBaseCounter:
    """Test bot instance counting functionality"""

    def test_bot_counter(self):
        """Test that bot counter increments correctly"""
        initial_count = 0
        bots: list[DummyChatbot] = []
        for i in range(3):
            config = ChatbotConfig(
                name=f"CountBot{i}",
                system_prompt="test",
                model=ChatbotModel(type="DUMMY", version="test"),
            )
            bot = DummyChatbot(config)
            bots.append(bot)
            initial_count += 1
            assert bot.bot_index == initial_count
            assert DummyChatbot.get_total_bots() == initial_count


class TestChatbotBaseModelType:
    """Test model type validation"""

    def test_invalid_model_type(self, dummy_chatbot: DummyChatbot):
        """Test that invalid model types are rejected"""
        with pytest.raises(ValueError, match="Invalid model type"):
            config = ChatbotConfig(
                name="TestBot",
                system_prompt="test",
                model=ChatbotModel(type="INVALID", version="test"),
            )
            DummyChatbot(config)


class TestModelValidation:
    """Test validation logic in _Model"""

    def test_empty_model_type(self):
        """Test that empty model types are rejected"""
        with pytest.raises(ValueError, match="Model type cannot be empty"):
            _Model(
                type="",
                version="1.0",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(type="DUMMY", version="test"),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_whitespace_model_type(self):
        """Test that whitespace-only model types are rejected"""
        with pytest.raises(ValueError, match="Model type cannot be empty"):
            _Model(
                type="   ",
                version="1.0",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(type="DUMMY", version="test"),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_empty_model_version(self):
        """Test that empty model versions are rejected"""
        with pytest.raises(ValueError, match="Model version cannot be empty"):
            _Model(
                type="DUMMY",
                version="",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(type="DUMMY", version="test"),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )

    def test_whitespace_model_version(self):
        """Test that whitespace-only model versions are rejected"""
        with pytest.raises(ValueError, match="Model version cannot be empty"):
            _Model(
                type="DUMMY",
                version="   ",
                timeout=ChatbotConfig(
                    name="TestBot",
                    system_prompt="test",
                    model=ChatbotModel(type="DUMMY", version="test"),
                ).timeout,
                temperature=0.7,
                max_tokens=100,
            )
