"""Tests for DummyChatbot-specific functionality"""

from unittest.mock import patch

from chatbot_conversation.models import ChatbotConfig, ChatbotModel
from chatbot_conversation.models.base import ConversationMessage
from chatbot_conversation.models.bots.dummy_bot import DummyChatbot


def test_should_retry_on_exception() -> None:
    """Test that DummyChatbot never retries"""
    config = ChatbotConfig(
        name="TestBot",
        system_prompt="test",
        model=ChatbotModel(type="DUMMY", version="test"),
    )
    bot = DummyChatbot(config)

    # Test with various exception types
    assert not bot._should_retry_on_exception(  # pyright: ignore[reportPrivateUsage]
        Exception()
    )
    assert not bot._should_retry_on_exception(  # pyright: ignore[reportPrivateUsage]
        ValueError()
    )
    assert not bot._should_retry_on_exception(  # pyright: ignore[reportPrivateUsage]
        RuntimeError()
    )


def test_generate_response_uses_predefined_responses() -> None:
    """Test that responses come from predefined list"""
    config = ChatbotConfig(
        name="TestBot",
        system_prompt="test",
        model=ChatbotModel(type="DUMMY", version="test"),
    )
    bot = DummyChatbot(config)

    # Create test conversation
    conversation: list[ConversationMessage] = [
        {"bot_index": 1, "content": "Hello"},
        {"bot_index": 2, "content": "Hi there"},
    ]

    # Get multiple responses and verify they're from the predefined list
    responses: set[str] = set()
    for _ in range(20):  # Get enough responses to likely see multiple unique ones
        response = bot.generate_response(conversation)
        responses.add(response)
        assert (
            response
            in bot._responses  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        )

    # Verify we got multiple different responses
    assert len(responses) > 1, "Random selection doesn't seem to be working"


def test_generate_response_ignores_conversation() -> None:
    """Test that response generation ignores conversation content"""
    config = ChatbotConfig(
        name="TestBot",
        system_prompt="test",
        model=ChatbotModel(type="DUMMY", version="test"),
    )
    bot = DummyChatbot(config)

    # Test with different conversations
    conv1: list[ConversationMessage] = [
        {"bot_index": 1, "content": "Hello"},
    ]
    conv2: list[ConversationMessage] = [
        {"bot_index": 1, "content": "Completely different content"},
    ]

    # Mock random.choice to return a fixed response
    with patch("random.choice", return_value="Test response"):
        response1 = bot.generate_response(conv1)
        response2 = bot.generate_response(conv2)

        # Responses should be identical despite different conversations
        assert response1 == response2


def test_model_constants() -> None:
    """Test that model constants are correctly defined"""
    config = ChatbotConfig(
        name="TestBot",
        system_prompt="test",
        model=ChatbotModel(type="DUMMY", version="test"),
    )
    bot = DummyChatbot(config)

    assert bot.model_type == "DUMMY"
    assert bot._default_temperature == 1.0  # pyright: ignore[reportPrivateUsage]
    assert bot.get_default_max_tokens() == 50  # pyright: ignore[reportPrivateUsage]
