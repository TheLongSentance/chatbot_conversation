"""
This module contains tests for the BotsInitializer class.
"""
from chatbot_conversation.conversation import ConversationConfig
from chatbot_conversation.conversation.bots_initializer import BotsInitializer
from chatbot_conversation.models import ChatbotBase


class TestBotsInitializer:
    """
    Tests for the BotsInitializer class and its functionality.
    """

    def test_initialize_bots(self, sample_conversation_config: ConversationConfig) -> None:
        """
        Test that initialize_bots returns a non-empty list of ChatbotBase instances.
        """
        initializer = BotsInitializer(sample_conversation_config)
        result = initializer.initialize_bots(sample_conversation_config)
        assert len(result) == 2
        for bot in result:
            assert isinstance(bot, ChatbotBase)
