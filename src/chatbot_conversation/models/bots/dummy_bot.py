"""
This module implements a simple DummyChatbot for testing and demonstration purposes.

The DummyChatbot provides a basic implementation of the ChatbotBase class that:
- Returns random responses from a predefined list
- Demonstrates the basic structure of a chatbot implementation
- Serves as a testing and development reference
- Requires minimal setup with no external API dependencies

This implementation is useful for:
- Testing the chatbot framework
- Local development without API credentials
- Understanding the basic chatbot architecture

Classes:
    DummyChatbot: Simple chatbot implementation returning random responses.
"""

import random
from typing import List, Optional

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot

# Default temperature for Dummy models
DUMMY_DEFAULT_TEMP = 1.0


@register_bot("DUMMY")
class DummyChatbot(ChatbotBase):
    """
    Simple chatbot implementation that returns random predefined responses.

    This implementation provides:
    - Zero-dependency response generation
    - Predefined set of generic responses
    - Demonstration of ChatbotBase interface
    - Mock temperature parameter (not used in generation)

    Note: Temperature setting has no effect on response generation as responses
    are selected randomly from a fixed list.

    Attributes:
        responses (List[str]): Collection of predefined response messages
        temp (float): Unused temperature parameter (included for interface consistency)
        model_version (str): Version identifier (for compatibility)
        system_prompt (str): System instructions (for compatibility)
    """

    def __init__(
        self,
        bot_name: str,
        bot_system_prompt: str,
        bot_model_version: str,
        bot_temp: Optional[float] = None,
    ) -> None:
        """
        Initialize the DummyChatbot with model version, system prompt, and bot name.

        Args:
            bot_name: The name of the bot
            bot_system_prompt: The system prompt for the bot
            bot_model_version: The version of the bot model
            bot_temp: Optional temperature parameter (defaults to None)
        """
        super().__init__(  # pylint: disable=duplicate-code
            bot_name=bot_name,
            bot_system_prompt=bot_system_prompt,
            bot_model_version=bot_model_version,
            bot_temp=bot_temp,
        )

        # Typically bot-specific api initialization would be done here

        # In this case just dummy responses for the dummy bot
        self.responses = [
            "Hello! How can I assist you today?",
            "I'm here to help you with any questions.",
            "What can I do for you?",
            "Feel free to ask me anything.",
            "I'm a dummy bot, but I'll try my best to help.",
            "How can I make your day better?",
            "Let's chat! What do you want to talk about?",
            "I'm here to assist you with your queries.",
            "Ask me anything, I'm here to help.",
            "What would you like to know today?",
        ]

    def _get_default_temperature(self) -> float:
        """
        Return the default temperature setting for Dummy models.

        Returns:
            float: Default temperature value (1.0) for Dummy response generation
        """
        return DUMMY_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Check if the exception is a network error or timeout.

        Args:
            exception (Exception): The exception to check.

        Returns:
            bool: True if the exception is a network error or timeout, False otherwise.
        """
        return False

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate a random response from the predefined list.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: A random response from the predefined list.
        """
        # Dummy bot returns a random response from the predefined list
        # instead of using an API to generate a response
        return random.choice(self.responses)
