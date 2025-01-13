"""
This module contains the DummyChatbot class, a concrete implementation of the ChatbotBase class,
which returns random predefined responses.

The DummyChatbot class handles:
- Generating random responses from a predefined list of sentences.

Classes:
    DummyChatbot: Concrete implementation of chatbot returning random responses.
"""

import random
from typing import List

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot


@register_bot("DUMMY")
class DummyChatbot(ChatbotBase):
    """
    Concrete implementation of chatbot returning random responses.

    Attributes:
        responses: List of predefined responses.
    """

    def __init__(
        self,
        bot_name: str,
        bot_system_prompt: str,
        bot_model_version: str,
        bot_temp: float = 0.7,
    ) -> None:
        """
        Initialize the DummyChatbot with model version, system prompt, and bot name.

        Args:
            bot_model_version (str): The version of the bot model
            bot_system_prompt (str): The system prompt for the bot
            bot_name (str): The name of the bot
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
