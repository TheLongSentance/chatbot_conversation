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

    def _initialize_api(self) -> None:
        """
        Dummy initialization method.

        Returns:
            None
        """
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
        return random.choice(self.responses)
