"""
Mock chatbot implementation for testing and demonstration.

Provides a simple implementation of ChatbotBase that returns random responses
from a predefined list. Useful for testing the framework, development without
API dependencies, and as a reference implementation.

Major Classes:
    DummyChatbot: Testing-focused chatbot implementation
"""

import random
from typing import List

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot

MODEL_TYPE = "DUMMY"

# Default temperature for Dummy models
DUMMY_DEFAULT_TEMP = 1.0
# Default max tokens for Dummy models
DUMMY_MAX_TOKENS = 50


@register_bot(MODEL_TYPE)
class DummyChatbot(ChatbotBase):
    """
    Mock chatbot implementation returning predefined responses.

    Provides a concrete implementation of ChatbotBase focused on testing
    and demonstration, requiring no external dependencies.

    Features:
    - Zero-dependency response generation
    - Predefined response collection
    - Configurable mock parameters
    - Simplified error handling
    - Local-only operation

    Args:
        config (ChatbotConfig): Configuration object containing:
            - name: Bot instance identifier
            - system_prompt: Unused but required system instructions
            - model: Model configuration (only type validation used)
            - timeout: Unused timeout settings

    Attributes:
        Inherits all attributes from ChatbotBase plus:
        _responses (List[str]): Collection of predefined responses
    """

    def __init__(self, config: ChatbotConfig) -> None:
        """
        Initialize dummy chatbot with basic configuration.

        Sets up predefined responses and validates configuration.

        Args:
            config (ChatbotConfig): Basic configuration (mostly unused)
        """
        super().__init__(config)  # pylint: disable=duplicate-code

        # Typically bot-specific api initialization would be done here

        # In this case just dummy responses for the dummy bot
        self._responses = [
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

    def _get_model_type(self) -> str:
        """
        Get the model type identifier for dummy models.

        Returns:
            str: "DUMMY" as the model type identifier
        """
        return MODEL_TYPE

    @property
    def _default_temperature(self) -> float:
        """Default temperature override"""
        return DUMMY_DEFAULT_TEMP

    def _get_default_max_tokens(self) -> int:
        """
        Get default maximum token count for dummy models.
        Token count has no effect on response generation.

        Returns:
            int: Unused token limit (50)
        """
        return DUMMY_MAX_TOKENS

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an operation should be retried.
        Arbitrarily to retry on ConnectionError for
        testing and demonstration purposes.

        Args:
            exception: Unused exception parameter

        Returns:
            bool: True if ConnectionError - False otherwise
        """
        return isinstance(exception, (ConnectionError))

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a mock response by random selection.

        Ignores conversation history and returns a random predefined message.
        No API calls or external dependencies are used.

        Args:
            conversation: Unused conversation history

        Returns:
            str: Randomly selected predefined response
        """
        return random.choice(self._responses)
