"""
A simple mock chatbot implementation for testing and demonstration purposes.

This module provides a stateless chatbot implementation that returns random responses
from a predefined list. It serves multiple purposes:
- Testing the chatbot framework without external dependencies
- Providing a reference implementation for custom chatbots
- Demonstrating basic chatbot functionality

Key Components:
    DummyChatbot: A concrete implementation of ChatbotBase for testing
    MODEL_TYPE: Constant identifier for the dummy bot model
"""

import random
import re
from typing import Any, ClassVar, Iterator, List

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot

# Default temperature values for Dummy models
MINIMUM_TEMPERATURE = 0.0
MAXIMUM_TEMPERATURE = 1.0
DEFAULT_TEMPERATURE = 1.0

# Default max tokens for Dummy models
DUMMY_MAX_TOKENS = 50

MODEL_TYPE = "DUMMY"


@register_bot(MODEL_TYPE)
class DummyChatbot(ChatbotBase):
    """
    A mock chatbot that provides predefined responses for testing purposes.

    This implementation demonstrates the basic structure of a chatbot while
    requiring no external dependencies. It randomly selects responses from
    a predefined list, making it suitable for testing the framework and
    showing basic chatbot behavior.

    Key Features:
        - Dependency-free response generation
        - Configurable mock parameters
        - Support for both streaming and non-streaming responses
        - Simple error handling demonstration
        - Test-friendly implementation

    Args:
        config (ChatbotConfig): Configuration containing:
            name (str): Unique identifier for this bot instance
            system_prompt (str): Initial instructions (not used in dummy implementation)
            model (dict): Model configuration (only type validation)
            timeout (float): Response timeout in seconds (not used)
    """

    # Not a standard thing to do for a model, but for demonstration purposes
    _responses: ClassVar[List[str]] = [
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

    @classmethod
    def _get_class_model_type(cls) -> str:
        """
        Retrieve the model type identifier for this chatbot implementation.

        Returns:
            str: The constant MODEL_TYPE value identifying this as a dummy bot
        """
        return MODEL_TYPE

    # typically __init__ would be defined here with call to
    # super().__init__(config) to initialize the base class
    # and then specifics for the bot implementation

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if a failed operation should be retried.

        Implements a simple retry strategy that only retries on ConnectionError,
        primarily for demonstration purposes.

        Args:
            exception (Exception): The exception that occurred during operation

        Returns:
            bool: True if the operation should be retried, False otherwise
        """
        return isinstance(exception, (ConnectionError))

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Create a response by randomly selecting from predefined messages.

        This method demonstrates the simplest possible response generation,
        ignoring the conversation history and selecting randomly from a fixed list.

        Args:
            conversation (List[ConversationMessage]): The conversation history (unused)

        Returns:
            str: A randomly selected response from the predefined list
        """
        return random.choice(self._responses)

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from a stream chunk.

        In this dummy implementation, chunks are already strings and are
        passed through unchanged.

        Args:
            chunk (Any): A chunk of response text

        Returns:
            str: The input chunk, unmodified
        """
        return chunk  # type: ignore

    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[Any]:
        """
        Simulate a streaming response by yielding tokens sequentially.

        Demonstrates streaming behavior by splitting a fixed response into
        word-based tokens and yielding them one at a time. This provides
        a way to test streaming functionality without external dependencies.

        Args:
            conversation (list[ConversationMessage]): The conversation history (unused)

        Returns:
            Iterator[Any]: A stream of text tokens representing words and spaces

        Note:
            Uses regex to preserve spacing, creating a more realistic streaming simulation
        """
        response = (
            "Hello! I'm a simple bot, pretending to stream a response, "
            "regardless of what you say."
        )
        # Use regex to split and keep spaces as separate elements
        tokens = re.findall(r"\S+(?:\s+)?", response)

        yield from tokens

    @property
    def model_min_temperature(self) -> float:
        """
        Get the minimum allowed temperature setting.

        Returns:
            float: The minimum valid temperature value (0.0)
        """
        return MINIMUM_TEMPERATURE

    @property
    def model_max_temperature(self) -> float:
        """
        Get the maximum allowed temperature setting.

        Returns:
            float: The maximum valid temperature value (1.0)
        """
        return MAXIMUM_TEMPERATURE

    @property
    def model_default_temperature(self) -> float:
        """
        Get the default temperature setting.

        Returns:
            float: The default temperature value (1.0)
        """
        return DEFAULT_TEMPERATURE
