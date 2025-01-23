"""
Mock chatbot implementation for testing and demonstration.

Provides a simple implementation of ChatbotBase that returns random responses
from a predefined list. Useful for testing the framework, development without
API dependencies, and as a reference implementation.

Major Classes:
    DummyChatbot: Testing-focused chatbot implementation
"""

import random
import re
from typing import Any, ClassVar, Iterator, List

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
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
        Get the model type identifier for GPT models.

        Returns:
            str: "GPT" as the model type identifier
        """
        return MODEL_TYPE

    # typically __init__ would be defined here with call to
    # super().__init__(config) to initialize the base class
    # and then specifics for the bot implementation

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

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from a dummy stream chunk.

        Simple pass-through implementation for testing purposes.

        Args:
            chunk (Any): A chunk of text from the dummy stream

        Returns:
            str: The chunk content unchanged
        """
        return chunk

    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[Any]:
        """
        Generate a mock streaming response.

        Simulates streaming by splitting a predefined message into tokens
        and yielding them one at a time. Used for testing stream functionality
        without external dependencies.

        Args:
            conversation (list[ConversationMessage]): Unused conversation history

        Returns:
            Iterator[Any]: Stream of text tokens from the predefined response

        Note:
            Splits response on word boundaries and spaces for realistic simulation
        """
        response = (
            "Hello! I'm a simple bot, pretending to stream a response, "
            "regardless of what you say."
        )
        # Use regex to split and keep spaces as separate elements
        tokens = re.findall(r"\S+(?:\s+)?", response)

        yield from tokens
