"""
Example chatbot implementation for demonstration and testing purposes.

This module serves as a reference implementation to illustrate how to create
new chatbot integrations. It implements all required functionality of ChatbotBase
while remaining completely self-contained (no external API dependencies).

Key Educational Points:
- Complete docstrings explaining implementation decisions
- Type hints for all methods and parameters
- Error handling patterns and validation
- Streaming response simulation
- Temperature and token limit implementations
- Proper class hierarchy and method overrides

Example Usage:
    ```python
    config = ChatbotConfig(
        name="example_bot",
        system_prompt="You are a helpful assistant",
        model=ChatbotModel(
            type="DUMMY",
            version="demo-v1",
            params_opt=ChatbotParamsOpt(temperature=0.7)
        )
    )
    
    bot = DummyChatbot(config)
    response = bot.generate_response([
        {"bot_index": 0, "content": "What is 2+2?"}
    ])
    ```

Note:
    This implementation is intended for educational purposes and testing.
    It should not be used in production environments.
"""

import random
import re
import time
from typing import Any, ClassVar, Iterator, List, Optional, Type

from chatbot_conversation.models.base import (
    ChatbotBase,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import APIException, ErrorSeverity

# Model configuration constants
DUMMY_MINIMUM_TEMPERATURE = 0.0
DUMMY_MAXIMUM_TEMPERATURE = 1.0
DUMMY_DEFAULT_TEMPERATURE = 0.7

# Unique identifier for this model type
DUMMY_MODEL_TYPE = "DUMMY"

# Example responses to demonstrate dynamic content generation
_EXAMPLE_RESPONSES = [
    "This is a simulated response showing how chatbots work.",
    "Here's another example of what a bot might say.",
    "Responses can vary in length and content to seem more natural.",
    "Temperature settings influence how random these selections are.",
    "Streaming responses show how content arrives incrementally.",
]


@register_bot(DUMMY_MODEL_TYPE)
class DummyChatbot(ChatbotBase):
    """
    Educational example of a ChatbotBase implementation.

    This class demonstrates the minimum requirements for creating a new chatbot
    integration. It simulates API-like behavior while remaining self-contained
    for testing and learning purposes.

    Implementation Features:
    - Temperature-influenced response selection
    - Simulated streaming responses
    - Proper error handling patterns
    - Example retry scenarios
    - Token limit demonstrations

    Example Config:
        ```python
        config = ChatbotConfig(
            name="test_bot",
            system_prompt="Be helpful and concise",
            model=ChatbotModel(
                type="DUMMY",
                version="demo-v1",
                params_opt=ChatbotParamsOpt(
                    temperature=0.5,
                    max_tokens=100
                )
            )
        )
        ```

    Note:
        This implementation intentionally includes "simulated failures" to
        demonstrate error handling and retry mechanisms.
    """

    # Class variable to store available versions
    _available_versions: ClassVar[List[str]] = [
        "tpg-o1",
        "tpg-o4-mini",
        "tpg-o5-beta",
    ]

    @classmethod
    def available_versions(cls) -> Optional[List[str]]:
        """
        Get list of supported model versions.

        For this dummy implementation, we return a static list of fake versions
        to demonstrate version handling patterns.

        Returns:
            Optional[List[str]]: List of supported version strings
        """
        return cls._available_versions.copy()

    @classmethod
    def _get_class_model_type(cls) -> str:
        """
        Get the model type identifier.

        Returns:
            str: "DUMMY" as the model identifier
        """
        return DUMMY_MODEL_TYPE

    @classmethod
    def _get_model_min_temperature(cls) -> float:
        """Get the minimum allowed temperature value (0.0)."""
        return DUMMY_MINIMUM_TEMPERATURE

    @classmethod
    def _get_model_max_temperature(cls) -> float:
        """Get the maximum allowed temperature value (1.0)."""
        return DUMMY_MAXIMUM_TEMPERATURE

    @classmethod
    def _get_model_default_temperature(cls) -> float:
        """Get the default temperature value (0.7)."""
        return DUMMY_DEFAULT_TEMPERATURE

    @classmethod
    def _retryable_exceptions(cls) -> tuple[Type[Exception], ...]:
        """
        Define which exception types should trigger retry attempts.

        This example shows both standard exceptions and custom API exceptions
        that would warrant retry attempts in a real implementation.

        Returns:
            tuple: Exception types that should trigger retries
        """
        return (APIException, ConnectionError, TimeoutError)

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response based on conversation history.

        This implementation demonstrates:
        1. How to use temperature to influence randomness
        2. Basic response generation patterns
        3. Error simulation for testing
        4. Token limit handling

        Args:
            conversation: List of previous messages in the conversation

        Returns:
            str: Generated response text

        Raises:
            APIException: On simulated API errors
            TimeoutError: On simulated timeout conditions
        """
        # Simulate potential API failures (1 in 100 chance)
        if random.random() < 0.01:
            raise APIException(
                message="Simulated API error for testing",
                user_message="The dummy bot is simulating an API error",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )

        # Use temperature to influence response selection
        if random.random() > self.model_temperature:
            # Low temperature = more deterministic
            response = _EXAMPLE_RESPONSES[0]
        else:
            # High temperature = more random
            response = random.choice(_EXAMPLE_RESPONSES)

        # Simulate processing delay
        time.sleep(0.1)

        return response

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from a streaming response chunk.

        In this dummy implementation, chunks are simple strings.
        Real implementations would need to handle API-specific chunk formats.

        Args:
            chunk: A response chunk (string in this implementation)

        Returns:
            str: Extracted text content
        """
        return str(chunk)

    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[str]:
        """
        Simulate streaming response generation.

        This implementation shows how to:
        1. Generate incremental response chunks
        2. Handle streaming timeouts
        3. Simulate network delays
        4. Demonstrate token limits

        Args:
            conversation: List of conversation messages (unused in dummy implementation)

        Yields:
            str: Response text chunks

        Raises:
            TimeoutError: On simulated timeout conditions
        """
        # Select base response using temperature
        response = self._generate_response(conversation)

        # Split into words while preserving spaces
        chunks = re.findall(r"\S+\s*", response)

        # Stream each chunk with simulated delay
        for chunk in chunks:
            # Simulate network delay
            time.sleep(0.05)

            # Simulate random timeout
            if random.random() < 0.01:
                raise TimeoutError("Simulated streaming timeout")

            yield chunk
