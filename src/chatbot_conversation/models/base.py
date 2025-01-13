"""
This module defines the core components for AI chatbot implementations.

The module provides abstract base classes and supporting classes for implementing
AI chatbot functionality across different models and APIs. It includes message 
structures, configuration settings, and retry logic for API calls.

Classes:
    ChatMessage: A chat message with role and content attributes
    ConversationMessage: A message in a conversation with bot identifier
    BotConfig: Configuration parameters for chatbot initialization
    ChatbotTimeout: Timeout and retry settings for API calls
    SystemPrompt: Manager for chatbot system prompts
    ChatbotBase: Abstract base class for chatbot implementations
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, TypedDict

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    stop_any,
    wait_random_exponential,
)

from chatbot_conversation.utils import get_logger

# Timeout constants (in seconds)
DEFAULT_TOTAL_TIMEOUT = 45  # Maximum time for total trip through API
DEFAULT_API_TIMEOUT = 10  # For per try of API call if child classes api parameter
DEFAULT_MAX_RETRIES = 5  # Maximum number of retry attempts
DEFAULT_MIN_WAIT = 1
DEFAULT_MAX_WAIT = 10
DEFAULT_WAIT_MULTIPLIER = 1.5

logger = get_logger("models")


class ChatMessage(TypedDict):
    """
    A structured message format for API communication.

    Attributes:
        role: The role of the message sender (e.g., 'system', 'user', 'assistant')
        content: The actual message text
    """

    role: str
    content: str


class ConversationMessage(TypedDict):
    """
    A message format for tracking conversations between users and bots.

    Attributes:
        bot_index: Unique identifier for the bot in the conversation
        content: The message content
    """

    bot_index: int
    content: str


@dataclass
class BotConfig:
    """
    Configuration settings for initializing a chatbot instance.

    Attributes:
        bot_name: Display name for the bot instance
        bot_type: The type/model of the chatbot
        bot_version: Version identifier for the bot model
        bot_temp: Temperature setting for response generation (0.0 to 2.0)
        bot_system_prompt: Initial system instructions for the bot
    """

    bot_name: str
    bot_system_prompt: str
    bot_type: str
    bot_version: str
    bot_temp: float = 0.7


@dataclass
class ChatbotTimeout:
    """
    Configuration for API timeout and retry behavior.

    Attributes:
        total: Maximum total time allowed for API operations in seconds
        api_timeout: Timeout for individual API calls in seconds
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds
        wait_multiplier: Exponential backoff multiplier for retry delays
    """

    total: int = DEFAULT_TOTAL_TIMEOUT
    api_timeout: int = DEFAULT_API_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    min_wait: int = DEFAULT_MIN_WAIT
    max_wait: int = DEFAULT_MAX_WAIT
    wait_multiplier: float = DEFAULT_WAIT_MULTIPLIER


class SystemPrompt:
    """
    Manages and tracks changes to a chatbot's system prompt.

    Provides methods for updating, appending, and removing content from the system
    prompt while tracking whether the prompt needs to be refreshed in the chat context.
    """

    def __init__(self, content: str = "") -> None:
        self._content: str = content
        self._needs_update: bool = True

    @property
    def content(self) -> str:
        """Get the current content of the system prompt."""
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        """
        Sets the content of the system prompt and marks it as needing an update.

        Args:
            value (str): The new content for the system prompt
        """
        self._content = value
        self._needs_update = True

    @property
    def needs_update(self) -> bool:
        """Check if the system prompt needs to be updated in the chat context."""
        return self._needs_update

    def mark_updated(self) -> None:
        """Mark the system prompt as having been updated in the chat context."""
        self._needs_update = False

    def add_suffix(self, additional_prompt: str) -> None:
        """
        Append text to the end of the system prompt.

        Args:
            additional_prompt (str): Text to append to the prompt
        """
        if additional_prompt:
            self.content += additional_prompt

    def remove_suffix(self, text_to_remove: str) -> None:
        """
        Remove specified text from the end of the system prompt if present.

        Args:
            text_to_remove (str): Text to remove from the end of the prompt
        """
        if text_to_remove and self.content.endswith(text_to_remove):
            self.content = self.content[: -len(text_to_remove)]

    def add_prefix(self, text_to_add: str) -> None:
        """Add text to start of prompt."""
        if text_to_add:
            self.content = text_to_add + self.content

    def remove_prefix(self, text_to_remove: str) -> None:
        """Remove text from start of prompt if present."""
        if text_to_remove and self.content.startswith(text_to_remove):
            self.content = self.content[len(text_to_remove) :]


class ChatbotBase(ABC):
    """
    Abstract base class defining the core interface for chatbot implementations.

    This class provides common functionality for managing chat state, handling API
    interactions, and implementing retry logic. Child classes must implement
    specific API integration methods.

    Class Attributes:
        _total_count: Tracks the total number of chatbot instances created

    Attributes:
        timeout: ChatbotTimeout configuration for API interactions
        model_version: Version identifier for the bot model
        name: Display name for the bot instance
        api: Instance of the API client for the specific bot implementation
        _bot_index: Unique identifier for this bot instance
        _system_prompt: SystemPrompt instance managing the bot's system instructions
    """

    _total_count: int = 0  # Class variable to track total instances

    @classmethod
    def reset_total_count(cls) -> None:
        """
        Reset the total count of chatbot instances.
        Typically only used in tests to ensure a clean state.
        """
        cls._total_count = 0

    @classmethod
    def get_total_bots(cls) -> int:
        """Get the total number of bot instances."""
        return cls._total_count

    def __init__(
        self,
        bot_name: str,
        bot_system_prompt: str,
        bot_model_version: str,
        bot_temp: float = 0.7,
    ) -> None:
        """
        Initialize the chatbot with model version, system prompt, and bot name.

        Args:
            bot_name (str): The name of the bot.
            bot_model_version (str): The version of the bot model.
            bot_system_prompt (str): The system prompt for the bot.
            bot_temp (float): The temperature setting for response generation.
        """

        self.timeout = ChatbotTimeout()
        self.model_version: str = bot_model_version
        self._system_prompt = SystemPrompt(content=bot_system_prompt)
        self.name: str = bot_name
        self.api: Any = None  # default value for child classes to override
        self.temp: float = bot_temp

        ChatbotBase._total_count += 1
        self._bot_index: int = ChatbotBase._total_count

    @property
    def bot_index(self) -> int:
        """Get the unique identifier for this bot instance."""
        return self._bot_index

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt content."""
        return self._system_prompt.content

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """
        Set the system prompt content.

        Args:
            value (str): The new system prompt content
        """
        self._system_prompt.content = value

    @property
    def system_prompt_needs_update(self) -> bool:
        """Check if the system prompt needs to be refreshed in the chat context."""
        return self._system_prompt.needs_update

    def system_prompt_updated(self) -> None:
        """Mark the system prompt as having been updated in the chat context."""
        self._system_prompt.mark_updated()

    def system_prompt_add_suffix(self, additional_prompt: str) -> None:
        """
        Append additional text to the system prompt.

        Args:
            additional_prompt (str): The text to append.
        """
        self._system_prompt.add_suffix(additional_prompt)

    def system_prompt_remove_suffix(self, text_to_remove: str) -> None:
        """
        Remove specific text from the end of the system prompt.

        Args:
            text_to_remove (str): The text to remove.
        """
        self._system_prompt.remove_suffix(text_to_remove)

    @abstractmethod
    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """Bot-specific logic for which exceptions warrant retry."""
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate a response from the model without any formatting.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The raw response from the model.
        """
        pass  # pylint: disable=unnecessary-pass

    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response based on the conversation history with retry logic.

        Implements exponential backoff and retry logic for API calls, handling
        timeouts and transient errors based on bot-specific criteria.

        Args:
            conversation: List of previous messages in the conversation

        Returns:
            The generated response text

        Raises:
            ValueError: If the model returns an empty response
            Various exceptions: Based on specific API implementations
        """

        # @retry around _inner_generate_response inside generate_response because
        # scope of self._should_retry_on_exception is not available to tenacity
        # when applied as a decorator to _generate_response directly
        @retry(
            stop=stop_any(
                stop_after_attempt(self.timeout.max_retries),
                stop_after_delay(self.timeout.total),
            ),
            wait=wait_random_exponential(
                multiplier=self.timeout.wait_multiplier,
                min=self.timeout.min_wait,
                max=self.timeout.max_wait,
            ),
            retry=retry_if_exception(
                lambda e: (
                    self._should_retry_on_exception(e)
                    if isinstance(e, Exception)
                    else False
                )
            ),
        )
        def _inner_generate_response() -> str:
            return self._generate_response(conversation)

        response_content: str = _inner_generate_response()
        if response_content == "":
            empty_response_error = "Model returned an empty response"
            self._log_error(empty_response_error)
            raise ValueError(empty_response_error)
        return response_content

    def _format_conv_for_api_util(
        self, conversation: List[ConversationMessage], add_system_prompt: bool = True
    ) -> List[ChatMessage]:
        """
        Format conversation history for API submission.

        Converts the internal conversation format to the structure expected by
        common chat APIs. Optionally includes the system prompt at the start
        of the message list.

        Args:
            conversation: List of conversation messages to format
            add_system_prompt: Whether to include system prompt at the start

        Returns:
            List of formatted messages ready for API submission
        """

        messages: List[ChatMessage] = []
        if add_system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for contribution in conversation:
            role = (
                "assistant" if contribution["bot_index"] == self.bot_index else "user"
            )
            messages.append({"role": role, "content": contribution["content"]})

        self._log_debug(json.dumps(messages, indent=2))

        return messages

    def _log_error(self, error_text: str) -> None:
        """
        Logs a model error with the specified format.

        Args:
            error_text (str): The content of the response to log.
        """
        logger.error(
            "Bot Class: %s, Bot Name: %s, Bot Index: %s, %s",
            self.__class__.__name__,
            self.name,
            self.bot_index,
            error_text,
        )

    def _log_debug(self, debug_text: str) -> None:
        """
        Logging for model debug with the specified format.

        Args:
            error_text (str): The content of the response to log.
        """
        logger.debug(
            "Bot Class: %s, Bot Name: %s, Bot Index: %s, %s",
            self.__class__.__name__,
            self.name,
            self.bot_index,
            debug_text,
        )
