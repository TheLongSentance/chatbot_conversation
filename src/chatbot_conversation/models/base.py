"""
This module defines the abstract base class for AI chatbot implementations.

Classes:
    ChatMessage: Represents a typical chat message with a role and content.
    ConversationMessage: Represents a conversation message with a bot index and content.
    BotConfig: Configuration for creating a chatbot.
    ChatbotBase: Abstract base class defining interface for AI chatbot implementations.
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
DEFAULT_TOTAL_TIMEOUT = 30  # Maximum time for total trip through API
DEFAULT_API_TIMEOUT = 6  # For per try of API call if child classes api parameter
DEFAULT_MAX_RETRIES = 5  # Maximum number of retry attempts
DEFAULT_MIN_WAIT = 1
DEFAULT_MAX_WAIT = 10
DEFAULT_WAIT_MULTIPLIER = 1.5

logger = get_logger("models")


class ChatMessage(TypedDict):
    """Represents a typical chat message with a role and content."""

    role: str
    content: str


class ConversationMessage(TypedDict):
    """Represents a conversation message with a bot index and content."""

    bot_index: int
    content: str


@dataclass
class BotConfig:
    """Configuration for creating a chatbot."""

    bot_type: str
    bot_version: str
    bot_system_prompt: str
    bot_name: str


@dataclass
class ChatbotTimeout:
    """Configuration settings for chatbot behavior."""

    total: int = DEFAULT_TOTAL_TIMEOUT
    api_timeout: int = DEFAULT_API_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    min_wait: int = DEFAULT_MIN_WAIT
    max_wait: int = DEFAULT_MAX_WAIT
    wait_multiplier: float = DEFAULT_WAIT_MULTIPLIER


class ChatbotBase(ABC):
    """Abstract base class defining the interface for AI chatbot implementations."""

    _total_count: int = 0  # Class variable to track total instances

    @classmethod
    def reset_total_count(cls) -> None:
        """
        Reset the total count of chatbot instances.
        Typically only used in tests to ensure a clean state.
        """
        cls._total_count = 0

    def __init__(
        self,
        bot_model_version: str,
        bot_system_prompt: str,
        bot_name: str,
    ):
        """
        Initialize the chatbot with model version, system prompt, and bot name.

        Args:
            bot_model_version (str): The version of the bot model.
            bot_system_prompt (str): The system prompt for the bot.
            bot_name (str): The name of the bot.
        """

        self.timeout = ChatbotTimeout()
        self.model_version: str = bot_model_version
        self._system_prompt: str = bot_system_prompt
        self.name: str = bot_name
        self.api = self._initialize_api()
        ChatbotBase._total_count += 1
        self._bot_index: int = ChatbotBase._total_count

    @property
    def bot_index(self) -> int:
        """Get the index of the bot instance."""
        return self._bot_index

    @classmethod
    def get_total_bots(cls) -> int:
        """Get the total number of bot instances."""
        return cls._total_count

    @property
    def system_prompt(self) -> str:
        """Get the system prompt of the bot."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, new_prompt: str) -> None:
        """Set a new system prompt for the bot."""
        self._system_prompt = new_prompt

    def append_to_system_prompt(self, additional_prompt: str) -> None:
        """Append additional text to the current system prompt."""
        self._system_prompt += additional_prompt

    def remove_from_system_prompt(self, text_to_remove: str) -> None:
        """
        Remove a specific string from the system prompt if it exists.

        Args:
            text_to_remove (str): The string to remove from the system prompt.
        """
        if text_to_remove in self._system_prompt:
            self._system_prompt = self._system_prompt.replace(text_to_remove, "")

    def unappend_from_system_prompt(self, text_to_remove: str) -> None:
        """
        Remove text from the end of the system prompt if it exists.
        Only removes if the text appears at the very end of the prompt.

        Args:
            text_to_remove (str): The string to remove from the end of system prompt.
        """
        if self._system_prompt.endswith(text_to_remove):
            self._system_prompt = self._system_prompt[: -len(text_to_remove)]

    @abstractmethod
    def _initialize_api(self) -> Any:
        """Initialize the API for the chatbot."""
        pass  # pylint: disable=unnecessary-pass

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
        Generate a response from the model using the conversation history.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The response from the model.
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
        Format message history for submission to APIs.

        Prepends system prompt and formats all messages according to
        typical bot API's expected structure.

        Does not begin with system prompt if add_system_prompt is False.
        (e.g., for Claude API)

        Not applicable for Gemini API.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages to format.
            add_system_prompt (bool, optional): Whether to prepend system prompt. Defaults to True.

        Returns:
            List[ChatMessage]: Messages formatted for OpenAI, Claude, or Ollama API.
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
