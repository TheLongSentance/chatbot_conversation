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

from chatbot_conversation.models.bot_types import BotType
from chatbot_conversation.utils import get_logger

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

    bot_type: BotType
    bot_model_version: str
    bot_system_prompt: str
    bot_name: str


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
        self.model_version: str = bot_model_version
        self.system_prompt: str = bot_system_prompt
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

    @abstractmethod
    def _initialize_api(self) -> Any:
        """Initialize the API for the chatbot."""
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response from the model without any formatting.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The raw response from the model.
        """
        pass  # pylint: disable=unnecessary-pass

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

        logger.debug(
            "Bot Class: %s, Bot Name: %s, Bot Index: %s, Formatted Messages: %s",
            self.__class__.__name__,
            self.name,
            self.bot_index,
            json.dumps(messages, indent=2),
        )

        return messages

    def log_error(self, error_text: str) -> None:
        """
        Logs an error with the specified format.

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
