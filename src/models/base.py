"""
This module defines the abstract base class for AI chatbot implementations.

Classes:
    ChatMessage: Represents a chat message with a role and content.
    GeminiMessage: Represents a Gemini message with a role and parts.
    ConversationMessage: Represents a conversation message with a bot index and content.
    BotType: Enumeration of different bot types.
    ChatbotBase: Abstract base class defining interface for AI chatbot implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any, TypedDict, TypeVar, Generic
from enum import Enum, auto

class ChatMessage(TypedDict):
    """Represents a typical chat message with a role and content."""
    role: str
    content: str

class GeminiMessage(TypedDict):
    """Represents a Gemini-specific message with a role and parts."""
    role: str
    parts: str

# Define generic model input message type
T = TypeVar('T', ChatMessage, GeminiMessage)

class ConversationMessage(TypedDict):
    """Represents a conversation message with a bot index and content."""
    bot_index: int
    content: str

class BotType(Enum):
    """Enumeration of different bot types."""
    GPT = auto()
    CLAUDE = auto()
    GEMINI = auto()
    OLLAMA = auto()

class ChatbotBase(ABC, Generic[T]):
    """Abstract base class defining interface for AI chatbot implementations."""

    _total_count: int = 0  # Class variable to track total instances

    @classmethod
    def reset_total_count(cls) -> None:
        """
        Reset the total count of chatbot instances.
        Typically only used in tests to ensure a clean state.
        """
        cls._total_count = 0

    def __init__(self, bot_model_version: str,
                 bot_specific_system_prompt: str,
                 bot_name: str,
                 shared_system_prompt_prefix: str):
        """
        Initialize the chatbot with model version, system prompt, and bot name.

        Args:
            bot_model_version (str): The version of the bot model.
            bot_specific_system_prompt (str): The specific system prompt for the bot.
            bot_name (str): The name of the bot.
            shared_system_prompt_prefix (str): The shared system prompt prefix.
        """
        self.model_version: str = bot_model_version
        self.system_prompt: str = shared_system_prompt_prefix.format(bot_name=bot_name) \
                                + bot_specific_system_prompt
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
    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate raw response from the model without any formatting.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The raw response from the model.
        """
        pass  # pylint: disable=unnecessary-pass

    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate and format response with proper name prefix.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The formatted response.
        """
        try:
            raw_response = self._generate_raw_response(conversation)
            return self.format_response(raw_response)
        except Exception as e: # pylint: disable=broad-except
            print(f"Error generating response: {e}")
            return self.format_response("Error: Unable to generate response.")

    def _strip_name_prefix(self, response: str) -> str:
        """
        Remove bot name prefix if present in response.

        Args:
            response (str): The response string.

        Returns:
            str: The response without the bot name prefix.
        """
        prefix = f"{self.name}: "
        return response[len(prefix):] if response.startswith(prefix) else response

    def format_response(self, response: str) -> str:
        """
        Format response with bot name prefix.

        Args:
            response (str): The response string.

        Returns:
            str: The formatted response with bot name prefix.
        """
        clean_response = self._strip_name_prefix(response)
        return f"{self.name}: {clean_response}"

    @abstractmethod
    def _format_message(self, conversation: List[ConversationMessage]) -> List[T]:
        """
        Format the conversation messages into the required format.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            List[T]: The formatted messages.
        """
        pass  # # pylint: disable=unnecessary-pass
