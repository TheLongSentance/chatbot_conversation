"""
Core components for building AI chatbot systems.

This module provides a comprehensive framework for implementing AI chatbots 
that can interface with various language models and APIs. It includes robust
message handling, configuration management, and fault-tolerant API communication.

Core Components:
- Message Structures: Standardized formats for API and internal communication
- Configuration: Settings for bot initialization, timeouts, and model parameters
- System Prompt Management: Stateful handling of system instructions
- Base Implementation: Abstract foundation with common chatbot functionality
- Error Handling: Comprehensive retry logic for API resilience

Major Classes:
    ChatMessage: Standardized API message format
    ConversationMessage: Internal message tracking format
    ChatbotTimeout: API communication settings
    ChatbotParamsOpt: Optional LLM runtime parameters
    ChatbotModel: Model configuration container
    ChatbotConfig: Bot instance configuration
    ChatbotBase: Abstract foundation for implementations
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, TypedDict

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
# Model temperature range (most seem to be adopting range of 0.0 to 2.0)
MIN_MODEL_TEMP = 0.0
MAX_MODEL_TEMP = 2.0
# Default maximum tokens for response generation
DEFAULT_MAX_TOKENS = 300

logger = get_logger("models")


class ChatMessage(TypedDict):
    """
    Standard message format for API interactions.

    Represents a single message in the chat sequence with role-based
    attribution and content.

    Attributes:
        role (str): Identifies message source ('system', 'user', 'assistant')
        content (str): The actual message text
    """

    role: str
    content: str


class ConversationMessage(TypedDict):
    """
    Internal format for conversation state management.

    Tracks messages with their source bot for multi-bot conversations
    and maintains conversation history.

    Attributes:
        bot_index (int): Integer identifier of the source bot
        content (str): The message content
    """

    bot_index: int
    content: str


@dataclass
class ChatbotTimeout:
    """
    API communication timeout and retry configuration.

    Controls the timing and retry behavior for API interactions to ensure
    reliable operation even with unstable connections.

    Attributes:
        total (int): Overall timeout for complete API operations (seconds)
        api_timeout (int): Individual API call timeout (seconds)
        max_retries (int): Maximum retry attempts for failed calls
        min_wait (int): Minimum delay between retries (seconds)
        max_wait (int): Maximum delay between retries (seconds)
        wait_multiplier (float): Factor for exponential backoff calculation
    """

    total: int = DEFAULT_TOTAL_TIMEOUT
    api_timeout: int = DEFAULT_API_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    min_wait: int = DEFAULT_MIN_WAIT
    max_wait: int = DEFAULT_MAX_WAIT
    wait_multiplier: float = DEFAULT_WAIT_MULTIPLIER


@dataclass
class ChatbotParamsOpt:
    """
    Optional LLM runtime parameters for chatbot instances.

    Attributes:
        temperature (Optional[float]): Temperature setting for response generation
        max_tokens (Optional[int]): Maximum tokens for response generation
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # Add validation
    def __post_init__(self):
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")


@dataclass
class ChatbotModel:
    """
    Model information for chatbot instances.

    Attributes:
        type (str): The type of the model
        version (str): The version of the model
        params_opt (ChatbotParamsOpt): Optional runtime parameters for the model
    """

    type: str
    version: str
    params_opt: ChatbotParamsOpt = field(default_factory=ChatbotParamsOpt)


@dataclass
class ChatbotConfig:
    """
    Runtime configuration for chatbot instances.

    Attributes:
        name (str): The name of the bot
        system_prompt (str): The system prompt for the bot
        model (ChatbotModel): The model configuration for the bot
        timeout (ChatbotTimeout): The timeout and retry configuration
    """

    name: str
    system_prompt: str
    model: ChatbotModel
    timeout: ChatbotTimeout = field(default_factory=ChatbotTimeout)


class ChatbotBase(ABC):
    """
    Foundation class for chatbot implementations.

    Provides a robust framework for building chatbot implementations with
    common functionality for state management, API interaction, and error
    handling. Child classes need only implement specific API integration
    methods.

    Features:
    - Managed system prompt handling
    - Configurable timeout and retry logic
    - Conversation state management
    - Unique bot instance tracking
    - Temperature control for response generation
    - Debug and error logging support

    Class Attributes:
        _total_count (int): Running total of chatbot instances created

    Instance Attributes:
        _name (str): Bot instance display name
        _model_type (str): Bot model type identifier
        _model_version (str): Bot model version identifier
        _model_timeout (ChatbotTimeout): API timeout and retry configuration
        _model_temperature (float): Temperature setting for response generation
        _model_max_tokens (int): Maximum tokens for response generation
        _system_prompt (str): System prompt manager
        _model_api (Any): API client instance (set by child classes)
        _bot_index (int): Unique instance identifier
        _model_system_prompt_needs_update (bool): Flag indicating if system prompt needs update
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
        """
        Get the total number of bot instances.

        Returns:
            int: The total number of bot instances created.
        """
        return cls._total_count

    def __init__(
        self,
        config: ChatbotConfig,
    ) -> None:
        """
        Initialize the chatbot with model version, system prompt, and bot name.

        Args:
            config (ChatbotConfig): The configuration for the chatbot instance.
        """
        # Validate model type before setting attributes
        expected_type = self._get_model_type()
        if config.model.type != expected_type:
            raise ValueError(
                f"Invalid model type for {self.__class__.__name__}: "
                f"got '{config.model.type}', expected '{expected_type}'"
            )

        # Read-only configuration
        self._name: str = config.name
        self._model_type: str = config.model.type
        self._model_version: str = config.model.version
        self._model_timeout = config.timeout
        self._update_temperature(config.model.params_opt.temperature)
        self._model_max_tokens: int = (
            config.model.params_opt.max_tokens
            if config.model.params_opt.max_tokens is not None
            else self._get_default_max_tokens()
        )

        # Mutable state
        self._system_prompt: str = config.system_prompt
        self._model_api: Any = None

        # Internal tracking
        self._model_system_prompt_needs_update: bool = True
        ChatbotBase._total_count += 1
        self._bot_index: int = ChatbotBase._total_count

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_temperature(self) -> float:
        return self._model_temperature

    @property
    def _min_temperature(self) -> float:
        """Protected minimum temperature, can be overridden"""
        return MIN_MODEL_TEMP

    @property
    def _max_temperature(self) -> float:
        """Protected maximum temperature, can be overridden"""
        return MAX_MODEL_TEMP

    @property
    @abstractmethod
    def _default_temperature(self) -> float:
        """Protected default temperature, must be overridden"""
        pass  # pylint: disable=unnecessary-pass

    def _update_temperature(self, value: float | None) -> None:
        """Protected method to update temperature with validation"""
        if value is None:
            value = self._default_temperature
        if not self._min_temperature <= value <= self._max_temperature:
            raise ValueError(
                f"Temperature for {self.__class__.__name__} must be between "
                f"{self._min_temperature} and {self._max_temperature}"
            )
        self._model_temperature = value

    @property
    def model_max_tokens(self) -> int:
        return self._model_max_tokens

    @property
    def model_timeout(self) -> ChatbotTimeout:
        return self._model_timeout

    @property
    def bot_index(self) -> int:
        return self._bot_index

    # Read-write properties
    @property
    def model_api(self) -> Any:
        return self._model_api

    @model_api.setter
    def model_api(self, value: Any) -> None:
        self._model_api = value

    @property
    def system_prompt(self) -> str:
        """
        Get the current system prompt content.

        Returns:
            str: The current system prompt content.
        """
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """
        Set the system prompt content.

        Args:
            value (str): The new system prompt content.
        """
        self._system_prompt = value
        self._model_system_prompt_needs_update = True

    @property
    def model_system_prompt_needs_update(self) -> bool:
        """
        Check if the system prompt needs to be updated in the model.

        Returns:
            bool: True if the system prompt needs to be updated, False otherwise.
        """
        return self._model_system_prompt_needs_update

    def model_system_prompt_updated(self) -> None:
        """
        Mark the model system prompt as updated.
        """
        self._model_system_prompt_needs_update = False

    @abstractmethod
    def _get_model_type(self) -> str:
        """
        Get the model type identifier for the chatbot.

        Returns:
            str: The model type identifier for the chatbot.
        """
        pass  # pylint: disable=unnecessary-pass

    def _get_default_max_tokens(self) -> int:
        """
        Get default max_tokens.

        Returns:
            int: The default maximum tokens for response generation.
        """
        return DEFAULT_MAX_TOKENS

    @abstractmethod
    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Bot-specific logic for which exceptions warrant retry.

        Args:
            exception (Exception): The exception to evaluate.

        Returns:
            bool: True if the exception warrants a retry, False otherwise.
        """
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
        Generate a response with automatic retry handling.

        Implements fault-tolerant API communication with exponential backoff
        and configurable retry behavior for handling transient failures.

        Args:
            conversation (List[ConversationMessage]): Sequential list of prior conversation messages.

        Returns:
            str: Generated response text from the model.

        Raises:
            ValueError: When model produces empty response.
            Various API-specific exceptions from implementations.
        """

        # @retry around _inner_generate_response inside generate_response because
        # scope of self._should_retry_on_exception is not available to tenacity
        # when applied as a decorator to _generate_response directly
        @retry(
            stop=stop_any(
                stop_after_attempt(self.model_timeout.max_retries),
                stop_after_delay(self.model_timeout.total),
            ),
            wait=wait_random_exponential(
                multiplier=self.model_timeout.wait_multiplier,
                min=self.model_timeout.min_wait,
                max=self.model_timeout.max_wait,
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
            conversation (List[ConversationMessage]): List of conversation messages to format.
            add_system_prompt (bool): Whether to include system prompt at the start.

        Returns:
            List[ChatMessage]: List of formatted messages ready for API submission.
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
            debug_text (str): The content of the response to log.
        """
        logger.debug(
            "Bot Class: %s, Bot Name: %s, Bot Index: %s, %s",
            self.__class__.__name__,
            self.name,
            self.bot_index,
            debug_text,
        )
