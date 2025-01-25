"""
Core framework for building AI chatbot systems.

Provides a comprehensive architecture for implementing AI chatbots with support for:
- Multiple language model backends and APIs
- Message format standardization and history management
- Configurable system prompts and runtime parameters
- Fault-tolerant API communication with retry logic
- Bot instance uniqueness and validation
- Streaming response capabilities
- Temperature and token limit controls

Key Components:
- Message Classes: ChatMessage (API format) and ConversationMessage (internal format)
- Configuration Classes: ChatbotTimeout, ChatbotParamsOpt, ChatbotModel, ChatbotConfig
- Base Implementation: ChatbotBase abstract class with core functionality

Usage:
    Extend ChatbotBase to implement specific model backends:
    ```python
    class MyModelBot(ChatbotBase):
        @property
        def model_min_temperature(self) -> float:
            return 0.0
            
        @property
        def model_max_temperature(self) -> float:
            return 1.0
            
        @property
        def model_default_temperature(self) -> float:
            return 0.7
            
        @classmethod
        def _get_class_model_type(cls) -> str:
            return "my_model"
        
        def _generate_response(self, conversation: List[ConversationMessage]) -> str:
            # Implement model-specific response generation
    ```
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator, List, Optional, Set, TypedDict

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
DEFAULT_TOTAL_TIMEOUT = 90  # Maximum time for total trip through API
DEFAULT_API_TIMEOUT = 30  # For per try of API call if child classes api parameter
DEFAULT_MAX_RETRIES = 5  # Maximum number of retry attempts
DEFAULT_MIN_WAIT = 1
DEFAULT_MAX_WAIT = 10
DEFAULT_WAIT_MULTIPLIER = 1.5

# Default maximum tokens for response generation
DEFAULT_MAX_TOKENS = 300

logger = get_logger("models")


class ChatMessage(TypedDict):
    """
    Standardized message format for API communication.

    Used to structure messages in a format compatible with common LLM APIs,
    maintaining consistent role attribution and content formatting.

    Attributes:
        role (str): Message source identifier ('system', 'user', 'assistant')
        content (str): Actual message text content
    """

    role: str
    content: str


class ConversationMessage(TypedDict):
    """
    Internal message format for conversation tracking.

    Manages message history with bot attribution for multi-bot conversations.
    Used for state management before converting to API-specific formats.

    Attributes:
        bot_index (int): Unique identifier of the source bot
        content (str): Message text content
    """

    bot_index: int
    content: str


@dataclass
class ChatbotTimeout:
    """
    Configuration for API communication timeouts and retry behavior.

    Manages timing parameters for reliable API operations with exponential
    backoff retry logic for handling transient failures.

    Attributes:
        total (int): Maximum total time in seconds for API operation completion
        api_timeout (int): Timeout in seconds for individual API calls
        max_retries (int): Maximum number of retry attempts
        min_wait (int): Minimum delay in seconds between retries
        max_wait (int): Maximum delay in seconds between retries
        wait_multiplier (float): Exponential backoff multiplier for retry delays
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
    Runtime optional parameters for chatbot instances.

    Attributes:
        temperature (Optional[float]): Temperature setting for response generation
        max_tokens (Optional[int]): Maximum tokens for response generation
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class ChatbotModel:
    """
    Runtime configuration for model information in chatbot instances.

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


@dataclass
class _Model:
    """
    Internal container for model-related attributes.
    """

    type: str
    version: str
    timeout: ChatbotTimeout
    temperature: float
    max_tokens: int
    api: Any = None

    def __post_init__(self) -> None:
        """Validate model type and version after initialization."""
        if not self.type or not self.type.strip():
            raise ValueError("Model type cannot be empty")
        if not self.version or not self.version.strip():
            raise ValueError("Model version cannot be empty")


class ChatbotBase(ABC):
    """
    Abstract base class for chatbot implementations.

    Provides core functionality and structure for building model-specific chatbots:
    - Unique bot instance management
    - System prompt handling
    - Conversation state tracking
    - API communication with retry logic
    - Response streaming capabilities
    - Comprehensive error handling
    - Temperature and token limit management

    Implementation Requirements:
        Subclasses must implement:
        - _get_class_model_type(): Define the model type identifier
        - _generate_response(): Implement model-specific response generation
        - _should_retry_on_exception(): Define retry logic for specific errors
        - _get_text_from_chunk(): Extract text from streaming response chunks
        - _generate_stream(): Implement model-specific response streaming
        - model_min_temperature: Property defining minimum temperature value
        - model_max_temperature: Property defining maximum temperature value
        - model_default_temperature: Property defining default temperature value

    Class Attributes:
        _total_count (int): Total number of bot instances created
        _used_names (Set[str]): Tracking of assigned bot names

    Instance Attributes:
        name (str): Unique bot identifier
        system_prompt (str): Current system instructions
        model_type (str): Model backend identifier
        model_version (str): Model version identifier
        model_temperature (float): Response randomness setting (0.0 to 1.0)
        model_max_tokens (int): Response length limit
        bot_index (int): Unique numerical identifier
        model_api (Any): API client instance reference
        model_timeout (ChatbotTimeout): Timeout and retry configuration

    Raises:
        ValueError: On invalid configuration (name conflicts, invalid parameters)
    """

    _total_count: ClassVar[int] = 0  # Class variable to track total instances
    _used_names: ClassVar[Set[str]] = set()  # Class variable to track used names

    @classmethod
    def get_total_bots(cls) -> int:
        """
        Get the total number of bot instances.

        Returns:
            int: The total number of bot instances created.
        """
        return cls._total_count

    @classmethod
    @abstractmethod
    def _get_class_model_type(cls) -> str:
        """
        Get the model type identifier for the chatbot.

        Returns:
            str: The model type identifier for the chatbot.
        """
        pass  # pylint: disable=unnecessary-pass

    def __init__(
        self,
        config: ChatbotConfig,
    ) -> None:
        """
        Initialize the chatbot with model version, system prompt, and unique bot name.

        Args:
            config (ChatbotConfig): The configuration for the chatbot instance.

        Raises:
            ValueError: If the bot name is empty or whitespace-only,
                      contains invalid characters,
                      already in use by another instance,
                      if model type doesn't match implementation,
                      if temperature is outside valid range,
                      or if max tokens is less than 1
        """
        # Validate then set bot name
        name = config.name.strip()
        self._validate_name(name)
        self._name: str = name
        self._used_names.add(self._name)

        # Use the public setter for system prompt
        # - will also validate and set update flag
        self.system_prompt = config.system_prompt

        # Validate config model type against model implementation
        self._validate_model_type(config)

        # Validate temperature and set to default if not provided
        temperature = self._initialise_temperature(config)
        self._validate_temperature(temperature)

        # Validate max tokens and set to default if not provided
        max_tokens = self._initialise_max_tokens(config)
        self._validate_max_tokens(max_tokens)

        # Initialize model container
        self._model = _Model(
            type=config.model.type,
            version=config.model.version,
            timeout=config.timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Initialize bot index and update class count
        ChatbotBase._total_count += 1
        self._bot_index: int = ChatbotBase._total_count

    @property
    def name(self) -> str:
        """Get the name of the chatbot instance."""
        return self._name

    def _validate_name(self, name: str) -> None:
        """
        Validate the bot name.

        Args:
            name (str): The name to validate.

        Returns:
            None

        Raises:
            ValueError: If the name is empty, contains invalid characters,
                      or is already in use by another instance
        """
        if not name:  # Empty or whitespace-only
            raise ValueError(
                "Bot name must be a non-empty string without only whitespace"
            )
        # Regex to match to reject special characters
        # and invalid underscore usage at start and end of the name
        if (
            re.search(r"[^a-zA-Z0-9_]", name)
            or name.startswith("_")
            or name.endswith("_")
        ):
            raise ValueError(
                f"Bot name '{name}' contains "
                "invalid characters or invalid underscore usage"
            )

        # Validate bot name uniqueness
        if name in self._used_names:
            raise ValueError(
                f"Bot name '{name}' is already in use by another bot instance"
            )

    @property
    def model_type(self) -> str:
        """Get the model type identifier."""
        return self._model.type

    @property
    def model_version(self) -> str:
        """Get the model version identifier."""
        return self._model.version

    @property
    @abstractmethod
    def model_min_temperature(self) -> float:
        """Get the minimum allowed temperature value."""
        pass  # pylint: disable=unnecessary-pass

    @property
    @abstractmethod
    def model_max_temperature(self) -> float:
        """Get the maximum allowed temperature value."""
        pass  # pylint: disable=unnecessary-pass

    @property
    @abstractmethod
    def model_default_temperature(self) -> float:
        """Get the default temperature value."""
        pass  # pylint: disable=unnecessary-pass

    @property
    def model_default_max_tokens(self) -> int:
        """Get the default max tokens value."""
        return DEFAULT_MAX_TOKENS

    @property
    def model_temperature(self) -> float:
        """Get the current temperature setting for response generation."""
        return self._model.temperature

    def _initialise_temperature(self, config: ChatbotConfig) -> float:
        """
        Get the initial temperature value from config or default.

        Args:
            config (ChatbotConfig): The configuration for the chatbot instance.

        Returns:
            float: The initial temperature value to use.
        """
        return (
            config.model.params_opt.temperature
            if config.model.params_opt.temperature is not None
            else self.model_default_temperature
        )

    def _validate_temperature(self, temperature: float) -> None:
        """
        Validate the temperature setting.

        Args:
            temperature (float): The temperature value to validate.

        Raises:
            ValueError: If temperature is outside valid range
        """
        if not self.model_min_temperature <= temperature <= self.model_max_temperature:
            raise ValueError(
                f"Temperature for {self.__class__.__name__} must be between "
                f"{self.model_min_temperature} and {self.model_max_temperature}"
            )

    @property
    def model_max_tokens(self) -> int:
        """Get the maximum tokens setting for response generation."""
        return self._model.max_tokens

    def _validate_max_tokens(self, max_tokens: int) -> None:
        """
        Validate the max tokens setting.

        Args:
            max_tokens (int): The max tokens value to validate.

        Raises:
            ValueError: If max tokens is less than 1
        """
        if max_tokens < 1:
            raise ValueError(
                f"Max tokens for {self.__class__.__name__} must be greater than 0"
            )

    def _initialise_max_tokens(self, config: ChatbotConfig) -> int:
        """
        Get the initial max tokens value from config or default.

        Args:
            config (ChatbotConfig): The configuration for the chatbot instance.

        Returns:
            int: The initial max tokens value to use.
        """
        return (
            config.model.params_opt.max_tokens
            if config.model.params_opt.max_tokens is not None
            else self.model_default_max_tokens
        )

    @property
    def model_timeout(self) -> ChatbotTimeout:
        """Get the timeout and retry configuration."""
        return self._model.timeout

    @property
    def bot_index(self) -> int:
        """Get the unique instance identifier."""
        return self._bot_index

    @property
    def model_api(self) -> Any:
        """Get the API client instance."""
        return self._model.api

    @model_api.setter
    def model_api(self, value: Any) -> None:
        """Set the API client instance."""
        self._model.api = value

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

        Raises:
            ValueError: If the system prompt is empty or contains only whitespace
        """
        if not value or not value.strip():
            raise ValueError("System prompt cannot be empty")
        self._system_prompt = value

    def _validate_model_type(self, config: ChatbotConfig) -> None:
        """
        Validate the model type against implementation.

        Args:
            config (ChatbotConfig): The configuration for the chatbot instance.

        Raises:
            ValueError: If model type doesn't match implementation
        """
        expected_type = self._get_class_model_type()
        if config.model.type != expected_type:
            raise ValueError(
                f"Invalid model type for {self.__class__.__name__}: "
                f"got '{config.model.type}', expected '{expected_type}'"
            )

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
        Generate a model response with automatic retry handling.

        Implements fault-tolerant API communication using configured retry
        parameters and exponential backoff for transient failures.

        Args:
            conversation: Sequential list of prior messages as ConversationMessage

        Returns:
            Generated response text from the model

        Raises:
            ValueError: If model produces empty response
            Exception: Model-specific API exceptions from implementations
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

    @abstractmethod
    def _get_text_from_chunk(self, chunk: Any) -> str:
        """Extract text from a chunk in model-specific format"""
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[Any]:
        """Generate stream of chunks in model-specific format"""
        pass  # pylint: disable=unnecessary-pass

    def stream_response(self, conversation: list[ConversationMessage]) -> Iterator[str]:
        """
        Stream model responses as they are generated.

        Provides incremental response chunks for real-time output handling.
        Converts model-specific chunk formats to plain text.
        Implements fault-tolerant API communication using configured retry
        parameters and exponential backoff for transient failures.

        Args:
            conversation: Sequential list of prior messages as ConversationMessage

        Yields:
            Text segments of the generated response as they become available

        Raises:
            Exception: Model-specific API exceptions from implementations
        """
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
        def _inner_stream_response() -> Iterator[str]:
            stream = self._generate_stream(conversation)
            for chunk in stream:
                yield self._get_text_from_chunk(chunk)

        yield from _inner_stream_response()

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
