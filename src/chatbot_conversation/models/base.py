"""
Core framework for building AI chatbot systems.

Provides architecture for implementing AI chatbots with:
- Language model backends and APIs
- Message history management
- System prompts and runtime parameters
- Fault-tolerant API communication
- Bot instance uniqueness
- Streaming responses
- Temperature and token controls

Implementation:
    Extend ChatbotBase to implement model backends:
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
from typing import Any, ClassVar, Final, Iterator, List, Optional, Set, TypedDict, Type

from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    stop_any,
    wait_random_exponential,
)

from chatbot_conversation.utils import (
    LOGNAME_MODEL,
    APIException,
    ChatbotException,
    ErrorSeverity,
    ModelException,
    SystemException,
    ValidationException,
    get_logger,
)

# pylint: disable=duplicate-code

# Timeout constants (in seconds)
DEFAULT_TOTAL_TIMEOUT: Final[int] = 90  # Maximum time for total trip through API
DEFAULT_API_TIMEOUT: Final[int] = 30  # Timeout for individual API calls
DEFAULT_MAX_RETRIES: Final[int] = 5  # Maximum number of retry attempts
DEFAULT_MIN_WAIT: Final[int] = 1
DEFAULT_MAX_WAIT: Final[int] = 10
DEFAULT_WAIT_MULTIPLIER: Final[float] = 1.5

# Default maximum tokens for response generation to prevent excessive response
# length. This is more of a hard limit to prevent runaway responses rather than
# something that models can be prompted to adhere to.
#
# Practical experience with shows that they may try to adhere at first if their
# system prompt tells them to stay well within their max_tokens limit, but:
#
# 1) They are not very good at counting the number of tokens in their response.
#
# 2) Few-shot learning from more recent responses seems to dominate over system
# prompt instructions and they often adopt a concensus response length that is
# similar to recent responses in the conversation.
#
# Truncation with low max_tokens values (say below 300) # are particularly
# problematic as they can cut off the response mid-sentence, however if
# the initial system prompts are explicit (e.g. "Generate one paragraph on
# the topic of...") and moderator comments are used to remind the bots of the
# token limit, they can be effective in keeping the response length down.
#
# Typically one page of writing contains around 300-400 words (in a word doc),
# so approximately 400 x 1.3 = 520 tokens. So add a further 30%+ buffer to this
# to get to approximately 700 tokens.
#
DEFAULT_MAX_TOKENS: Final[int] = 700


class ChatMessage(TypedDict):
    """
    Message format for API communication.

    Attributes:
        role: Message source ('system', 'user', 'assistant')
        content: Message text
    """

    role: str
    content: str


class ConversationMessage(TypedDict):
    """
    Internal message format for conversation tracking.

    Attributes:
        bot_index: Unique identifier of source bot
        content: Message text
    """

    bot_index: int
    content: str


@dataclass
class ChatbotTimeout:
    """
    Configuration for API timeouts and retries.

    Attributes:
        total: Maximum total time for API operation (seconds)
        api_timeout: Timeout for individual API calls (seconds)
        max_retries: Maximum retry attempts
        min_wait: Minimum delay between retries (seconds)
        max_wait: Maximum delay between retries (seconds)
        wait_multiplier: Exponential backoff multiplier
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
    Optional runtime parameters.

    Attributes:
        temperature: Response randomness (0.0-1.0)
        max_tokens: Maximum response length
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class ChatbotModel:
    """
    Model configuration.

    Attributes:
        type: Model type identifier
        version: Model version
        params_opt: Optional runtime parameters
    """

    type: str
    version: str
    params_opt: ChatbotParamsOpt = field(default_factory=ChatbotParamsOpt)


@dataclass
class ChatbotConfig:
    """
    Chatbot instance configuration.

    Attributes:
        name: Bot identifier
        system_prompt: Initial system instructions
        model: Model configuration
        timeout: Timeout settings
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
            raise ValidationException(
                message="Model type cannot be empty",
                user_message=(
                    "Model type cannot be empty, "
                    "please check conversation configuration file"
                ),
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )
        if not self.version or not self.version.strip():
            raise ValidationException(
                message="Model version cannot be empty",
                user_message=(
                    "Model version cannot be empty, "
                    "please check conversation configuration file"
                ),
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )


class ChatbotBase(ABC):
    """
    Abstract base class for building AI chatbot implementations.

    This class provides the core framework for implementing chatbot models with consistent
    behavior around initialization, configuration, error handling, and response generation.

    Core Features:
        - Unique bot instance tracking and naming validation
        - System prompt and message history management
        - Fault-tolerant API communication with configurable retries
        - Response streaming capabilities
        - Temperature and token limit controls
        - Standardized error handling and logging

    Required Implementations:
        Subclasses must implement these abstract methods:
        - available_versions(): Get supported model versions
        - _get_class_model_type(): Get model identifier
        - _generate_response(): Core response generation logic
        - _generate_stream(): Stream response chunks
        - _get_text_from_chunk(): Parse stream chunks
        - _should_retry_on_exception(): Retry policy logic
        - Temperature properties (min/max/default)

    Attributes:
        name (str): Unique identifier for this bot instance
        system_prompt (str): Initial instructions for model behavior
        bot_index (int): Unique numerical identifier
        model_type (str): Type of model backend being used
        model_version (str): Specific model version
        model_temperature (float): Current temperature setting
        model_max_tokens (int): Current max tokens limit
        model_timeout (ChatbotTimeout): Timeout and retry settings

    Raises:
        ValidationException: If configuration validation fails
        ModelException: For model-specific errors
        APIException: For API communication issues
        SystemException: For unexpected system errors
    """

    # Class Variables
    _total_count: ClassVar[int] = 0
    _used_names: ClassVar[Set[str]] = set()
    _available_versions_cache: ClassVar[Optional[List[str]]] = None

    _logger = get_logger(LOGNAME_MODEL)

    # Class Methods - Core
    @classmethod
    @abstractmethod
    def available_versions(cls) -> Optional[List[str]]:
        """
        Get available model versions for this bot type.

        Returns:
            Optional[List[str]]: List of valid model versions, or None if
            versions are not applicable/available

        Raises:
            APIError: If API call to retrieve versions fails
        """
        pass  # pylint: disable=unnecessary-pass

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

    @classmethod
    @abstractmethod
    def _retryable_exceptions(cls) -> tuple[Type[Exception], ...]:
        """
        Returns tuple of exception types that should trigger a retry.
        
        Returns:
            tuple: Exception types that warrant retry attempts
        """
        pass  # pylint: disable=unnecessary-pass

    @classmethod
    def _should_retry_on_exception(cls, exception: BaseException) -> bool:
        """
        Determine if an API call should be retried based on Claude-specific exceptions.

        Handles common Claude API errors that warrant retry attempts:
        - APIError: General API communication failures
        - APIConnectionError: Network connectivity issues
        - RateLimitError: API quota/throughput limits

        Args:
            exception: The caught exception

        Returns:
            bool: True if retry is recommended, False otherwise
        """

        # Check both direct and wrapped exceptions against retryable types
        retryable_types = cls._retryable_exceptions()
        # Logic below needed for potential nested exceptions
        if isinstance(exception, APIException):
            if isinstance(
                exception, retryable_types
            ):  # pyright: ignore[reportUnnecessaryIsInstance]
                return True
            if exception.original_error:  # checked wrapped exception
                return isinstance(exception.original_error, retryable_types)
            return False
        return isinstance(exception, retryable_types)

    # Class Methods - Model Configuration
    @classmethod
    @abstractmethod
    def _get_model_min_temperature(cls) -> float:
        """Get the minimum allowed temperature value."""
        pass  # pylint: disable=unnecessary-pass

    @classmethod
    @abstractmethod
    def _get_model_max_temperature(cls) -> float:
        """Get the maximum allowed temperature value."""
        pass  # pylint: disable=unnecessary-pass

    @classmethod
    @abstractmethod
    def _get_model_default_temperature(cls) -> float:
        """Get the default temperature value."""
        pass  # pylint: disable=unnecessary-pass

    @classmethod
    def _get_model_default_max_tokens(cls) -> int:
        """Get the default max tokens value."""
        return DEFAULT_MAX_TOKENS

    @classmethod
    def _initialise_temperature(cls, config: ChatbotConfig) -> float:
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
            else cls._get_model_default_temperature()
        )

    @classmethod
    def _initialise_max_tokens(cls, config: ChatbotConfig) -> int:
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
            else cls._get_model_default_max_tokens()
        )

    # Class Methods - Validation
    @classmethod
    def _validate_name(cls, name: str) -> None:
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
            raise ValidationException(
                message="Bot name must be a non-empty string without only whitespace",
                user_message=(
                    "Bot name must be a non-empty string without only whitespace, "
                    "please check conversation configuration file"
                ),
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )
        # Regex to match to reject special characters
        # and invalid underscore usage at start and end of the name
        if (
            re.search(r"[^a-zA-Z0-9_]", name)
            or name.startswith("_")
            or name.endswith("_")
        ):
            error_msg = (
                f"Bot name '{name}' contains "
                "invalid characters or invalid underscore usage"
            )
            raise ValidationException(
                message=error_msg,
                user_message=f"{error_msg}, please check conversation configuration file",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )
        # Validate bot name uniqueness
        if name in cls._used_names:
            error_msg = f"Bot name '{name}' is already in use by another bot instance"
            raise ValidationException(
                message=error_msg,
                user_message=f"{error_msg}, please check conversation configuration file",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )

    @classmethod
    def _validate_model_type(cls, config: ChatbotConfig) -> None:
        """
        Validate the model type against implementation.

        Args:
            config (ChatbotConfig): The configuration for the chatbot instance.

        Raises:
            ValueError: If model type doesn't match implementation
        """
        expected_type = cls._get_class_model_type()
        if config.model.type != expected_type:
            error_msg = (
                f"Invalid model type for {cls.__name__}: "
                f"got '{config.model.type}', expected '{expected_type}'"
            )
            raise ValidationException(
                message=error_msg,
                user_message=f"{error_msg}, please check conversation configuration file",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )

    @classmethod
    def _validate_model_version(cls, config: ChatbotConfig) -> None:
        """
        Validate that the configured model version is available.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If version validation fails
        """
        versions = cls.available_versions()
        if versions is not None and config.model.version not in versions:
            error_msg = (
                f"Invalid model version '{config.model.version}' for {cls.__name__}. "
                f"Available versions: {', '.join(versions)}"
            )
            raise ValidationException(
                message=error_msg,
                user_message=f"{error_msg}, please check conversation configuration file",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )

    @classmethod
    def _validate_temperature(cls, temperature: float) -> None:
        """
        Validate the temperature setting.

        Args:
            temperature (float): The temperature value to validate.

        Raises:
            ValueError: If temperature is outside valid range
        """
        minimum_temperature = cls._get_model_min_temperature()
        maximum_temperature = cls._get_model_max_temperature()
        if not minimum_temperature <= temperature <= maximum_temperature:
            error_msg = (
                f"Temperature for {cls.__name__} must be between "
                f"{minimum_temperature} and {maximum_temperature}"
            )
            raise ValidationException(
                message=error_msg,
                user_message=f"{error_msg}, please check conversation configuration file",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )

    @classmethod
    def _validate_max_tokens(cls, max_tokens: int) -> None:
        """
        Validate the max tokens setting.

        Args:
            max_tokens (int): The max tokens value to validate.

        Raises:
            ValueError: If max tokens is less than 1
        """
        if max_tokens < 1:
            error_msg = f"Max tokens for {cls.__name__} must be greater than 0"
            raise ValidationException(
                message=error_msg,
                user_message=f"{error_msg}, please check conversation configuration file",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )

    # Instance Initialization
    def __init__(
        self,
        config: ChatbotConfig,
    ) -> None:
        """
        Initialize the chatbot with model version, system prompt, and unique bot name.

        Args:
            config (ChatbotConfig): The configuration for the chatbot instance.

        """
        # Validate then set bot name
        name = config.name.strip()
        self._validate_name(name)
        self._name: str = name
        self._used_names.add(self._name)

        # Set system prompt
        self._system_prompt = config.system_prompt

        # Validate config model type against model implementation
        self._validate_model_type(config)

        # Validate model version
        self._validate_model_version(config)

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

    # Core Properties
    @property
    def name(self) -> str:
        """Get the name of the chatbot instance."""
        return self._name

    @property
    def system_prompt(self) -> str:
        """
        Get the current system prompt content.

        Returns:
            str: The current system prompt content.
        """
        return self._system_prompt

    @property
    def bot_index(self) -> int:
        """Get the unique instance identifier."""
        return self._bot_index

    # Model Properties
    @property
    def model_type(self) -> str:
        """Get the model type identifier."""
        return self._model.type

    @property
    def model_version(self) -> str:
        """Get the model version identifier."""
        return self._model.version

    @property
    def _model_api(self) -> Any:
        """Get the API client instance."""
        return self._model.api

    @_model_api.setter
    def _model_api(self, value: Any) -> None:
        """Set the API client instance."""
        self._model.api = value

    @property
    def model_temperature(self) -> float:
        """Get the current temperature setting for response generation."""
        return self._model.temperature

    @property
    def model_min_temperature(self) -> float:
        """Get the minimum allowed temperature value."""
        return self._get_model_min_temperature()

    @property
    def model_max_temperature(self) -> float:
        """Get the maximum allowed temperature value."""
        return self._get_model_max_temperature()

    @property
    def model_default_temperature(self) -> float:
        """Get the default temperature value."""
        return self._get_model_default_temperature()

    @property
    def model_max_tokens(self) -> int:
        """Get the maximum tokens setting for response generation."""
        return self._model.max_tokens

    @property
    def model_default_max_tokens(self) -> int:
        """Get the default max tokens value."""
        return self._get_model_default_max_tokens()

    @property
    def model_timeout(self) -> ChatbotTimeout:
        """Get the timeout and retry configuration."""
        return self._model.timeout

    # Core Instance Methods
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

        This method provides fault-tolerant API communication by implementing
        configurable retries with exponential backoff for transient failures.

        Args:
            conversation (List[ConversationMessage]): Sequential message history,
                where each message contains:
                - bot_index: Identifier of the source bot
                - content: The message text

        Returns:
            str: The generated response text from the model

        Raises:
            ValidationException: If validation of inputs fails
            ModelException: If model returns empty response or exceeds retries
            APIException: On API communication failures
            SystemException: On unexpected errors
            TimeoutError: If total response time exceeds configured timeout
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
            retry=retry_if_exception(self._should_retry_on_exception),
        )
        def _inner_generate_response() -> str:
            try:
                return self._generate_response(conversation)
            except Exception as e:
                if isinstance(e, ChatbotException):
                    raise
                raise APIException(
                    message=f"API error during response generation: {str(e)}",
                    user_message=(
                        "There seems to be a problem with the model API, "
                        "please review the application log and if no errors found please "
                        "try again later."
                    ),
                    severity=ErrorSeverity.ERROR,
                    original_error=e,
                ) from e

        try:
            response_content: str = _inner_generate_response()

            if response_content == "":
                raise ModelException(
                    message="Model returned an empty string response",
                    user_message=(
                        "A model response was generated but it was empty, "
                        "please review the application log for more information."
                    ),
                    severity=ErrorSeverity.ERROR,
                )
            return response_content

        except RetryError as e:
            if isinstance(e.last_attempt.exception(), TimeoutError):
                raise APIException(
                    message=f"Response generation timed out after {self.model_timeout.total}s",
                    user_message=(
                        "A model took too long to respond, "
                        "please review the application log for more information."
                    ),
                    severity=ErrorSeverity.ERROR,
                    original_error=e,
                ) from e
            raise APIException(
                message=(
                    f"Max retries ({self.model_timeout.max_retries}) "
                    "exceeded during response generation"
                ),
                user_message=(
                    "A model failed to generate a response after multiple attempts, "
                    "please review the application log for more information."
                ),
                severity=ErrorSeverity.ERROR,
                original_error=e,
            ) from e
        except Exception as e:
            if isinstance(e, ChatbotException):
                raise
            raise SystemException(
                message=f"Unexpected error during response generation: {str(e)}",
                user_message=(
                    "An unexpected error occurred during response generation, "
                    "please review the application log for more information."
                ),
                severity=ErrorSeverity.FATAL,
                original_error=e,
            ) from e

    @abstractmethod
    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[Any]:
        """Generate stream of chunks in model-specific format"""
        pass  # pylint: disable=unnecessary-pass

    def stream_response(self, conversation: list[ConversationMessage]) -> Iterator[str]:
        """
        Stream model responses incrementally as they are generated.

        This method enables real-time response handling by yielding text segments
        as soon as they become available. It handles model-specific chunk formats
        and implements the same fault-tolerance as generate_response().

        Args:
            conversation (List[ConversationMessage]): Sequential message history,
                where each message contains:
                - bot_index: Identifier of the source bot
                - content: The message text

        Yields:
            str: Text segments of the generated response as they become available

        Raises:
            ValidationException: If validation of inputs fails
            ModelException: If model returns empty chunks or exceeds retries
            APIException: On API communication failures
            SystemException: On unexpected errors
            TimeoutError: If total streaming time exceeds configured timeout
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
            retry=retry_if_exception(self._should_retry_on_exception),
        )
        def _inner_stream_response() -> Iterator[str]:
            try:
                stream = self._generate_stream(conversation)
                for chunk in stream:
                    yield self._get_text_from_chunk(chunk)

            except Exception as e:
                if isinstance(e, ChatbotException):
                    raise
                raise APIException(
                    message=f"API error during stream generation: {str(e)}",
                    user_message=(
                        "There seems to be a problem with the model API stream, "
                        "please review the application log and if no errors found please "
                        "try again later."
                    ),
                    severity=ErrorSeverity.ERROR,
                    original_error=e,
                ) from e

        try:
            yield from _inner_stream_response()

        except RetryError as e:
            if isinstance(e.last_attempt.exception(), TimeoutError):
                raise APIException(
                    message=f"Stream generation timed out after {self.model_timeout.total}s",
                    user_message=(
                        "The model stream took too long to respond, "
                        "please review the application log for more information."
                    ),
                    severity=ErrorSeverity.ERROR,
                    original_error=e,
                ) from e
            raise APIException(
                message=(
                    f"Max retries ({self.model_timeout.max_retries})"
                    "exceeded during stream generation"
                ),
                user_message=(
                    "The model stream failed after multiple attempts, "
                    "please review the application log for more information."
                ),
                severity=ErrorSeverity.ERROR,
                original_error=e,
            ) from e
        except Exception as e:
            if isinstance(e, ChatbotException):
                raise
            raise SystemException(
                message=f"Unexpected error during stream generation: {str(e)}",
                user_message=(
                    "An unexpected error occurred during stream generation, "
                    "please review the application log for more information."
                ),
                severity=ErrorSeverity.FATAL,
                original_error=e,
            ) from e

    @abstractmethod
    def _get_text_from_chunk(self, chunk: Any) -> str:
        """Extract text from a chunk in model-specific format"""
        pass  # pylint: disable=unnecessary-pass

    # Utility Methods
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
        self._logger.error(
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
        self._logger.debug(
            "Bot Class: %s, Bot Name: %s, Bot Index: %s, %s",
            self.__class__.__name__,
            self.name,
            self.bot_index,
            debug_text,
        )
