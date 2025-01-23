"""
Google Gemini API integration for chatbot functionality.

Provides a concrete implementation of ChatbotBase for Google's Gemini models,
handling API communication, message formatting, and conversation management
with configurable parameters.

Major Classes:
    GeminiChatbot: Gemini-specific chatbot implementation

Supported Models:
    - gemini-1.5-flash: Fast inference optimized model
    - gemini-1.5-pro: Latest pro-tier model
    - gemini-1.0-pro-vision: Multimodal capabilities
    - gemini-1.0-pro-002: Enhanced reasoning
    - gemini-1.0-pro-001: Base pro model
"""

import json
from typing import Any, Iterator, List, TypedDict

import google.api_core.exceptions

# no stub file from google.generativeai so ignore for pylance etc
import google.generativeai  # type: ignore

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import get_logger

# Gemini 1.5 models default temperature (others may vary)
# Inherits range from 0.0 to 2.0 from the base class
# Other specify in the config file for a specific model
GEMINI_DEFAULT_TEMP = 1.0

MODEL_TYPE = "GEMINI"


class _GeminiMessage(TypedDict):
    """
    Internal message format for Gemini API calls.

    Attributes:
        role (str): Message source ('model' or 'user')
        parts (str): Message content text
    """

    role: str
    parts: str


logger = get_logger("models")


@register_bot(MODEL_TYPE)
class GeminiChatbot(ChatbotBase):
    """
    Chatbot implementation using Google's Gemini API service.

    Provides a concrete implementation of ChatbotBase for Gemini AI models,
    extending core functionality with Gemini-specific API integration.

    Features:
    - Gemini API authentication and communication
    - System prompt initialization handling
    - Temperature-controlled response generation
    - Automatic API reinitialization on prompt changes
    - Model-specific temperature ranges
    - Stateful conversation management

    Args:
        config (ChatbotConfig): Configuration object containing:
            - name: Bot instance identifier
            - system_prompt: Initial system behavior instructions
            - model: Model type, version and parameters
            - timeout: API communication settings

    Attributes:
        Inherits all attributes from ChatbotBase plus:
        model_api (google.generativeai.GenerativeModel): Configured Gemini API client

    Notes:
        Unlike other APIs, Gemini requires system prompts to be set during
        model initialization, necessitating API reinitialization when
        prompts change.
    """

    @classmethod
    def _get_class_model_type(cls) -> str:
        """
        Get the model type identifier for GPT models.

        Returns:
            str: "GPT" as the model type identifier
        """
        return MODEL_TYPE

    def __init__(self, config: ChatbotConfig) -> None:
        """
        Initialize Gemini chatbot with specified configuration.

        Validates configuration and sets up Gemini API client with system prompt
        and temperature settings.

        Args:
            config (ChatbotConfig): Complete bot configuration
        """
        super().__init__(config)

        # no stub file from google.generativeai so ignore for pylance (-> pyright) etc
        google.generativeai.configure()  # pyright: ignore[reportUnknownMemberType]

        # initialise api here, but will be updated in _generate_response
        # when system prompt is set or updated since it is not passed in
        # the generate_content call for Gemini as either a parameter or
        # part of the message history

        self._initialize_model_api()

    @property
    def _default_temperature(self) -> float:
        """Default temperature override"""
        return GEMINI_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an API call should be retried based on Gemini-specific exceptions.

        Handles common Gemini API errors that warrant retry attempts:
        - DeadlineExceeded: Request timeout errors
        - ServiceUnavailable: Temporary API availability issues

        Args:
            exception: The caught exception

        Returns:
            bool: True if retry is recommended, False otherwise
        """
        return isinstance(
            exception,
            (
                google.api_core.exceptions.DeadlineExceeded,
                google.api_core.exceptions.ServiceUnavailable,
            ),
        )

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response using the Gemini API.

        Formats conversation history and handles system prompt updates.
        Reinitializes API client when system prompt changes due to
        Gemini's architectural requirements.

        Args:
            conversation: Sequential list of prior conversation messages

        Returns:
            str: Generated response text from Gemini

        Raises:
            google.api_core.exceptions.DeadlineExceeded: On request timeout
            google.api_core.exceptions.ServiceUnavailable: On API unavailability
        """
        formatted_messages = self._format_conv_for_gemini_api(conversation)

        # test if system prompt has changed and re-initialize API in order
        # to reset the system prompt for Gemini API. This is not typical
        # for other models as they include system prompt in either:
        # - as a parameter in the api call (e.g. Claude)
        # - or as part of the message history (e.g. OpenAI, Ollama)
        # for Gemini, this will happen when the system prompt is first set
        # and whenever it is updated (first round, after first round, before last)

        if self.model_system_prompt_needs_update:
            self._initialize_model_api()

        message = (
            self.model_api.generate_content(  # pyright: ignore[reportUnknownMemberType]
                formatted_messages
            )
        )
        response: str = message.text
        return response

    def _format_conv_for_gemini_api(
        self, conversation: List[ConversationMessage]
    ) -> List[_GeminiMessage]:
        """
        Format conversation history for Gemini API submission.

        Converts internal message format to Gemini's expected structure with
        appropriate role assignments ('model' or 'user').

        Args:
            conversation: Complete conversation history to format

        Returns:
            List[_GeminiMessage]: Messages formatted for Gemini API submission
        """
        messages: List[_GeminiMessage] = []

        for contribution in conversation:
            role = "model" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "parts": contribution["content"]})

        self._log_debug(json.dumps(messages, indent=2))

        return messages

    def _initialize_model_api(self) -> None:
        """
        Initialize or reinitialize the Gemini API client.

        Creates a new GenerativeModel instance with current configuration settings
        including system prompt, temperature, and token limits. Called on first
        initialization and when system prompt changes.

        Note:
            Updates model_system_prompt_updated flag after initialization
        """
        self.model_api = google.generativeai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt,
            generation_config=google.generativeai.GenerationConfig(
                temperature=self.model_temperature,
                max_output_tokens=self.model_max_tokens,
            ),
        )
        self.model_system_prompt_updated()

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from a streaming response chunk.

        Args:
            chunk (Any): Response chunk from Gemini streaming API

        Returns:
            str: Extracted text content from the chunk, or empty string if not found
        """
        return chunk.text or ""

    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[Any]:
        """
        Generate streaming responses using the Gemini API.

        Reinitializes API client if system prompt has changed, then generates
        content in streaming mode.

        Args:
            conversation (list[ConversationMessage]): List of conversation messages

        Returns:
            Iterator[Any]: Iterator yielding response chunks from Gemini's streaming API
        """
        if self.model_system_prompt_needs_update:
            self._initialize_model_api()

        return self.model_api.generate_content(  # type: ignore
            self._format_conv_for_gemini_api(conversation),
            stream=True,
        )
