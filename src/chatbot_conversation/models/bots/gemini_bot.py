"""
Chatbot implementation using Google's Gemini API service.

Provides a chatbot interface for various Gemini model versions including:
- gemini-1.5-flash
- gemini-1.5-pro
- gemini-1.0-pro-vision
- gemini-1.0-pro-002
- gemini-1.0-pro-001
"""

import json
from typing import List, Optional, TypedDict

import google.api_core.exceptions

# no stub file from google.generativeai so ignore for pylance etc
import google.generativeai  # type: ignore

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import get_logger

# Gemini 1.5 models default temperature (others may vary)
# Inherits range from 0.0 to 2.0 from the base class
# Other specify in the config file for a specific model
GEMINI_DEFAULT_TEMP = 1.0


class _GeminiMessage(TypedDict):
    """Internal type for Gemini API message format with role and content parts."""

    role: str
    parts: str


logger = get_logger("models")


@register_bot("GEMINI")
class GeminiChatbot(ChatbotBase):
    """
    Chatbot implementation using Google's Gemini API service.

    Handles API initialization, message history, system prompts, and temperature
    control. Unlike other APIs, Gemini requires system prompts to be set during
    model initialization.

    Temperature ranges:
    - gemini-1.5-flash/pro: 0.0 - 2.0 (default: 1.0)
    - gemini-1.0-pro-vision: 0.0 - 1.0 (default: 0.4)
    - gemini-1.0-pro-002: 0.0 - 2.0 (default: 1.0)
    - gemini-1.0-pro-001: 0.0 - 1.0 (default: 0.9)
    """

    def __init__(
        self,
        bot_name: str,
        bot_system_prompt: str,
        bot_model_version: str,
        bot_temp: Optional[float] = None,
    ) -> None:
        """
        Initialize a new GeminiChatbot instance.

        Args:
            bot_name: Identifier for this chatbot instance
            bot_system_prompt: Initial system instructions
            bot_model_version: Specific Gemini model version
            bot_temp: Temperature for response generation (model-specific range)
        """
        super().__init__(  # pylint: disable=duplicate-code
            bot_name=bot_name,
            bot_system_prompt=bot_system_prompt,
            bot_model_version=bot_model_version,
            bot_temp=bot_temp,
        )

        # no stub file from google.generativeai so ignore for pylance (-> pyright) etc
        google.generativeai.configure()  # pyright: ignore[reportUnknownMemberType]

        # initialise api here, but will be updated in _generate_response
        # when system prompt is set or updated since it is not passed in
        # the generate_content call for Gemini as either a parameter or
        # part of the message history

        self.api = google.generativeai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt,
            generation_config=google.generativeai.GenerationConfig(
                temperature=self.temp
            ),
        )

    def _get_default_temperature(self) -> float:
        """
        Return the default temperature setting for Gemini models.

        Returns:
            float: Default temperature value (1.0) for Gemini response generation
        """
        return GEMINI_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an API exception warrants a retry attempt.

        Args:
            exception: The caught exception to evaluate

        Returns:
            True if the operation should be retried
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
        Generate a response using the Gemini API based on conversation history.

        Handles message formatting, system prompt updates, and response generation.
        System prompt changes require API reinitialization.

        Args:
            conversation: Complete conversation history

        Returns:
            Generated response text from the model
        """
        formatted_messages = self._format_conv_for_gemini_api(conversation)

        # test if system prompt has changed and re-initialize API in order
        # to reset the system prompt for Gemini API. This is not typical
        # for other models as they include system prompt in either:
        # - as a parameter in the api call (e.g. Claude)
        # - or as part of the message history (e.g. OpenAI, Ollama)
        # for Gemini, this will happen when the system prompt is first set
        # and whenever it is updated (first round, after first round, before last)

        if self.system_prompt_needs_update:
            self.api = google.generativeai.GenerativeModel(
                model_name=self.model_version,
                system_instruction=self.system_prompt,
                generation_config=google.generativeai.GenerationConfig(
                    temperature=self.temp
                ),
            )
            self.system_prompt_updated()

        message = self.api.generate_content(formatted_messages)
        response: str = message.text
        return response

    def _format_conv_for_gemini_api(
        self, conversation: List[ConversationMessage]
    ) -> List[_GeminiMessage]:
        """
        Convert conversation messages to Gemini API format.

        Args:
            conversation: Generic conversation messages

        Returns:
            Messages formatted for Gemini API with appropriate roles
        """
        messages: List[_GeminiMessage] = []

        for contribution in conversation:
            role = "model" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "parts": contribution["content"]})

        self._log_debug(json.dumps(messages, indent=2))

        return messages
