"""
This module contains the GeminiChatbot class, a concrete implementation of the ChatbotBase class,
which uses Google's Gemini API service to generate responses.

The GeminiChatbot class handles:
- Initialization of the Gemini client
- Formatting messages specific to Gemini's expected format
- Generating responses using the Gemini API
"""

import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, List

import google.generativeai

from .base import ChatbotBase, ConversationMessage, GeminiMessage


class GeminiChatbot(ChatbotBase[GeminiMessage]):
    """Concrete implementation of chatbot using Google's Gemini API service.

    Handles initialization of Gemini model with system prompt during setup,
    message formatting specific to Gemini's expected format using 'parts' instead
    of 'content', and response generation.
    """

    def __init__(
        self,  # pylint: disable=useless-parent-delegation
        bot_model_version: str,
        bot_specific_system_prompt: str,
        bot_name: str,
        shared_system_prompt_prefix: str,
    ):
        """Initialize Gemini chatbot with specific model and behavior.

        Args:
            bot_model_version: Gemini model version to use (e.g. "gemini-2")
            bot_specific_system_prompt: System instruction defining bot behavior
            bot_name: Name of the chatbot
            shared_system_prompt_prefix: Prefix for shared system instructions
        """
        super().__init__(
            bot_model_version,  # pylint: disable=useless-parent-delegation
            bot_specific_system_prompt,
            bot_name,
            shared_system_prompt_prefix,
        )

    def _initialize_api(self) -> Any:
        """Initialize connection to Gemini API with system prompt."""
        google.generativeai.configure()
        return google.generativeai.GenerativeModel(
            model_name=self.model_version, system_instruction=self.system_prompt
        )

    async def _generate_with_timeout(
        self, formatted_messages: List[GeminiMessage], timeout: int = 30
    ) -> str:
        """Wrapper to call Gemini API with timeout."""
        try:
            message = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.api.generate_content(formatted_messages)
                ),
                timeout=timeout,
            )
            if not isinstance(message.text, str):
                raise TypeError("Expected message.text to be a string")
            return message.text
        except FuturesTimeoutError as error:
            raise FuturesTimeoutError(
                f"Gemini API call timed out after {timeout} seconds"
            ) from error

    def _format_message(
        self, conversation: List[ConversationMessage]
    ) -> List[GeminiMessage]:
        """Format message history for Gemini API submission."""
        messages: List[GeminiMessage] = []

        for contribution in conversation:
            role = "model" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "parts": contribution["content"]})

        return messages

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using Gemini model with timeout."""
        formatted_messages = self._format_message(conversation)
        response = asyncio.run(self._generate_with_timeout(formatted_messages))
        return response
