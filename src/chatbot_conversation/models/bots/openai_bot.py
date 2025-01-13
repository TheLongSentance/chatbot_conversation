"""
This module contains the OpenAIChatbot class, a concrete implementation of the ChatbotBase class,
which uses OpenAI's API service to generate responses using the GPT model.

The OpenAIChatbot class handles:
- Initialization of the OpenAI client
- Formatting messages specific to OpenAI's expected format
- Generating responses using the GPT model

Classes:
    OpenAIChatbot: Concrete implementation of chatbot using OpenAI's API service.
"""

from typing import List

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot


@register_bot("GPT")
class OpenAIChatbot(ChatbotBase):
    """
    Concrete implementation of chatbot using OpenAI's API service.

    Handles initialization of OpenAI client, message formatting specific to OpenAI's
    expected format, and response generation using the GPT model.

    Attributes:
        api: OpenAI client instance.
        model_version: Version of GPT model to use.
        system_prompt: System instruction for bot behavior.
    """

    def __init__(
        self,
        bot_model_version: str,
        bot_system_prompt: str,
        bot_name: str,
    ) -> None:
        """
        Initialize the OpenAIChatbot with model version, system prompt, and bot name.

        Args:
            bot_model_version (str): The version of the bot model
            bot_system_prompt (str): The system prompt for the bot
            bot_name (str): The name of the bot
        """
        super().__init__(
            bot_model_version=bot_model_version,
            bot_system_prompt=bot_system_prompt,
            bot_name=bot_name,
        )

        self.api = OpenAI()

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Check if the exception is a network error or timeout.

        Args:
            exception (Exception): The exception to check.

        Returns:
            bool: True if the exception is a network error or timeout, False otherwise.
        """
        return isinstance(exception, (APIError, APIConnectionError, RateLimitError))

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate response using OpenAI's chat completion.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages.

        Returns:
            str: Generated response from the model.
        """
        response_content: str = ""
        formatted_messages = self._format_conv_for_api_util(conversation)
        completion = self.api.chat.completions.create(
            model=self.model_version,
            messages=formatted_messages,
            timeout=self.timeout.api_timeout,
        )
        response_content = completion.choices[0].message.content
        return response_content
