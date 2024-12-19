"""
This module contains the OpenAIChatbot class, a concrete implementation of the ChatbotBase class,
which uses OpenAI's API service to generate responses using the GPT model.

The OpenAIChatbot class handles:
- Initialization of the OpenAI client
- Formatting messages specific to OpenAI's expected format
- Generating responses using the GPT model
"""

from typing import Any, List

from openai import OpenAI

from .base import ChatbotBase, ChatMessage, ConversationMessage


class OpenAIChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using OpenAI's API service.

    Handles initialization of OpenAI client, message formatting specific to OpenAI's
    expected format, and response generation using the GPT model.

    Attributes:
        api: OpenAI client instance
        model_version: Version of GPT model to use
        system_prompt: System instruction for bot behavior
    """

    def __init__(
        self,  # pylint: disable=useless-parent-delegation
        bot_model_version: str,
        bot_specific_system_prompt: str,
        bot_name: str,
        shared_system_prompt_prefix: str,
    ):
        """Initialize OpenAI chatbot with specific model and behavior.

        Args:
            bot_model_version (str): GPT model version to use (e.g. "gpt-4")
            bot_specific_system_prompt (str): System instruction defining bot behavior
            bot_name (str): Name of the chatbot
            shared_system_prompt_prefix (str): Prefix for shared system instructions
        """
        super().__init__(
            bot_model_version,
            bot_specific_system_prompt,
            bot_name,
            shared_system_prompt_prefix,
        )

    def _initialize_api(self) -> Any:
        """Initialize connection to OpenAI API.

        Returns:
            OpenAI: Configured OpenAI client instance
        """
        return OpenAI()

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using OpenAI's chat completion.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages

        Returns:
            str: Generated response from the model
        """
        formatted_messages = self._format_message(conversation)
        completion = self.api.chat.completions.create(
            model=self.model_version, messages=formatted_messages, timeout=10
        )
        response_content = completion.choices[0].message.content
        if not isinstance(response_content, str):
            raise ValueError("Expected response content to be a string")
        return response_content

    def _format_message(
        self, conversation: List[ConversationMessage]
    ) -> List[ChatMessage]:
        """Format message history for OpenAI API submission.

        Prepends system prompt and formats all messages according to
        OpenAI's expected structure.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages to format

        Returns:
            List[ChatMessage]: Messages formatted for OpenAI API
        """
        messages: List[ChatMessage] = [
            {"role": "system", "content": self.system_prompt}
        ]

        for contribution in conversation:
            role = (
                "assistant" if contribution["bot_index"] == self.bot_index else "user"
            )
            messages.append({"role": role, "content": contribution["content"]})

        return messages
