"""
This module contains the OllamaChatbot class, a concrete implementation of the ChatbotBase class,
which uses Ollama's API service to generate responses.

The OllamaChatbot class handles:
- Initialization of the Ollama client
- Formatting messages specific to Ollama's expected format
- Generating responses using the Ollama API
"""
from typing import List, Any
import ollama  # type: ignore
from .base import ChatbotBase, ChatMessage, ConversationMessage

class OllamaChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using Ollama's API service.

    Handles initialization of Ollama client, message formatting specific to Ollama's
    expected format, and response generation.
    """

    def __init__(self, bot_model_version: str, # pylint: disable=useless-parent-delegation
                 bot_specific_system_prompt: str,
                 bot_name: str,
                 shared_system_prompt_prefix: str):
        """Initialize Ollama chatbot with specific model and behavior.

        Args:
            bot_model_version: Ollama model version to use (e.g. "ollama-1")
            bot_specific_system_prompt: System instruction defining bot behavior
            bot_name: Name of the chatbot
            shared_system_prompt_prefix: Prefix for shared system instructions
        """
        super().__init__(bot_model_version,
                         bot_specific_system_prompt,
                         bot_name,
                         shared_system_prompt_prefix)

    def _initialize_api(self) -> Any:
        """Initialize connection to Ollama API."""
        return None  # Ollama doesn't need initialization

    def _format_message(self, conversation: List[ConversationMessage]) -> List[ChatMessage]:
        """Format message history for Ollama API submission."""
        messages: List[ChatMessage] = [{"role": "system", "content": self.system_prompt}]

        for contribution in conversation:
            role = "assistant" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "content": contribution["content"]})

        return messages

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using Ollama's chat model."""
        formatted_messages = self._format_message(conversation)
        response = ollama.chat(                 # type: ignore
            model=self.model_version,
            messages=formatted_messages
        )
        return response['message']['content']   # type: ignore
