from typing import List, Any
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

    def __init__(self, bot_model_version: str, bot_specific_system_prompt: str, bot_name: str, shared_system_prompt_prefix: str):
        """Initialize OpenAI chatbot with specific model and behavior.

        Args:
            model_version: GPT model version to use (e.g. "gpt-4")
            system_prompt: System instruction defining bot behavior
            name: Name of the chatbot
        """
        super().__init__(bot_model_version, bot_specific_system_prompt, bot_name, shared_system_prompt_prefix)
    
    def _initialize_api(self) -> Any:
        """Initialize connection to OpenAI API.

        Returns:
            OpenAI: Configured OpenAI client instance
        """
        return OpenAI()
        
    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using OpenAI's chat completion."""
        formatted_messages = self._format_message(conversation)
        completion = self.api.chat.completions.create(
            model=self.model_version,
            messages=formatted_messages,
            timeout=10
        )
        return completion.choices[0].message.content

    def _format_message(self, conversation: List[ConversationMessage]) -> List[ChatMessage]:
        """Format message history for OpenAI API submission.

        Prepends system prompt and formats all messages according to
        OpenAI's expected structure.

        Args:
            conversation: List of conversation messages to format

        Returns:
            List[ChatMessage]: Messages formatted for OpenAI API
        """
        messages: List[ChatMessage] = [{"role": "system", "content": self.system_prompt}]

        for contribution in conversation:
            role = "assistant" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "content": contribution["content"]})

        return messages


