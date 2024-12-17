from typing import List, Any
import anthropic  # type: ignore
from .base import ChatbotBase, ChatMessage, ConversationMessage

class ClaudeChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using Claude's API service.
    
    Handles initialization of Claude client, message formatting specific to Claude's
    expected format, and response generation using the Claude model.
    
    Attributes:
        api: Claude client instance
        model_version: Version of Claude model to use
        system_prompt: System instruction for bot behavior
    """

    def __init__(self, bot_model_version: str, bot_specific_system_prompt: str, bot_name: str, shared_system_prompt_prefix: str):
        """Initialize Claude chatbot with specific model and behavior.

        Args:
            model_version: Claude model version to use (e.g. "claude-3")
            system_prompt: System instruction defining bot behavior
            name: Name of the chatbot
        """
        super().__init__(bot_model_version, bot_specific_system_prompt, bot_name, shared_system_prompt_prefix)
    
    def _initialize_api(self) -> Any:
        """Initialize connection to Claude API.

        Returns:
            Claude: Configured Claude client instance
        """
        return anthropic.Anthropic() # type: ignore

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using Claude's chat model."""
        formatted_messages = self._format_message(conversation)
        message = self.api.messages.create(
            model=self.model_version,
            system=self.system_prompt,
            messages=formatted_messages,
            max_tokens=500,
            timeout=10
        )
        return message.content[0].text

    def _format_message(self, conversation: List[ConversationMessage]) -> List[ChatMessage]:
        """Format message history for Claude API submission.

        Formats all messages according to Claude's expected structure.
        System prompt is not included in the message list for Claude.

        Args:
            conversation: List of conversation messages to format

        Returns:
            List[ChatMessage]: Messages formatted for Claude API
        """

        messages: List[ChatMessage] = []

        for contribution in conversation:
            role = "assistant" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "content": contribution["content"]})

        return messages
