from typing import List, Any
import google.generativeai  # type: ignore
import asyncio
from concurrent.futures import TimeoutError
from .base import ChatbotBase, GeminiMessage, ConversationMessage

class GeminiChatbot(ChatbotBase[GeminiMessage]):
    """Concrete implementation of chatbot using Google's Gemini API service.
    
    Handles initialization of Gemini model with system prompt during setup,
    message formatting specific to Gemini's expected format using 'parts' instead
    of 'content', and response generation.
    """

    def __init__(self, model_version: str, system_prompt: str, name: str):
        """Initialize Gemini chatbot with specific model and behavior."""
        super().__init__(model_version, system_prompt, name)
    
    def _initialize_api(self) -> Any:
        """Initialize connection to Gemini API with system prompt."""
        google.generativeai.configure() # type: ignore
        return google.generativeai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt
        )
        
    async def _generate_with_timeout(self, formatted_messages: List[GeminiMessage], timeout: int = 30) -> str:
        """Wrapper to call Gemini API with timeout."""
        try:
            message = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.api.generate_content(formatted_messages)
                ),
                timeout=timeout
            )
            return message.text
        except TimeoutError:
            raise TimeoutError("Gemini API call timed out after {timeout} seconds")

    def _format_message(self, conversation: List[ConversationMessage]) -> List[GeminiMessage]:
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
