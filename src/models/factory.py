from typing import Any
from .base import BotType, ChatbotBase
from .openai_bot import OpenAIChatbot
from .claude_bot import ClaudeChatbot
from .gemini_bot import GeminiChatbot
from .ollama_bot import OllamaChatbot

class ChatbotFactory:
    """Factory for creating different types of chatbots."""
    
    def create_bot(
        self, 
        bot_type: BotType, 
        model_version: str,
        system_prompt: str,
        name: str
    ) -> ChatbotBase[Any]:
        """Create a new chatbot instance based on type.
        
        Args:
            bot_type: Type of bot to create (GPT, CLAUDE, etc.)
            model_version: Model version to use
            system_prompt: System instruction for bot behavior
            name: Name of the bot
            
        Returns:
            ChatbotBase: Initialized chatbot instance
            
        Raises:
            ValueError: If bot_type is not recognized
        """
        if bot_type == BotType.GPT:
            return OpenAIChatbot(model_version, system_prompt, name)
        elif bot_type == BotType.CLAUDE:
            return ClaudeChatbot(model_version, system_prompt, name)
        elif bot_type == BotType.GEMINI:
            return GeminiChatbot(model_version, system_prompt, name)
        elif bot_type == BotType.OLLAMA:
            return OllamaChatbot(model_version, system_prompt, name)
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")
