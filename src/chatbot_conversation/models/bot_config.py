"""
This module contains the configuration for the chatbot modules.

BOT_MODULES: A list of strings representing the import paths of the available chatbot modules.
"""

BOT_MODULES = [
    "chatbot_conversation.models.bots.claude_bot",
    "chatbot_conversation.models.bots.gemini_bot",
    "chatbot_conversation.models.bots.ollama_bot",
    "chatbot_conversation.models.bots.openai_bot",  # Add new bots here
]
