"""
Subpackage containing the conversation module of the chatbot. 
It is responsible for handling the conversation between the user and the chatbot. 
"""

from chatbot_conversation.conversation.loader import (
    ERROR_EMPTY_BOTS,
    ERROR_EMPTY_FIELD,
    ERROR_EMPTY_PREFIX,
    ERROR_EMPTY_SEED,
    ERROR_INVALID_ROUNDS,
    ConfigurationLoader,
    ConversationConfig,
)
from chatbot_conversation.conversation.manager import ConversationManager

__all__ = [
    "ERROR_EMPTY_BOTS",
    "ERROR_EMPTY_FIELD",
    "ERROR_EMPTY_PREFIX",
    "ERROR_EMPTY_SEED",
    "ERROR_INVALID_ROUNDS",
    "ConfigurationLoader",
    "ConversationConfig",
    "ConversationManager",
]
