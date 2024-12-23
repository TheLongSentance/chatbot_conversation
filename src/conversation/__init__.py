"""
This module initializes the conversation package by importing necessary classes 
and making them available for external use. It includes the following classes:
- ConversationManager: Manages the conversation flow.
- ConfigurationLoader: Loads configuration settings.
- ConversationConfig: Holds conversation configuration details.
- BotConfig: Holds bot configuration details.
"""

from .loader import BotConfig, ConfigurationLoader, ConversationConfig
from .manager import ConversationManager

__all__ = [
    "ConversationManager",
    "ConfigurationLoader",
    "ConversationConfig",
    "BotConfig",
]
