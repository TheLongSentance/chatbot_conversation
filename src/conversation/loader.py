"""
This module contains the ConfigurationLoader class, which is responsible for loading
conversation configurations from a JSON file.

The ConfigurationLoader class handles:
- Loading and parsing the JSON configuration file
- Providing the configuration data in a structured format
"""
from typing import List, TypedDict
import json
import typing

class BotConfig(TypedDict):
    """Typed dictionary representing the configuration for a single bot.

    Attributes:
        bot_name: Name of the bot
        bot_type: Type of the bot (e.g., "openai")
        bot_model_version: Version of the bot model to use
        bot_specific_system_prompt: System prompt specific to the bot
    """
    bot_name: str
    bot_type: str
    bot_model_version: str
    bot_specific_system_prompt: str

class ConversationConfig(TypedDict):
    """Typed dictionary representing the configuration for a conversation.

    Attributes:
        conversation_seed: Seed text to start the conversation
        rounds: Number of rounds in the conversation
        shared_system_prompt_prefix: Prefix for shared system instructions
        bots: List of bot configurations
    """
    conversation_seed: str
    rounds: int
    shared_system_prompt_prefix: str
    bots: List[BotConfig]

class ConfigurationLoader:      # pylint: disable=too-few-public-methods
    """Class responsible for loading conversation configurations from a JSON file."""

    @staticmethod
    def load_config(config_path: str) -> ConversationConfig:
        """Load conversation configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            ConversationConfig: Loaded configuration
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return typing.cast(ConversationConfig, data)
