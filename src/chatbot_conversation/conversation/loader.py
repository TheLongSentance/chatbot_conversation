"""
This module contains the ConfigurationLoader class, which is responsible for loading
conversation configurations from a JSON file.

The ConfigurationLoader class handles:
- Loading and parsing the JSON configuration file
- Providing the configuration data in a structured format

Classes:
    BotConfigData: Typed dictionary representing the configuration for a single bot.
    ConversationConfig: Typed dictionary representing the configuration for a conversation.
    ConfigurationLoader: Class responsible for loading conversation configurations from a JSON file.
"""

import json
import typing
from typing import List, TypedDict


class BotConfigData(TypedDict):
    """Typed dictionary representing the configuration for a single bot."""

    bot_name: str
    bot_type: str
    bot_version: str
    bot_prompt: str


class ConversationConfig(TypedDict):
    """Typed dictionary representing the configuration for a conversation."""

    conversation_seed: str
    rounds: int
    shared_prefix: str
    first_round_postfix: str
    last_round_postfix: str
    bots: List[BotConfigData]


class ConfigurationLoader:  # pylint: disable=too-few-public-methods
    """Class responsible for loading conversation configurations from a JSON file."""

    @staticmethod
    def load_config(config_path: str) -> ConversationConfig:
        """
        Load conversation configuration from JSON file.

        Args:
            config_path (str): Path to JSON configuration file.

        Returns:
            ConversationConfig: Loaded configuration.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return typing.cast(ConversationConfig, data)
