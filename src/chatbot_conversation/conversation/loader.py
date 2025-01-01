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
        config = typing.cast(ConversationConfig, data)
        ConfigurationLoader.validate_config(config)
        return config

    @staticmethod
    def validate_config(config: ConversationConfig) -> None:
        """
        Validate the conversation configuration.

        Args:
            config (ConversationConfig): The configuration to validate.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not config.get("conversation_seed"):
            raise ValueError("Conservation seed cannot be empty")

        if config["rounds"] <= 0:
            raise ValueError("Rounds must be a positive integer")

        if not config.get("shared_prefix"):
            raise ValueError("Shared system prompt prefix cannot be empty")

        if not config.get("first_round_postfix"):
            raise ValueError("First round system prompt postfix cannot be empty")

        if not config.get("last_round_postfix"):
            raise ValueError("Last round system prompt postfix cannot be empty")

        if not config.get("bots") or len(config["bots"]) == 0:
            raise ValueError("Bots list cannot be empty")

        for bot in config["bots"]:
            if not bot["bot_name"]:
                raise ValueError("Each bot must have a non-empty 'bot_name' field")
            if not bot["bot_type"]:
                raise ValueError("Each bot must have a non-empty 'bot_type' field")
            if not bot["bot_version"]:
                raise ValueError("Each bot must have a non-empty 'bot_version' field")
            if not bot["bot_prompt"]:
                raise ValueError("Each bot must have a non-empty 'bot_prompt' field")
