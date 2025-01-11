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
from typing import List

from pydantic import BaseModel, Field, ValidationError


class BotConfigData(BaseModel):
    """Pydantic model representing the configuration for a single bot."""

    bot_name: str
    bot_type: str
    bot_version: str
    bot_prompt: str


class ConversationConfig(BaseModel):
    """Pydantic model representing the configuration for a conversation."""

    author: str = Field(..., description="Author of the conversation")
    conversation_seed: str = Field(
        ..., min_length=1, description="Conversation seed cannot be empty"
    )
    rounds: int = Field(gt=0, description="Rounds must be a positive integer")
    shared_prefix: str
    first_round_postfix: str
    last_round_postfix: str
    bots: List[BotConfigData] = Field(
        ..., min_length=1, description="Bots list cannot be empty"
    )


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
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            ) from e
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file: {str(e)}", e.doc, e.pos
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error reading configuration: {str(e)}") from e

        try:
            config = ConversationConfig(**data)
            return config
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {str(e)}") from e
