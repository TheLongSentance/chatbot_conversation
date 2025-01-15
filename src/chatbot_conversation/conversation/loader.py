"""
This module contains classes for loading and validating conversation configurations from JSON files.

Classes:
    ChatbotParamsOptData: Optional parameters for bot configuration.
    ChatbotConfigData: Configuration model for a single bot participant.
    ConversationConfig: Configuration model for the entire conversation.
    ConfigurationLoader: Loads and validates conversation configurations.
"""

import json
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError


class ChatbotParamsOptData(BaseModel):
    """Optional parameters for bot configuration.

    Attributes:
        temperature (float | None): Temperature parameter for response randomness
        max_tokens (int | None): Maximum tokens for response generation
    """
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature parameter for response randomness",
        ge=0.0,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for response generation",
        gt=0,
    )


class ChatbotConfigData(BaseModel):
    """Configuration model for a single bot participant in the conversation.

    Attributes:
        bot_name (str): Unique identifier for the bot
        bot_prompt (str): Initial system prompt for the bot
        bot_type (str): Type/model of the bot (e.g., 'gpt-4', 'claude')
        bot_version (str): Version identifier for the bot
        bot_params_opt (ChatbotParamsOptData): Optional parameters for the bot
    """

    bot_name: str = Field(..., min_length=1, description="Bot name cannot be empty")
    bot_prompt: str = Field(..., min_length=1, description="Bot prompt cannot be empty")
    bot_type: str = Field(..., min_length=1, description="Bot type cannot be empty")
    bot_version: str = Field(
        ..., min_length=1, description="Bot version cannot be empty"
    )
    bot_params_opt: ChatbotParamsOptData = Field(
        default_factory=ChatbotParamsOptData,
        description="Optional parameters for the bot",
    )


class ConversationConfig(BaseModel):
    """Configuration model for managing a multi-bot conversation.

    Attributes:
        author (str): Creator or owner of the conversation
        conversation_seed (str): Initial topic or context for the conversation
        rounds (int): Number of conversation rounds to execute
        shared_prefix (str): Common prefix added to all bot prompts
        first_round_postfix (str): Text appended to the first round's prompt
        last_round_postfix (str): Text appended to the final round's prompt
        bots (List[ChatbotConfigData]): List of bot configurations for participants
    """

    author: str = Field(..., min_length=1, description="Author of the conversation")
    conversation_seed: str = Field(
        ..., min_length=1, description="Conversation seed cannot be empty"
    )
    rounds: int = Field(gt=0, description="Rounds must be a positive integer")
    shared_prefix: str  # Can be empty string
    first_round_postfix: str  # Can be empty string
    last_round_postfix: str  # Can be empty string
    bots: List[ChatbotConfigData] = Field(
        ..., min_length=1, description="Bots list cannot be empty"
    )


class ConfigurationLoader:  # pylint: disable=too-few-public-methods
    """Handles loading and validation of conversation configurations from JSON files.

    This class provides static methods to safely load and parse JSON configuration files,
    with proper error handling for file operations and data validation.
    """

    @staticmethod
    def load_config(config_path: str) -> ConversationConfig:
        """Load and validate a conversation configuration from a JSON file.

        Args:
            config_path (str): Path to the JSON configuration file

        Returns:
            ConversationConfig: Validated configuration object

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            ValueError: If the configuration data fails validation
            RuntimeError: For other unexpected errors during loading
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
