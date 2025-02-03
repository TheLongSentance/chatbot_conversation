"""
This module contains classes for loading and validating conversation configurations from JSON files.

The module provides a robust configuration system that validates:
- Bot configuration parameters (name format, uniqueness, etc.)
- Template variable usage in prompts
- Moderator message round numbers and uniqueness
- Various required fields and their constraints

Classes:
    ChatbotParamsOptData: Optional parameters for bot configuration
    ChatbotConfigData: Configuration model for a single bot participant
    ModeratorMessage: Configuration model for round-specific moderator messages
    ConversationConfig: Configuration model for the entire conversation
    ConfigurationLoader: Loads and validates conversation configurations
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator


class ChatbotParamsOptData(BaseModel):
    """Optional parameters for bot configuration.

    Attributes:
        temperature: Controls response randomness (0.0 to 2.0)
        max_tokens: Maximum tokens for response generation
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
        bot_name: Unique identifier for the bot
        bot_prompt: Role-specific instructions for the bot
        bot_type: Type of model to use (e.g., "GPT", "CLAUDE")
        bot_version: Specific model version identifier
        bot_params_opt: Optional configuration parameters
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

    @field_validator("bot_prompt")
    @classmethod
    def validate_bot_prompt_templates(cls, v: str) -> str:
        """Validate template variables in bot prompts.

        Args:
            v: The bot prompt string to validate

        Returns:
            str: The validated bot prompt

        Raises:
            ValueError: If template variables are malformed or invalid
        """
        if v.count("{") != v.count("}"):
            raise ValueError("Mismatched template variable braces in bot_prompt")

        allowed_vars = {"bot_name", "max_tokens"}
        template_vars = re.findall(r"\{([^}]+)\}", v)
        invalid_vars = set(template_vars) - allowed_vars
        if invalid_vars:
            raise ValueError(
                f"Invalid template variables in bot_prompt: {invalid_vars}"
            )
        return v


class ModeratorMessage(BaseModel):
    """Configuration model for moderator messages at specific rounds.

    Attributes:
        round_number: The conversation round this message applies to
        content: The message content to be displayed
    """

    round_number: int = Field(gt=0, description="Round number must be positive")
    content: str = Field(
        ..., min_length=1, description="Message content cannot be empty"
    )
    display_opt: bool = Field(
        default=False,
        description="Optional flag to control message display"
    )

class ConversationConfig(BaseModel):
    """Configuration model for managing a multi-bot conversation.

    Attributes:
        author: Name of the configuration author
        conversation_seed: Initial prompt to start the discussion
        rounds: Number of conversation rounds
        core_prompt: Base instructions provided to all bots
        moderator_messages: Optional list of round-specific moderator messages
        bots: List of bot configurations

    """

    author: str = Field(..., min_length=1, description="Author name cannot be empty")
    conversation_seed: str = Field(
        ..., min_length=1, description="Conversation seed cannot be empty"
    )
    rounds: int = Field(gt=0, description="Rounds must be a positive integer")
    core_prompt: str = Field(
        ..., min_length=1, description="Core prompt cannot be empty"
    )
    moderator_messages_opt: List[ModeratorMessage] = Field(
        default_factory=list, description="Optional round-specific moderator messages"
    )
    bots: List[ChatbotConfigData] = Field(
        ..., min_length=1, description="Bots list cannot be empty"
    )

    @field_validator("core_prompt")
    @classmethod
    def validate_template_variables(cls, v: str) -> str:
        """Validate that template variables in core_prompt are properly formatted.

        Args:
            v: The core_prompt string to validate

        Returns:
            str: The validated core_prompt string

        Raises:
            ValueError: If template variables are malformed or invalid
        """
        if v.count("{") != v.count("}"):
            raise ValueError("Mismatched template variable braces in core_prompt")

        allowed_vars = {"bot_name", "max_tokens"}
        template_vars = re.findall(r"\{([^}]+)\}", v)
        invalid_vars = set(template_vars) - allowed_vars
        if invalid_vars:
            raise ValueError(f"Invalid template variables found: {invalid_vars}")
        return v

    @field_validator("bots")
    @classmethod
    def validate_unique_bot_names(
        cls, v: List[ChatbotConfigData]
    ) -> List[ChatbotConfigData]:
        """Validate that bot names are unique and properly formatted.

        Args:
            v: List of bot configurations to validate

        Returns:
            List[ChatbotConfigData]: The validated list of bot configurations

        Raises:
            ValueError: If duplicate or invalid bot names are found
        """
        bot_name_pattern = re.compile(r"^[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*$")

        # Check for invalid name formats
        invalid_names = [
            bot.bot_name for bot in v if not bot_name_pattern.match(bot.bot_name)
        ]
        if invalid_names:
            raise ValueError(
                f"Invalid bot names (must be alphanumeric with optional underscores, "
                f"not starting/ending with underscore): {', '.join(invalid_names)}"
            )

        # Check for duplicates
        names: List[str] = [bot.bot_name for bot in v]
        name_counts: Dict[str, int] = Counter(names)
        duplicates: List[str] = [
            name for name, count in name_counts.items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate bot names found in configuration: {', '.join(duplicates)}"
            )
        return v

    @field_validator("moderator_messages_opt")
    @classmethod
    def validate_moderator_messages(
        cls, v: List[ModeratorMessage], info: ValidationInfo
    ) -> List[ModeratorMessage]:
        """Validate moderator messages round numbers if present.

        Args:
            v: List of moderator messages to validate
            info: Validation context containing other field values

        Returns:
            List[ModeratorMessage]: The validated list of moderator messages

        Raises:
            ValueError: If round numbers are invalid or duplicated
        """
        if not v:  # Empty list is valid
            return v

        total_rounds: Optional[int] = info.data.get("rounds")
        if total_rounds is None:
            raise ValueError("Cannot validate moderator messages without total rounds")

        # Check round numbers are unique
        round_nums: List[int] = [msg.round_number for msg in v]
        round_counts: Dict[int, int] = Counter(round_nums)
        duplicates: List[int] = [
            num for num, count in round_counts.items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate round numbers found in moderator messages: {', '.join(map(str, duplicates))}"
            )

        # Check round numbers don't exceed total rounds
        invalid_rounds: List[int] = [num for num in round_nums if num > total_rounds]
        if invalid_rounds:
            raise ValueError(
                f"Round numbers exceed total rounds ({total_rounds}): {', '.join(map(str, invalid_rounds))}"
            )

        return v


class ConfigurationLoader:
    """Handles loading and validation of conversation configurations from JSON files.

    This class provides static methods to safely load and parse JSON configuration files,
    with proper error handling for file operations and data validation. All loaded
    configurations are validated against the defined schema and constraints.
    """

    @staticmethod
    def load_config(config_path: str) -> ConversationConfig:
        """Load and validate a conversation configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file

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
