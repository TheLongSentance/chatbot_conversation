"""
This module contains classes and functions for loading and validating conversation configurations from JSON files.

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

Functions:
    load_conversation_config: Load and validate a conversation configuration from a JSON file
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

BOT_NAME_PATTERN = r"^[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*$"
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
ALLOWED_TEMPLATE_VARS = {"bot_name", "max_tokens"}
TEMPLATE_VARS_PATTERN = r"\{([^}]+)\}"

class BaseConfigModel(BaseModel):
    """Base configuration model with strict validation."""
    model_config = ConfigDict(
        extra='forbid',  # Prevent unknown fields
        str_strip_whitespace=True,  # Strip whitespace from strings
        frozen=True  # Make configs immutable after creation
    )

class ChatbotParamsOptData(BaseConfigModel):
    """Optional parameters for bot configuration.

    Attributes:
        temperature (Optional[float]): Controls response randomness (0.0 to 2.0)
        max_tokens (Optional[int]): Maximum tokens for response generation
    """

    temperature: Optional[float] = Field(
        default=None,
        description="Temperature parameter for response randomness",
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
        examples=[0.0, 0.5, 1.0, 1.5, 2.0],
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for response generation",
        gt=0,
    )


class ChatbotConfigData(BaseConfigModel):
    """Configuration model for a single bot participant in the conversation.

    Attributes:
        bot_name (str): Unique identifier for the bot
        bot_prompt (str): Role-specific instructions for the bot
        bot_type (str): Type of model to use (e.g., "GPT", "CLAUDE")
        bot_version (str): Specific model version identifier
        bot_params_opt (ChatbotParamsOptData): Optional configuration parameters
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
            v (str): The bot prompt string to validate

        Returns:
            str: The validated bot prompt

        Raises:
            ValueError: If template variables are malformed or invalid
        """
        if v.count("{") != v.count("}"):
            raise ValueError("Mismatched template variable braces in bot_prompt")

        allowed_vars = ALLOWED_TEMPLATE_VARS
        template_vars = re.findall(r"\{([^}]+)\}", v)
        invalid_vars = set(template_vars) - allowed_vars
        if invalid_vars:
            raise ValueError(
                f"Invalid template variables in bot_prompt: {invalid_vars}"
            )
        return v


class ModeratorMessage(BaseConfigModel):
    """Configuration model for moderator messages at specific rounds.

    Attributes:
        round_number (int): The conversation round this message applies to
        content (str): The message content to be displayed
        display_opt (bool): Optional flag to control message display
    """

    round_number: int = Field(gt=0, description="Round number must be positive")
    content: str = Field(
        ..., min_length=1, description="Message content cannot be empty"
    )
    display_opt: bool = Field(
        default=False, description="Optional flag to control message display"
    )


class ConversationConfig(BaseConfigModel):
    """Configuration model for managing a multi-bot conversation.

    Attributes:
        author (str): Name of the configuration author
        conversation_seed (str): Initial prompt to start the discussion
        rounds (int): Number of conversation rounds
        core_prompt (str): Base instructions provided to all bots
        moderator_messages_opt (List[ModeratorMessage]): Optional list of round-specific moderator messages
        bots (List[ChatbotConfigData]): List of bot configurations
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
            v (str): The core_prompt string to validate

        Returns:
            str: The validated core_prompt string

        Raises:
            ValueError: If template variables are malformed or invalid
        """
        if v.count("{") != v.count("}"):
            raise ValueError("Mismatched template variable braces in core_prompt")

        allowed_vars = {"bot_name", "max_tokens"}
        template_vars = re.findall(TEMPLATE_VARS_PATTERN, v)
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
            v (List[ChatbotConfigData]): List of bot configurations to validate

        Returns:
            List[ChatbotConfigData]: The validated list of bot configurations

        Raises:
            ValueError: If duplicate or invalid bot names are found
        """
        bot_name_pattern = re.compile(BOT_NAME_PATTERN)

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
            v (List[ModeratorMessage]): List of moderator messages to validate
            info (ValidationInfo): Validation context containing other field values

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
                "Duplicate round numbers found in moderator messages: "
                f"{', '.join(map(str, duplicates))}"
            )

        # Check round numbers don't exceed total rounds
        invalid_rounds: List[int] = [num for num in round_nums if num > total_rounds]
        if invalid_rounds:
            raise ValueError(
                "Round numbers exceed total rounds ({total_rounds}): "
                f"{', '.join(map(str, invalid_rounds))}"
            )

        return v

def load_conversation_config(config_path: str) -> ConversationConfig:
    """Load and validate a conversation configuration from a JSON file.
    
    Args:
        config_path: Must be a .json file
    """
    if not config_path.endswith('.json'):
        raise ValueError("Configuration file must be a .json file")
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
