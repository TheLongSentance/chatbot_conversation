"""
This module provides functionality for constructing and adjusting system prompts 
for Chatbot instances.
"""

from typing import Final

from chatbot_conversation.conversation.loader import ChatbotConfigData
from chatbot_conversation.models.base import DEFAULT_MAX_TOKENS, ChatbotBase

BOT_NAME_VARIABLE_PLACEHOLDER: Final[str] = "bot_name"
MAX_TOKENS_VARIABLE_PLACEHOLDER: Final[str] = "max_tokens"

# Multiplier for buffer tokens to avoid hitting the max token limit
MAX_TOKENS_BUFFER_MULTIPLIER: Final[float] = 0.66


def replace_variables(text: str, variables: dict[str, str]) -> str:
    """
    Replace placeholders in the text with the provided variable values.

    Args:
        text (str): The text containing placeholders.
        variables (dict): A dictionary with variable names as keys and their
        corresponding values. e.g. {"bot_name": "GPT-4", "max_tokens": "100"}
        will replace "{bot_name}" with "GPT-4" and "{max_tokens}" with "100".

    Returns:
        str: The text with placeholders replaced by their values.

    Example:
        >>> replace_variables("Hello, {bot_name}!", {"bot_name": "GPT-4"})
        'Hello, GPT-4!'
    """
    for key, value in variables.items():
        placeholder = f"{{{key}}}"
        text = text.replace(placeholder, str(value))
    return text


def add_suffix(bot: ChatbotBase, additional_prompt: str) -> None:
    """
    Append additional text to the system prompt.

    Args:
        bot (ChatbotBase): The chatbot instance whose system prompt will be modified.
        additional_prompt (str): The text to append.

    Example:
        >>> bot = ChatbotBase(system_prompt="Hello")
        >>> add_suffix(bot, ", world!")
        >>> bot.system_prompt
        'Hello, world!'
    """
    if additional_prompt:
        bot.system_prompt += additional_prompt


def remove_suffix(bot: ChatbotBase, text_to_remove: str) -> None:
    """
    Remove specific text from the end of the system prompt.

    Args:
        bot (ChatbotBase): The chatbot instance whose system prompt will be modified.
        text_to_remove (str): The text to remove.

    Example:
        >>> bot = ChatbotBase(system_prompt="Hello, world!")
        >>> remove_suffix(bot, ", world!")
        >>> bot.system_prompt
        'Hello'
    """
    if text_to_remove and bot.system_prompt.endswith(text_to_remove):
        bot.system_prompt = bot.system_prompt[: -len(text_to_remove)]


def construct_system_prompt(shared_prefix: str, bot_config: ChatbotConfigData) -> str:
    """
    Construct the system prompt for a bot based on the shared prefix and bot configuration.

    Args:
        shared_prefix (str): The shared prefix for the system prompt.
        bot_config (ChatbotConfigData): The configuration for the bot.

    Returns:
        str: The constructed system prompt.

    Example:
        >>> bot_config = ChatbotConfigData(bot_name="Bot1", bot_prompt="You are Bot1.")
        >>> construct_system_prompt("Shared prefix: ", bot_config)
        'Shared prefix: You are Bot1.'
    """
    if bot_config.bot_params_opt.max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    else:
        max_tokens = bot_config.bot_params_opt.max_tokens

    max_tokens = max_tokens_for_prompt(max_tokens)

    bot_system_prompt = shared_prefix + bot_config.bot_prompt
    bot_system_prompt = replace_variables(
        bot_system_prompt,
        {
            BOT_NAME_VARIABLE_PLACEHOLDER: bot_config.bot_name,
            MAX_TOKENS_VARIABLE_PLACEHOLDER: str(max_tokens),
        },
    )

    return bot_system_prompt


def max_tokens_for_prompt(max_tokens: int) -> int:
    """
    Calculate the maximum number of tokens to use for a system prompt.

    Args:
        max_tokens (int): The maximum number of tokens allowed for the model.

    Returns:
        int: The adjusted maximum number of tokens for the system prompt.
    """
    return int(max_tokens * MAX_TOKENS_BUFFER_MULTIPLIER)


class SuffixManager:
    """Manages temporary suffixes for bot system prompts during conversation rounds."""

    def __init__(self) -> None:
        self._bot_suffixes: dict[ChatbotBase, str] = {}

    def setup_round_suffix(self, bot: ChatbotBase, suffix_template: str) -> None:
        """
        Calculate and apply a round-specific suffix to a bot's system prompt.

        Args:
            bot: The bot to setup the suffix for
            suffix_template: Template string containing {bot_name} and {max_tokens} placeholders
        """

        # Apply butter to max tokens to avoid hitting the limit
        max_tokens = max_tokens_for_prompt(bot.model_max_tokens)

        suffix = replace_variables(
            suffix_template,
            {
                BOT_NAME_VARIABLE_PLACEHOLDER: bot.name,
                MAX_TOKENS_VARIABLE_PLACEHOLDER: str(max_tokens),
            },
        )
        self._bot_suffixes[bot] = suffix
        add_suffix(bot, suffix)

    def cleanup_round_suffix(self, bot: ChatbotBase) -> None:
        """
        Remove the stored suffix from a bot's system prompt.

        Args:
            bot: The bot to remove the suffix from
        """
        if bot in self._bot_suffixes:
            remove_suffix(bot, self._bot_suffixes[bot])
            del self._bot_suffixes[bot]
