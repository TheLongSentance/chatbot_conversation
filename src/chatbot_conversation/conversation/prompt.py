"""
This module provides functionality for constructing and adjusting system prompts 
for Chatbot instances.
"""

from typing import Final

from chatbot_conversation.conversation.loader import ChatbotConfigData
from chatbot_conversation.models.base import DEFAULT_MAX_TOKENS

BOT_NAME_VARIABLE_PLACEHOLDER: Final[str] = "bot_name"
MAX_TOKENS_VARIABLE_PLACEHOLDER: Final[str] = "max_tokens"


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

def construct_system_prompt(core_prompt: str, bot_config: ChatbotConfigData) -> str:
    """
    Construct the system prompt for a bot based on the shared prefix and bot configuration.

    Args:
        core_prompt (str): The shared prefix for the system prompt.
        bot_config (ChatbotConfigData): The configuration for the bot.

    Returns:
        str: The constructed system prompt.

    Example:
        >>> bot_config = ChatbotConfigData(bot_name="Bot1", bot_prompt="You are Bot1.")
        >>> construct_system_prompt("Core prompt: ", bot_config)
        'Core prompt: You are Bot1.'
    """
    if bot_config.bot_params_opt.max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    else:
        max_tokens = bot_config.bot_params_opt.max_tokens

    bot_system_prompt = core_prompt + bot_config.bot_prompt
    bot_system_prompt = replace_variables(
        bot_system_prompt,
        {
            BOT_NAME_VARIABLE_PLACEHOLDER: bot_config.bot_name,
            MAX_TOKENS_VARIABLE_PLACEHOLDER: str(max_tokens),
        },
    )

    return bot_system_prompt
