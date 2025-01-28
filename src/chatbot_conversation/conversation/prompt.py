"""
This module provides functionality for constructing and adjusting system prompts 
for Chatbot instances.
"""

from chatbot_conversation.conversation.loader import ChatbotConfigData
from chatbot_conversation.models.base import ChatbotBase, DEFAULT_MAX_TOKENS


class PromptManager:
    """Handles system prompt adjustments for bots."""

    @staticmethod
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
        """
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            text = text.replace(placeholder, str(value))
        return text

    @staticmethod
    def add_suffix(bot: ChatbotBase, additional_prompt: str) -> None:
        """
        Append additional text to the system prompt.

        Args:
            additional_prompt (str): The text to append.
        """
        if additional_prompt:
            bot.system_prompt += additional_prompt

    @staticmethod
    def remove_suffix(bot: ChatbotBase, text_to_remove: str) -> None:
        """
        Remove specific text from the end of the system prompt.

        Args:
            text_to_remove (str): The text to remove.
        """
        if text_to_remove and bot.system_prompt.endswith(text_to_remove):
            bot.system_prompt = bot.system_prompt[: -len(text_to_remove)]

    @staticmethod
    def construct_system_prompt(
        shared_prefix: str, bot_config: ChatbotConfigData
    ) -> str:
        """
        Construct the system prompt for a bot based on the shared prefix and bot configuration.

        Args:
            shared_prefix (str): The shared prefix for the system prompt.
            bot_config: The configuration for the bot.

        Returns:
            str: The constructed system prompt.
        """
        if bot_config.bot_params_opt.max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        else:
            max_tokens = bot_config.bot_params_opt.max_tokens

        bot_system_prompt = shared_prefix + bot_config.bot_prompt
        bot_system_prompt = PromptManager.replace_variables(
            bot_system_prompt,
            {"bot_name": bot_config.bot_name, "max_tokens": str(max_tokens)},
        )

        return bot_system_prompt
