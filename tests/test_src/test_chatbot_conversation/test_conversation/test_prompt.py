"""
Unit tests for the prompt management functionality in prompt.py.
"""

from chatbot_conversation.conversation import ConversationConfig
from chatbot_conversation.conversation.prompt import (
    BOT_NAME_VARIABLE_PLACEHOLDER,
    MAX_TOKENS_VARIABLE_PLACEHOLDER,
    construct_system_prompt,
    replace_variables,
)


def test_replace_variables() -> None:
    """
    Test the replace_variables utility function.

    This test ensures that the placeholders in the text are correctly replaced
    with the provided variable values.
    """
    text = f"Hello, {{{BOT_NAME_VARIABLE_PLACEHOLDER}}}! Your max tokens are {{{MAX_TOKENS_VARIABLE_PLACEHOLDER}}}."
    variables = {
        BOT_NAME_VARIABLE_PLACEHOLDER: "GPT-4",
        MAX_TOKENS_VARIABLE_PLACEHOLDER: "100",
    }
    result = replace_variables(text, variables)
    assert result == "Hello, GPT-4! Your max tokens are 100."


def test_construct_system_prompt(
    sample_conversation_config: ConversationConfig,
) -> None:
    """
    Test the construct_system_prompt utility function.

    This test ensures that the system prompt is correctly constructed based on
    the shared prefix and bot configuration.
    """
    shared_prefix = "Shared prefix: "
    bot_config = sample_conversation_config.bots[0]
    result = construct_system_prompt(shared_prefix, bot_config)
    expected_prompt = "Shared prefix: You are Bot1, an example bot."
    assert result == expected_prompt
