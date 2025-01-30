"""
Unit tests for the prompt management functionality in prompt.py.
"""

from chatbot_conversation.conversation import ConversationConfig
from chatbot_conversation.conversation.prompt import (
    SuffixManager,
    add_suffix,
    BOT_NAME_VARIABLE_PLACEHOLDER,
    construct_system_prompt,
    max_tokens_for_prompt,
    MAX_TOKENS_VARIABLE_PLACEHOLDER,
    remove_suffix,
    replace_variables,
)
from chatbot_conversation.models import ChatbotBase


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


def test_system_prompt_add_suffix(mock_bot: ChatbotBase) -> None:
    """
    Test the system_prompt_add_suffix utility function.

    This test verifies that additional text is correctly appended to the
    system prompt of the bot.
    """
    initial_prompt = mock_bot.system_prompt
    additional_prompt = " Additional text."
    add_suffix(mock_bot, additional_prompt)
    assert mock_bot.system_prompt == initial_prompt + additional_prompt


def test_system_prompt_remove_suffix(mock_bot: ChatbotBase) -> None:
    """
    Test the system_prompt_remove_suffix utility function.

    This test checks that specific text is correctly removed from the end of
    the system prompt of the bot.
    """
    initial_prompt = mock_bot.system_prompt
    text_to_remove = " Initial prompt"
    remove_suffix(mock_bot, text_to_remove)
    assert mock_bot.system_prompt == initial_prompt.replace(text_to_remove, "")


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


def test_suffix_manager_setup_round_suffix(mock_bot: ChatbotBase) -> None:
    """
    Test the setup_round_suffix method of SuffixManager.

    This test ensures that the suffix is correctly calculated and appended to the bot's system prompt.
    """
    suffix_manager = SuffixManager()
    suffix_template = (
        f" Suffix for {{{BOT_NAME_VARIABLE_PLACEHOLDER}}} "
        f"with max tokens {{{MAX_TOKENS_VARIABLE_PLACEHOLDER}}}."
    )
    suffix_manager.setup_round_suffix(mock_bot, suffix_template)
    expected_suffix = f" Suffix for TestBot with max tokens {max_tokens_for_prompt(100)}."
    assert mock_bot.system_prompt.endswith(expected_suffix)


def test_suffix_manager_cleanup_round_suffix(mock_bot: ChatbotBase) -> None:
    """
    Test the cleanup_round_suffix method of SuffixManager.

    This test ensures that the suffix is correctly removed from the bot's system prompt.
    """
    suffix_manager = SuffixManager()
    suffix_template = (
        f" Suffix for {{{BOT_NAME_VARIABLE_PLACEHOLDER}}} "
        f"with max tokens {{{MAX_TOKENS_VARIABLE_PLACEHOLDER}}}."
    )
    suffix_manager.setup_round_suffix(mock_bot, suffix_template)
    expected_suffix = (
        f" Suffix for TestBot with max tokens {max_tokens_for_prompt(100)}."
    )
    assert mock_bot.system_prompt.endswith(expected_suffix)
    suffix_manager.cleanup_round_suffix(mock_bot)
    assert not mock_bot.system_prompt.endswith(expected_suffix)
