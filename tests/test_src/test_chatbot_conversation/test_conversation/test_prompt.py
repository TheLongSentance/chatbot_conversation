"""
Unit tests for the PromptManager class in prompt.py.
"""

from chatbot_conversation.conversation.prompt import PromptManager
from chatbot_conversation.models import ChatbotBase
from chatbot_conversation.conversation import ConversationConfig


def test_replace_variables():
    """
    Test the replace_variables method of PromptManager.
    """
    text = "Hello, {bot_name}! Your max tokens are {max_tokens}."
    variables = {"bot_name": "GPT-4", "max_tokens": "100"}
    result = PromptManager.replace_variables(text, variables)
    assert result == "Hello, GPT-4! Your max tokens are 100."


def test_system_prompt_add_suffix(mock_bot: ChatbotBase):
    """
    Test the system_prompt_add_suffix method of PromptManager.
    """
    initial_prompt = mock_bot.system_prompt
    additional_prompt = " Additional text."
    PromptManager.add_suffix(mock_bot, additional_prompt)
    assert mock_bot.system_prompt == initial_prompt + additional_prompt


def test_system_prompt_remove_suffix(mock_bot: ChatbotBase):
    """
    Test the system_prompt_remove_suffix method of PromptManager.
    """
    initial_prompt = mock_bot.system_prompt
    text_to_remove = " Initial prompt"
    PromptManager.remove_suffix(mock_bot, text_to_remove)
    assert mock_bot.system_prompt == initial_prompt.replace(text_to_remove, "")


def test_construct_system_prompt(sample_conversation_config: ConversationConfig):
    """
    Test the construct_system_prompt method of PromptManager.
    """
    shared_prefix = "Shared prefix "
    bot_config = sample_conversation_config.bots[0]
    result = PromptManager.construct_system_prompt(shared_prefix, bot_config)
    expected_prompt = "Shared prefix You are Bot1, an example bot."
    assert result == expected_prompt
