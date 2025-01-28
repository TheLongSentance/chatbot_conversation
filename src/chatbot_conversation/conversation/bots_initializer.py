"""
This module provides initialization logic for chatbot bots by configuring
and creating Chatbot instances using the provided conversation configuration.
"""

from typing import List

from chatbot_conversation.conversation.loader import ConversationConfig
from chatbot_conversation.conversation.prompt import PromptManager
from chatbot_conversation.models import (
    BotRegistry,
    ChatbotBase,
    ChatbotConfig,
    ChatbotFactory,
    ChatbotModel,
    ChatbotParamsOpt,
)


class BotsInitializer:
    """
    Initializes and manages the creation of Chatbot instances.
    """

    def __init__(self) -> None:

        self.bot_registry = BotRegistry()  # get the singleton instance
        self.factory = ChatbotFactory(self.bot_registry)

    def initialize_bots(self, config: ConversationConfig) -> List[ChatbotBase]:
        """
        Create and return a list of ChatbotBase objects based on the conversation config.
        """
        bots: List[ChatbotBase] = []

        for bot_config in config.bots:
            # Construct the system prompt for the bot
            bot_system_prompt = PromptManager.construct_system_prompt(
                config.shared_prefix, bot_config
            )

            # Create ChatbotConfig object
            chatbot_config = ChatbotConfig(
                name=bot_config.bot_name,
                system_prompt=bot_system_prompt,
                model=ChatbotModel(
                    type=bot_config.bot_type,
                    version=bot_config.bot_version,
                    params_opt=ChatbotParamsOpt(
                        temperature=bot_config.bot_params_opt.temperature,
                        max_tokens=bot_config.bot_params_opt.max_tokens,
                    ),
                ),
            )

            bot = self.factory.create_bot(chatbot_config)
            bots.append(bot)
        return bots

    def get_bot_registry(self) -> BotRegistry:
        """
        Returns the bot registry instance.
        """
        return self.bot_registry
