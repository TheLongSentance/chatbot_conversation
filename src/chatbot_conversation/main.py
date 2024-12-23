"""
This module initializes and runs the chatbot conversation.
"""

import os

from chatbot_conversation.conversation.manager import ConversationManager
from chatbot_conversation.utils.env import APIConfig


def main() -> None:
    """
    Main function to set up environment variables, load configuration,
    initialize the conversation manager, and run the conversation.
    """
    # Set up environment variables for API access
    APIConfig.setup_env()

    # Load configuration and initialize conversation manager
    manager = ConversationManager.from_config(os.path.join("config", "config.json"))

    # Run conversation for configured number of rounds
    manager.run_conversation()


if __name__ == "__main__":
    main()
