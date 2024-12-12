from src.conversation.manager import ConversationManager
from src.utils.env import APIConfig

def main():
    # Set up environment variables for API access
    APIConfig.setup_env()
    
    # Load configuration and initialize conversation manager
    manager = ConversationManager.from_config('config.json')
    
    # Run conversation for configured number of rounds
    manager.run_conversation()

if __name__ == "__main__":
    main()