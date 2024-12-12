from src.conversation.loader import ConfigurationLoader
from src.conversation.manager import ConversationManager
from src.utils.env import APIConfig

def main():
    # Set up environment variables for API access
    APIConfig.setup_env()
    
    # Load configuration and initialize conversation manager
    config = ConfigurationLoader.load_config('config.json')
    manager = ConversationManager.from_config('config.json')
    
    # Run conversation for configured number of rounds
    print("\nStarting conversation...\n")
    for round_num in range(config['rounds']):
        print(f"\n--- Round {round_num + 1} ---")
        manager.run_round()

if __name__ == "__main__":
    main()