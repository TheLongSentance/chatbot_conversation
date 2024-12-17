from typing import List, TypedDict
import json

class BotConfig(TypedDict):
    bot_name: str
    bot_type: str
    bot_model_version: str
    bot_specific_system_prompt: str

class ConversationConfig(TypedDict):
    conversation_seed: str
    rounds: int
    shared_system_prompt_prefix: str
    bots: List[BotConfig]

class ConfigurationLoader:
    @staticmethod
    def load_config(config_path: str) -> ConversationConfig:
        """Load conversation configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            ConversationConfig: Loaded configuration
        """
        with open(config_path, 'r') as f:
            return json.load(f)
