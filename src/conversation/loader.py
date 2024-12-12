from typing import List, TypedDict
import json

class BotConfig(TypedDict):
    name: str
    bot_type: str
    model_version: str
    system_prompt: str

class ConversationConfig(TypedDict):
    conversation_seed: str
    rounds: int
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
