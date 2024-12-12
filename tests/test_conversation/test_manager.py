import pytest
import os
from src.conversation.manager import ConversationManager
from src.conversation.loader import ConfigurationLoader

def test_conversation_manager_initialization(test_config_path: str, mock_env: dict[str, str]):
    manager = ConversationManager.from_config(test_config_path)
    assert manager is not None
    assert len(manager.bots) == 2  # Based on test_config.json

def test_run_round(test_config_path: str, mock_env: dict[str, str]):
    manager = ConversationManager.from_config(test_config_path)
    manager.run_round()
    assert len(manager.conversation) > 1  # Initial message + at least one response

def test_invalid_config_path():
    with pytest.raises(FileNotFoundError):
        ConversationManager.from_config('nonexistent.json')

def test_conversation_history(test_config_path: str, mock_env: dict[str, str]):
    
    # Use actual environment variables if available
    openai_api_key = os.getenv('OPENAI_API_KEY', 'test_key')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', 'test_key')
    google_api_key = os.getenv('GOOGLE_API_KEY', 'test_key')

    # Set the environment variables for the test
    mock_env['OPENAI_API_KEY'] = openai_api_key
    mock_env['ANTHROPIC_API_KEY'] = anthropic_api_key
    mock_env['GOOGLE_API_KEY'] = google_api_key
    
    manager = ConversationManager.from_config(test_config_path)
    initial_length = len(manager.conversation)
    manager.run_round()
    assert len(manager.conversation) > initial_length
    assert all(isinstance(msg, str) for msg in manager.conversation)

def test_bot_validation(test_config_path: str, mock_env: dict[str, str]):
    config = ConfigurationLoader.load_config(test_config_path)
    config['bots'][0]['bot_type'] = 'INVALID'
    with pytest.raises(ValueError, match='Unsupported bot type'):
        ConversationManager(config)

def test_missing_env_variables(test_config_path: str):
    with pytest.raises(KeyError):
        ConversationManager.from_config(test_config_path)

def test_multiple_rounds(test_config_path: str, mock_env: dict[str, str]):
    manager = ConversationManager.from_config(test_config_path)
    initial_length = len(manager.conversation)
    num_rounds = 3
    for _ in range(num_rounds):
        manager.run_round()
    assert len(manager.conversation) >= initial_length + num_rounds * 2  # At least 2 messages per round

def test_bot_order(test_config_path: str, mock_env: dict[str, str]):
    manager = ConversationManager.from_config(test_config_path)
    manager.run_round()
    messages = manager.conversation[1:]  # Skip seed message
    # Verify alternating bot messages
    for i in range(0, len(messages), 2):
        assert messages[i].startswith("TestBot1:")
        if i + 1 < len(messages):
            assert messages[i + 1].startswith("TestBot2:")

def test_empty_conversation_seed(test_config_path: str, mock_env: dict[str, str]):
    config = ConfigurationLoader.load_config(test_config_path)
    config['conversation_seed'] = ""
    with pytest.raises(ValueError, match='Conversation seed cannot be empty'):
        ConversationManager(config)
