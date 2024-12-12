from src.conversation.manager import ConversationManager

def test_conversation_manager_initialization(test_config_path: str, mock_env: dict[str, str]):
    manager = ConversationManager.from_config(test_config_path)
    assert manager is not None
    assert len(manager.bots) == 2  # Based on test_config.json

def test_run_round(test_config_path: str, mock_env: dict[str, str]):
    manager = ConversationManager.from_config(test_config_path)
    manager.run_round()
    assert len(manager.conversation) > 1  # Initial message + at least one response
