import pytest
from src.conversation.manager import ConversationManager

def test_conversation_manager_initialization():
    manager = ConversationManager.from_config('tests/fixtures/test_config.json')
    assert manager is not None
    assert len(manager.bots) > 0

def test_run_round():
    manager = ConversationManager.from_config('tests/fixtures/test_config.json')
    response = manager.run_round()
    assert response is not None
