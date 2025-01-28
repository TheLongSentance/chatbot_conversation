import pytest
from chatbot_conversation.conversation.display.rich_display import RichDisplay

@pytest.fixture
def display() -> RichDisplay:
    """Fixture for RichDisplay instance."""
    return RichDisplay()
