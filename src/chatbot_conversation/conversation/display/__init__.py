"""Display package initialization."""

from chatbot_conversation.conversation.display.abstract_display import DisplayInterface
from chatbot_conversation.conversation.display.rich_display import RichDisplay

def create_display() -> DisplayInterface:
    """Create default display implementation.

    Returns:
        Display interface implementation
    """
    return RichDisplay()

__all__ = ["DisplayInterface", "RichDisplay", "create_display"]
