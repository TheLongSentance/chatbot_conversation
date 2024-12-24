"""
This module contains the BotType enumeration for different chatbot types.
"""

from enum import Enum, auto


class BotType(Enum):
    """Enumeration of different bot types."""

    GPT = auto()
    CLAUDE = auto()
    GEMINI = auto()
    OLLAMA = auto()
