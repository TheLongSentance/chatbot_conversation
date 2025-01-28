"""
This module provides display functionality for chatbot outputs, using the Rich library.
"""
from abc import ABC, abstractmethod
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
import os
from typing import Iterator, Any


class DisplayInterface(ABC):
    """Abstract base class defining display interface"""

    @abstractmethod
    def clear(self) -> None:
        """Clear the display"""
        pass

    @abstractmethod
    def show_text(self, text: str) -> None:
        """Display text"""
        pass

    @abstractmethod
    def show_streaming_text(self, text_generator: Iterator[Any]) -> str:
        """Display streaming text with updates"""
        pass


class RichDisplayManager(DisplayInterface):
    """Rich-based implementation of display manager"""

    def __init__(self):
        self.console = Console()

    def clear(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")

    def show_text(self, text: str) -> None:
        self.console.print(Markdown(f"{text}"))

    def show_streaming_text(self, text_generator: Iterator[Any]) -> str:
        current_text = ""
        with Live(Markdown(current_text), refresh_per_second=4) as live:
            for chunk in text_generator:
                current_text += chunk
                live.update(Markdown(current_text))
        return current_text
