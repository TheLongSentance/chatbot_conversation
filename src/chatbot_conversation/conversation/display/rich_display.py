"""Rich-based implementation of display interface."""

import os
from typing import Iterator, Any
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from .abstract_display import DisplayInterface


class RichDisplay(DisplayInterface):
    """Rich library implementation of display interface."""

    def __init__(self) -> None:
        """Initialize Rich console."""
        self.console = Console()

    def clear(self) -> None:
        """Clear the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def show_text(self, text: str) -> None:
        """Display markdown formatted text.

        Args:
            text: Text to display as markdown
        """
        self.console.print(Markdown(f"{text}"))

    def show_streaming_text(self, text_generator: Iterator[Any]) -> str:
        """Display streaming text with live updates.

        Args:
            text_generator: Iterator yielding text chunks

        Returns:
            Complete text after all chunks processed
        """
        current_text = ""
        with Live(Markdown(current_text), refresh_per_second=4) as live:
            for chunk in text_generator:
                current_text += chunk
                live.update(Markdown(current_text))
        return current_text
