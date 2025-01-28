"""Abstract base class for display implementations."""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class DisplayInterface(ABC):
    """Abstract base class defining display interface."""

    @abstractmethod
    def clear(self) -> None:
        """Clear the display."""
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def show_text(self, text: str) -> None:
        """Display text.

        Args:
            text: Text to display
        """
        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def show_streaming_text(self, text_generator: Iterator[Any]) -> str:
        """Display streaming text with updates.

        Args:
            text_generator: Iterator yielding text chunks

        Returns:
            Complete text after all chunks processed
        """
        pass  # pylint: disable=unnecessary-pass
