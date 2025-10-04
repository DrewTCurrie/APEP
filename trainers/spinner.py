"""
trainers/spinner.py

Cute terminal utilities for training feedback:
- CutePrinter: Color codes for terminal printing.
- cute_print: Print with colors easily.
- CuteSpinner: Animated spinner for live training feedback.
"""

import itertools
import sys
import time
from threading import Thread


class CutePrinter:
    """ANSI color codes for terminal printing inspired by Tokyo Night Storm theme."""
    HEADER = '\033[95m'
    PINK = '\033[95m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def cute_print(msg: str, color: str = "CYAN") -> None:
    """
    Print a message to the terminal with the specified color.

    Args:
        msg (str): Message to print.
        color (str, optional): Color name. Defaults to "CYAN".
    """
    col = getattr(CutePrinter, color.upper(), CutePrinter.CYAN)
    print(f"{col}{msg}{CutePrinter.END}")


class CuteSpinner:
    """
    Animated terminal spinner to indicate progress during training or other long tasks.
    
    Attributes:
        message (str): Message to display alongside spinner.
        color (str): Terminal color for the spinner message.
        emoji (str): Leading emoji for the spinner.
    """

    def __init__(self, message: str = "Training", color: str = "CYAN", emoji: str = "✨"):
        self.message = message
        self.color = getattr(CutePrinter, color.upper(), CutePrinter.CYAN)
        self.emoji = emoji
        self.spinner = itertools.cycle(["🌸", "💫", "🌙", "🐾", "💖"])
        self.running = False
        self.thread: Thread | None = None

    def start(self) -> None:
        """Start the spinner in a separate thread."""
        self.running = True
        self.thread = Thread(target=self._spin)
        self.thread.start()

    def _spin(self) -> None:
        """Internal loop for spinner animation."""
        while self.running:
            sym = next(self.spinner)
            sys.stdout.write(f"\r{self.color}{self.emoji} {self.message}... {sym}{CutePrinter.END}")
            sys.stdout.flush()
            time.sleep(0.2)

    def stop(self) -> None:
        """Stop the spinner and join the thread."""
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write(f"\r{self.color}{self.emoji} {self.message}... Done!{CutePrinter.END}\n")
        sys.stdout.flush()
