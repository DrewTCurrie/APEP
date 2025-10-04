"""
trainers/__init__.py

This module exposes the trainer-related utilities and callbacks for easier imports.

Modules included:
- visualizer.py: LiveTrainingVisualizer for real-time training plots.
- callbacks.py: Trainer callbacks like ValidationPreviewCallback.
- spinner.py: CuteSpinner and CutePrinter for terminal progress display.
"""

from .visualizer import LiveTrainingVisualizer
from .callbacks import ValidationPreviewCallback
from .spinner import CuteSpinner, cute_print
