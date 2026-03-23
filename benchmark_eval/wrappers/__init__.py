"""
Wrappers for different EEG-to-Text models to adapt to unified benchmark interface.
"""

from .cet_mae_wrapper import CETMAEWrapper
from .eeg_to_text_wrapper import EEGToTextWrapper
from .eeg2text_wrapper import EEG2TextWrapper
from .glim_wrapper import GLIMWrapper

__all__ = [
    "CETMAEWrapper",
    "EEGToTextWrapper",
    "EEG2TextWrapper",
    "GLIMWrapper",
]
