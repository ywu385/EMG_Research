"""
EMG components package for EMG Navigation Game.
Contains the EMG processor and related processing modules.
"""

# This allows importing from the package
from emg_comp.emg_processor import initialize_emg_processing, shutdown_emg_processing, update_emg_state

# Define what should be imported when using "from emg_comp import *"
__all__ = ['initialize_emg_processing', 'shutdown_emg_processing', 'update_emg_state']