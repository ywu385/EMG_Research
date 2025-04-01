import numpy as np
import time
from typing import Generator, List, Tuple
from scipy import signal
import os
import csv
import time
from statistics import mode
from abc import ABC, abstractmethod

# Abstract base class for processors
class SignalProcessor(ABC):
    def initialize(self, data: np.ndarray) -> None:
        """Optional initialization step. Default does nothing."""
        pass

    @abstractmethod 
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Required processing implementation.
        Args:
            data: Numpy array of shape (channels, samples)
        Returns:
            Processed data
        """
        pass

class FiveChannels(SignalProcessor):
    def __init__(self, indices = [4,5,6,7,8]):
        self.channels = indices

    def process(self, data:np.ndarray):
        return data[self.channels]

class ZeroChannelRemover(SignalProcessor):
    def __init__(self, threshold=1):  # Add threshold parameter
        self.active_channels = None
        self.threshold = threshold
        self.name = 'Zero Channel'
        
    def initialize(self, data: np.ndarray):
        # Calculate the mean absolute value for each channel
        channel_activity = np.mean(np.abs(data), axis=1)
        # Mark channels as active only if they have significant activity
        self.active_channels = channel_activity > self.threshold
        print(f"Channel activity levels: {channel_activity}")
        print(f"Active channels: {np.where(self.active_channels)[0]}")
        
    def process(self, data: np.ndarray) -> np.ndarray:
        if self.active_channels is None:
            self.initialize(data)
        return data[self.active_channels]


class DCRemover(SignalProcessor):
    def __init__(self):
        self.name = 'DCRemover'
    def process(self, data: np.ndarray) -> np.ndarray:
        # Remove DC offset from each channel
        return data - np.mean(data, axis=1, keepdims=True)

class NotchFilter(SignalProcessor):
    def __init__(self, notch_freqs: List[float], sampling_rate: int, quality_factor: float = 30.0):
        """
        Create notch filters for removing specific frequencies
        Args:
            notch_freqs: List of frequencies to remove (e.g., [60, 120, 180])
            sampling_rate: Signal sampling frequency in Hz
            quality_factor: Quality factor for notch filter (higher = narrower notch)
        """
        self.notch_freqs = notch_freqs
        self.sampling_rate = sampling_rate
        self.quality_factor = quality_factor
        self.b_filters = []
        self.a_filters = []
        self.name = 'Notch Filter'
        
        # Create filter coefficients for each frequency
        for freq in notch_freqs:
            b, a = signal.iirnotch(freq, quality_factor, sampling_rate)
            self.b_filters.append(b)
            self.a_filters.append(a)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply all notch filters in sequence"""
        filtered = data.copy()
        for b, a in zip(self.b_filters, self.a_filters):
            filtered = signal.filtfilt(b, a, filtered, axis=1)
        return filtered


class ButterFilter(SignalProcessor):
    def __init__(self, cutoff, sampling_rate, filter_type='bandpass', order=4):
        """
        Create a Butterworth filter for EMG signal processing
        
        Args:
            cutoff: Cutoff frequency or frequencies (Hz)
                - For lowpass/highpass: single value, e.g., 200
                - For bandpass/bandstop: list/tuple of [low, high], e.g., [20, 200]
            sampling_rate: Signal sampling frequency in Hz
            filter_type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
            order: Filter order (higher = steeper roll-off, but more ripple)
        """
        self.cutoff = cutoff
        self.sampling_rate = sampling_rate
        self.filter_type = filter_type
        self.order = order
        self.name = 'Butterworth Filter'
        
        # Normalize the cutoff frequency
        nyquist = 0.5 * sampling_rate
        if isinstance(cutoff, (list, tuple)):
            self.normalized_cutoff = [cf / nyquist for cf in cutoff]
        else:
            self.normalized_cutoff = cutoff / nyquist
            
        # Get filter coefficients
        self.b, self.a = signal.butter(
            order, 
            self.normalized_cutoff, 
            btype=filter_type, 
            analog=False
        )
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter to the signal"""
        filtered = signal.filtfilt(self.b, self.a, data, axis=1)
        return filtered
   


######################################################## Normalizing features  ######################################################################

class MaxNormalizer(SignalProcessor):
    """
    SignalProcessor that normalizes EMG data by dividing by the maximum value
    for each channel, continuously updating the maximum if larger values are encountered.
    """
    def __init__(self, initial_max_values=None, epsilon=1e-8):
        """
        Initialize the max normalizer
        
        Args:
            initial_max_values: Optional array of initial maximum values for each channel
            epsilon: Small value to avoid division by zero
        """
        self.max_values = initial_max_values  # Will be initialized if None
        self.epsilon = epsilon
        self.name = 'MaxNormalizer'
        self.initialized = False
    
    def initialize(self, data: np.ndarray) -> None:
        """Initialize max values from initial data"""
        n_channels = data.shape[0]
        
        # If max values weren't provided, initialize from data
        if self.max_values is None:
            self.max_values = np.max(np.abs(data), axis=1)
        # If max values were provided but not the right shape, reshape
        elif isinstance(self.max_values, (int, float)):
            self.max_values = np.ones(n_channels) * self.max_values
        # Ensure max_values is the right shape
        elif len(self.max_values) != n_channels:
            raise ValueError(f"Expected {n_channels} max values, got {len(self.max_values)}")
        
        self.initialized = True
    
    def set_max_values(self, max_values):
        """Manually set the maximum values for each channel"""
        self.max_values = np.array(max_values)
        self.initialized = True
    
    def update_max_values(self, data: np.ndarray) -> None:
        """Update max values if larger values are found"""
        current_max = np.max(np.abs(data), axis=1)
        self.max_values = np.maximum(self.max_values, current_max)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data by dividing by the maximum value for each channel
        
        Args:
            data: EMG data of shape (channels, samples)
            
        Returns:
            Normalized data with values between -1 and 1
        """
        # Initialize if not already
        if not self.initialized:
            self.initialize(data)
        
        # Update max values if larger values are found
        self.update_max_values(data)
        
        # Normalize by dividing by max values (with epsilon to avoid division by zero)
        normalized_data = data / (self.max_values[:, np.newaxis] + self.epsilon)
        
        return normalized_data
    

######################################################## OLD IMPLEMENTATIONS ######################################################################

######################################################## OLD IMPLEMENTATION ######################################################################
# Specific processor for removing zero channels
# class ZeroChannelRemover(SignalProcessor):
#     def __init__(self):
#         self.active_channels = None

#     def initialize(self, data: np.ndarray): # initializes channels that are non-zero.  This allows for less flucation when streaming
#         self.active_channels = np.any(data != 0, axis=1)
        
#     def process(self, data: np.ndarray) -> np.ndarray:
#         if self.active_channels is None:
#             self.initialize(data)
#         return data[self.active_channels]
######################################################## OLD IMPLEMENTATION ######################################################################