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

# Example of another simple processor

# class DCRemover(SignalProcessor):
#     def process(self, data: np.ndarray) -> np.ndarray:
#         # Remove DC offset from each channel
#         return data - np.mean(data, axis=1, keepdims=True)


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

   


from statistics import mode

class ModelProcessor(SignalProcessor):
    def __init__(self, model, window_size=250, overlap=0.5, sampling_rate=1000, aggregate=True):
        """
        Args:
            model: Loaded ML model for prediction
            window_size: Number of samples per window
            overlap: Overlap ratio between windows (0 to 1)
            sampling_rate: Sampling rate in Hz
        """
        self.model = model
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.sampling_rate = sampling_rate
        self.aggregate = aggregate
        
    def extract_features(self, signal):
        """
        Extract time-domain features from a signal array
        Args:
            signal (array-like): Input signal array
        Returns:
            dict: Dictionary containing computed features
        """
        # Convert to numpy array if not already
        signal = np.array(signal)
        # Root Mean Square (RMS)
        rms = np.sqrt(np.mean(signal**2))
        # Variance
        variance = np.var(signal)
        # Mean Absolute Value (MAV)
        mav = np.mean(np.abs(signal))
        # Slope Sign Change (SSC)
        diff = np.diff(signal)
        ssc = np.sum((diff[:-1] * diff[1:]) < 0)
        # Zero Crossing Rate (ZCR)
        zcr = np.sum(np.diff(np.signbit(signal).astype(int)) != 0)
        # Waveform Length (WL)
        wl = np.sum(np.abs(np.diff(signal)))
        return {
            'rms': rms,
            'variance': variance,
            'mav': mav,
            'ssc': ssc,
            'zcr': zcr,
            'wl': wl
        }
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Process EMG data with overlapping windows and return predictions"""
        n_channels, n_samples = data.shape
        predictions = []
        
        # Process overlapping windows
        for i in range(0, n_samples - self.window_size + 1, self.stride):
            window = data[:, i:i+self.window_size]
            features = []
            
            # Extract features from each channel
            for channel in window:
                features.extend(list(self.extract_features(channel).values()))
                
            # Make prediction
            pred = self.model.predict(np.array(features).reshape(1, -1))
            predictions.append(pred[0])
            print(f'Prediction of window {i}: {pred}')
        
        if self.aggregate:
            return mode(np.array(predictions))
        else:
            return np.array(predictions)

######################################################## Normalizing features  ######################################################################
class Normalize_EMG(SignalProcessor):
    def __init__(self, method='zscore', mode='window', buffer_size=None):
        """
        Initialize EMG normalization processor
        
        Args:
            method: Normalization method ('zscore' or 'maxdiv')
            mode: 'window' for window-based or 'buffer' for buffer-based normalization
            buffer_size: Number of samples to use for buffer mode (required if mode='buffer')
        """
        self.method = method.lower()
        self.mode = mode.lower()
        self.name = 'Normalize'
        
        if mode == 'buffer':
            if buffer_size is None:
                raise ValueError("buffer_size must be specified for buffer mode")
            self.buffer_size = buffer_size
            self.buffer = None
            self.buffer_position = 0
            
            # Store normalization parameters for buffer mode
            self.norm_mean = None
            self.norm_std = None
            self.norm_max = None
    
    def _init_buffer(self, n_channels):
        """Initialize buffer for storing samples"""
        self.buffer = np.zeros((n_channels, self.buffer_size))
        
    def _update_buffer(self, data: np.ndarray):
        """Update buffer with new data and recalculate normalization parameters"""
        if self.buffer is None:
            self._init_buffer(data.shape[0])
            
        # Calculate how much data we can add
        n_samples = data.shape[1]
        space_left = self.buffer_size - self.buffer_position
        samples_to_add = min(n_samples, space_left)
        
        # Add data to buffer
        self.buffer[:, self.buffer_position:self.buffer_position + samples_to_add] = \
            data[:, :samples_to_add]
        self.buffer_position += samples_to_add
        
        # If buffer is full, calculate normalization parameters
        if self.buffer_position >= self.buffer_size:
            if self.method == 'zscore':
                self.norm_mean = np.mean(self.buffer, axis=1, keepdims=True)
                self.norm_std = np.std(self.buffer, axis=1, keepdims=True)
            else:  # maxdiv
                self.norm_max = np.max(np.abs(self.buffer), axis=1, keepdims=True)
    
    def _normalize_window(self, data: np.ndarray) -> np.ndarray:
        """Normalize based on current window statistics"""
        if self.method == 'zscore':
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            return (data - mean) / (std + 1e-8)
        else:  # maxdiv
            max_vals = np.max(np.abs(data), axis=1, keepdims=True)
            return data / (max_vals + 1e-8)
    
    def _normalize_buffer(self, data: np.ndarray) -> np.ndarray:
        """Normalize using buffer statistics"""
        if not self.buffer_position >= self.buffer_size:
            # If buffer isn't full yet, normalize based on window
            return self._normalize_window(data)
            
        if self.method == 'zscore':
            return (data - self.norm_mean) / (self.norm_std + 1e-8)
        else:  # maxdiv
            return data / (self.norm_max + 1e-8)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize EMG data based on selected method and mode
        
        Args:
            data: Input EMG data of shape (channels, samples)
            
        Returns:
            Normalized EMG data of same shape
        """
        if self.mode == 'buffer':
            self._update_buffer(data)
            return self._normalize_buffer(data)
        else:  # window mode
            return self._normalize_window(data)
    

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
