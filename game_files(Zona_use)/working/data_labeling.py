import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

class ImprovedActivityLabeler:
    """
    An improved EMG activity labeler that uses both RMS thresholds and
    signal dynamics to better capture gesture transitions.
    """
    def __init__(self, 
                 rest_threshold_percentile=25,  # Use lower percentile to catch more active windows
                 derivative_threshold_factor=0.5,
                 minimum_active_duration=3,     # Minimum consecutive active windows
                 window_size=250,
                 history_size=5):              # Number of previous windows to consider
        """
        Initialize the activity labeler.
        
        Args:
            rest_threshold_percentile: Percentile for setting rest threshold (0-100)
            derivative_threshold_factor: Factor for derivative-based activity detection
            minimum_active_duration: Minimum consecutive windows to maintain active state
            window_size: Size of EMG windows
            history_size: Number of previous windows to consider for trend analysis
        """
        self.channel_rms = None
        self.rest_threshold = None
        self.derivative_threshold = None
        self.primary_channel = None
        self.rest_threshold_percentile = rest_threshold_percentile
        self.derivative_threshold_factor = derivative_threshold_factor
        self.minimum_active_duration = minimum_active_duration
        self.window_size = window_size
        self.history_size = history_size
        self.is_initialized = False
        
        # State variables
        self.rms_history = []
        self.derivative_history = []
        self.consecutive_active = 0
        self.consecutive_rest = 0
        
    def initialize_from_data(self, data: np.ndarray):
        """
        Initialize thresholds from complete dataset.
        
        Args:
            data: EMG data of shape (n_channels, n_samples)
        """
        n_channels, n_samples = data.shape
        
        # Calculate RMS for each channel over entire dataset
        self.channel_rms = np.sqrt(np.mean(np.square(data), axis=1))
        
        # Identify most active channel(s)
        self.primary_channel = np.argmax(self.channel_rms)
        
        # Calculate RMS for each window in the dataset using primary channel
        window_size = self.window_size
        stride = window_size // 2  # 50% overlap
        rms_values = []
        derivative_values = []
        prev_rms = None
        
        for i in range(0, n_samples - window_size + 1, stride):
            # Calculate RMS for this window
            window_rms = np.sqrt(np.mean(
                np.square(data[self.primary_channel, i:i+window_size])
            ))
            rms_values.append(window_rms)
            
            # Calculate derivative (change in RMS)
            if prev_rms is not None:
                derivative_values.append(abs(window_rms - prev_rms))
            prev_rms = window_rms
        
        # Set thresholds
        self.rest_threshold = np.percentile(rms_values, self.rest_threshold_percentile)
        self.derivative_threshold = np.percentile(derivative_values, 75) * self.derivative_threshold_factor
        
        print(f"Initialized improved activity labeler:")
        print(f"- Primary channel: {self.primary_channel}")
        print(f"- Rest threshold: {self.rest_threshold:.6f} (percentile: {self.rest_threshold_percentile}%)")
        print(f"- Derivative threshold: {self.derivative_threshold:.6f}")
        
        self.is_initialized = True
        self.rms_history = []
        self.derivative_history = []
        
    def label_window(self, window: np.ndarray) -> str:
        """
        Label a single window as 'active' or 'rest' based on both RMS and signal dynamics.
        
        Args:
            window: EMG window data of shape (n_channels, n_samples)
            
        Returns:
            'active' or 'rest'
        """
        if not self.is_initialized:
            raise ValueError("Activity labeler not initialized. Call initialize_from_data first.")
        
        # Calculate RMS for this window
        window_rms = np.sqrt(np.mean(np.square(window[self.primary_channel])))
        
        # Update history
        self.rms_history.append(window_rms)
        if len(self.rms_history) > self.history_size:
            self.rms_history.pop(0)
        
        # Calculate derivative if we have history
        current_derivative = 0
        if len(self.rms_history) >= 2:
            current_derivative = abs(self.rms_history[-1] - self.rms_history[-2])
            self.derivative_history.append(current_derivative)
            if len(self.derivative_history) > self.history_size:
                self.derivative_history.pop(0)
        
        # Define conditions for activity detection
        is_above_threshold = window_rms > self.rest_threshold
        is_changing = (len(self.derivative_history) > 0 and
                       max(self.derivative_history) > self.derivative_threshold)
        
        # Handle state transitions with hysteresis
        is_active = False
        
        if is_above_threshold or is_changing:
            self.consecutive_active += 1
            self.consecutive_rest = 0
            is_active = True
        else:
            self.consecutive_rest += 1
            # Only transition to rest after multiple consecutive rest windows
            if self.consecutive_rest >= 2:
                self.consecutive_active = 0
                is_active = False
            else:
                # Stay active if we were recently active
                is_active = self.consecutive_active > 0
        
        return 'active' if is_active else 'rest'
    
    def reset_state(self):
        """Reset the internal state (call when processing a new file)"""
        self.rms_history = []
        self.derivative_history = []
        self.consecutive_active = 0
        self.consecutive_rest = 0
    
    def label_windows(self, windows: List[np.ndarray]) -> List[str]:
        """
        Label multiple windows.
        
        Args:
            windows: List of EMG windows
            
        Returns:
            List of labels ('active' or 'rest')
        """
        # Reset state before processing a new sequence
        self.reset_state()
        return [self.label_window(window) for window in windows]
    
    def visualize_thresholds(self, data: np.ndarray, n_windows: int = 100):
        """
        Visualize the activity detection on a portion of the data.
        
        Args:
            data: EMG data of shape (n_channels, n_samples)
            n_windows: Number of windows to visualize
        """
        if not self.is_initialized:
            raise ValueError("Activity labeler not initialized. Call initialize_from_data first.")
        
        # Create windows from the data
        n_channels, n_samples = data.shape
        window_size = self.window_size
        stride = window_size // 2
        
        # Reset state
        self.reset_state()
        
        windows = []
        for i in range(0, min(n_samples - window_size, n_windows * stride), stride):
            windows.append(data[:, i:i+window_size])
        
        # Label windows
        labels = self.label_windows(windows[:n_windows])
        
        # Calculate RMS for visualization
        rms_values = [np.sqrt(np.mean(np.square(window[self.primary_channel]))) 
                      for window in windows[:n_windows]]
        
        # Create derivative values with zero-padding for first element
        derivative_values = [0] + [abs(rms_values[i] - rms_values[i-1]) 
                                  for i in range(1, len(rms_values))]
        
        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Raw signal
        time = np.arange(n_windows * stride + window_size) / 1000  # Assuming 1000 Hz
        signal_to_plot = data[self.primary_channel, :n_windows * stride + window_size]
        ax1.plot(time, signal_to_plot)
        ax1.set_title(f'Raw EMG Signal (Channel {self.primary_channel})')
        ax1.set_ylabel('Amplitude')
        
        # Plot 2: RMS values and threshold
        window_times = np.arange(window_size/2, window_size/2 + n_windows * stride, stride) / 1000
        ax2.plot(window_times, rms_values, label='Window RMS')
        ax2.axhline(y=self.rest_threshold, color='r', linestyle='--', 
                    label=f'Rest Threshold ({self.rest_threshold_percentile}%)')
        
        # Color the background based on labels
        for i, label in enumerate(labels):
            start = window_times[i] - window_size/2000  # Adjust for window size
            end = start + window_size/1000
            if label == 'active':
                ax2.axvspan(start, end, alpha=0.2, color='green')
            else:
                ax2.axvspan(start, end, alpha=0.1, color='red')
        
        ax2.set_title('Window RMS and Activity Detection')
        ax2.set_ylabel('RMS')
        ax2.legend()
        
        # Plot 3: Derivative values
        ax3.plot(window_times, derivative_values, label='RMS Change')
        ax3.axhline(y=self.derivative_threshold, color='r', linestyle='--', 
                    label=f'Derivative Threshold')
        ax3.set_title('RMS Change Between Windows')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Absolute Change')
        ax3.legend()
        
        plt.tight_layout()
        return fig
    
    def get_primary_channel(self):
        """Get the primary (most active) channel index"""
        if not self.is_initialized:
            raise ValueError("Activity labeler not initialized. Call initialize_from_data first.")
        return self.primary_channel
    

class EnhancedActivityLabeler:
    """
    An enhanced EMG activity labeler that uses both RMS thresholds and
    signal dynamics to better capture gesture transitions, with improved
    rest period detection.
    """
    def __init__(self, 
                 rest_threshold_percentile=25,  # Use lower percentile to catch more active windows
                 derivative_threshold_factor=0.5,
                 minimum_active_duration=3,     # Minimum consecutive active windows
                 minimum_rest_duration=3,       # NEW: Minimum consecutive rest windows
                 rest_consistency_factor=0.7,   # NEW: How consistent the RMS should be during rest
                 window_size=250,
                 history_size=5):              # Number of previous windows to consider
        """
        Initialize the activity labeler.
        
        Args:
            rest_threshold_percentile: Percentile for setting rest threshold (0-100)
            derivative_threshold_factor: Factor for derivative-based activity detection
            minimum_active_duration: Minimum consecutive windows to maintain active state
            minimum_rest_duration: Minimum consecutive windows to maintain rest state
            rest_consistency_factor: Factor to determine if RMS is consistent enough for rest (0-1)
            window_size: Size of EMG windows
            history_size: Number of previous windows to consider for trend analysis
        """
        self.channel_rms = None
        self.rest_threshold = None
        self.derivative_threshold = None
        self.primary_channel = None
        self.rest_threshold_percentile = rest_threshold_percentile
        self.derivative_threshold_factor = derivative_threshold_factor
        self.minimum_active_duration = minimum_active_duration
        self.minimum_rest_duration = minimum_rest_duration
        self.rest_consistency_factor = rest_consistency_factor
        self.window_size = window_size
        self.history_size = history_size
        self.is_initialized = False
        
        # State variables
        self.rms_history = []
        self.derivative_history = []
        self.consecutive_active = 0
        self.consecutive_rest = 0
        self.last_state = 'rest'  # Track the previous state
        
    def initialize_from_data(self, data: np.ndarray):
        """
        Initialize thresholds from complete dataset.
        
        Args:
            data: EMG data of shape (n_channels, n_samples)
        """
        n_channels, n_samples = data.shape
        
        # Calculate RMS for each channel over entire dataset
        self.channel_rms = np.sqrt(np.mean(np.square(data), axis=1))
        
        # Identify most active channel(s)
        self.primary_channel = np.argmax(self.channel_rms)
        
        # Calculate RMS for each window in the dataset using primary channel
        window_size = self.window_size
        stride = window_size // 2  # 50% overlap
        rms_values = []
        derivative_values = []
        prev_rms = None
        
        for i in range(0, n_samples - window_size + 1, stride):
            # Calculate RMS for this window
            window_rms = np.sqrt(np.mean(
                np.square(data[self.primary_channel, i:i+window_size])
            ))
            rms_values.append(window_rms)
            
            # Calculate derivative (change in RMS)
            if prev_rms is not None:
                derivative_values.append(abs(window_rms - prev_rms))
            prev_rms = window_rms
        
        # Set thresholds
        self.rest_threshold = np.percentile(rms_values, self.rest_threshold_percentile)
        self.derivative_threshold = np.percentile(derivative_values, 75) * self.derivative_threshold_factor
        
        # NEW: Calculate lower bound for rest RMS values
        self.deep_rest_threshold = np.percentile(rms_values, 10)
        
        print(f"Initialized enhanced activity labeler:")
        print(f"- Primary channel: {self.primary_channel}")
        print(f"- Rest threshold: {self.rest_threshold:.6f} (percentile: {self.rest_threshold_percentile}%)")
        print(f"- Deep rest threshold: {self.deep_rest_threshold:.6f} (percentile: 10%)")
        print(f"- Derivative threshold: {self.derivative_threshold:.6f}")
        
        self.is_initialized = True
        self.rms_history = []
        self.derivative_history = []
        
    def label_window(self, window: np.ndarray) -> str:
        """
        Label a single window as 'active' or 'rest' based on both RMS and signal dynamics.
        
        Args:
            window: EMG window data of shape (n_channels, n_samples)
            
        Returns:
            'active' or 'rest'
        """
        if not self.is_initialized:
            raise ValueError("Activity labeler not initialized. Call initialize_from_data first.")
        
        # Calculate RMS for this window
        window_rms = np.sqrt(np.mean(np.square(window[self.primary_channel])))
        
        # Update history
        self.rms_history.append(window_rms)
        if len(self.rms_history) > self.history_size:
            self.rms_history.pop(0)
        
        # Calculate derivative if we have history
        current_derivative = 0
        if len(self.rms_history) >= 2:
            current_derivative = abs(self.rms_history[-1] - self.rms_history[-2])
            self.derivative_history.append(current_derivative)
            if len(self.derivative_history) > self.history_size:
                self.derivative_history.pop(0)
        
        # Define conditions for activity detection
        is_above_threshold = window_rms > self.rest_threshold
        is_changing = (len(self.derivative_history) > 0 and
                       max(self.derivative_history) > self.derivative_threshold)
        
        # NEW: Define conditions for rest detection
        is_below_threshold = window_rms < self.rest_threshold
        
        # NEW: Check if RMS is consistent (low variation) - a sign of genuine rest
        is_consistent_rest = False
        if len(self.rms_history) >= 3:  # Need at least 3 points to check consistency
            recent_rms = self.rms_history[-3:]
            rms_mean = sum(recent_rms) / len(recent_rms)
            # Calculate coefficient of variation (std/mean)
            if rms_mean > 0:  # Avoid division by zero
                rms_variation = np.std(recent_rms) / rms_mean
                is_consistent_rest = (rms_variation < (1.0 - self.rest_consistency_factor))
        
        # Deep rest condition - very low RMS values
        is_deep_rest = window_rms < self.deep_rest_threshold
        
        # Handle state transitions with hysteresis
        if is_above_threshold or is_changing:
            # Conditions suggest activity
            self.consecutive_active += 1
            self.consecutive_rest = 0
            
            # Only transition to active after sufficient consecutive active windows
            if self.consecutive_active >= self.minimum_active_duration or self.last_state == 'active':
                self.last_state = 'active'
                return 'active'
            else:
                # Not enough consecutive active windows yet, stay in previous state
                return self.last_state
                
        elif is_deep_rest or (is_below_threshold and is_consistent_rest):
            # Strong conditions for rest - very low RMS or below threshold and consistent
            self.consecutive_rest += 1
            self.consecutive_active = 0
            
            # Only transition to rest after sufficient consecutive rest windows
            if self.consecutive_rest >= self.minimum_rest_duration or self.last_state == 'rest':
                self.last_state = 'rest'
                return 'rest'
            else:
                # Not enough consecutive rest windows yet, stay in previous state
                return self.last_state
                
        else:
            # No strong conditions either way, maintain the previous state with some decay
            if self.last_state == 'active':
                self.consecutive_active += 1
                # Allow active state to persist a bit longer
                return 'active'
            else:
                self.consecutive_rest += 1
                return 'rest'
    
    def reset_state(self):
        """Reset the internal state (call when processing a new file)"""
        self.rms_history = []
        self.derivative_history = []
        self.consecutive_active = 0
        self.consecutive_rest = 0
        self.last_state = 'rest'
    
    def label_windows(self, windows: List[np.ndarray]) -> List[str]:
        """
        Label multiple windows.
        
        Args:
            windows: List of EMG windows
            
        Returns:
            List of labels ('active' or 'rest')
        """
        # Reset state before processing a new sequence
        self.reset_state()
        return [self.label_window(window) for window in windows]
    
    def visualize_thresholds(self, data: np.ndarray, n_windows: int = 100):
        """
        Visualize the activity detection on a portion of the data.
        
        Args:
            data: EMG data of shape (n_channels, n_samples)
            n_windows: Number of windows to visualize
        """
        if not self.is_initialized:
            raise ValueError("Activity labeler not initialized. Call initialize_from_data first.")
        
        # Create windows from the data
        n_channels, n_samples = data.shape
        window_size = self.window_size
        stride = window_size // 2
        
        # Reset state
        self.reset_state()
        
        windows = []
        for i in range(0, min(n_samples - window_size, n_windows * stride), stride):
            windows.append(data[:, i:i+window_size])
        
        # Label windows
        labels = self.label_windows(windows[:n_windows])
        
        # Calculate RMS for visualization
        rms_values = [np.sqrt(np.mean(np.square(window[self.primary_channel]))) 
                      for window in windows[:n_windows]]
        
        # Create derivative values with zero-padding for first element
        derivative_values = [0] + [abs(rms_values[i] - rms_values[i-1]) 
                                  for i in range(1, len(rms_values))]
        
        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Raw signal
        time = np.arange(n_windows * stride + window_size) / 1000  # Assuming 1000 Hz
        signal_to_plot = data[self.primary_channel, :n_windows * stride + window_size]
        ax1.plot(time, signal_to_plot)
        ax1.set_title(f'Raw EMG Signal (Channel {self.primary_channel})')
        ax1.set_ylabel('Amplitude')
        
        # Plot 2: RMS values and thresholds
        window_times = np.arange(window_size/2, window_size/2 + n_windows * stride, stride) / 1000
        ax2.plot(window_times, rms_values, label='Window RMS')
        ax2.axhline(y=self.rest_threshold, color='r', linestyle='--', 
                    label=f'Rest Threshold ({self.rest_threshold_percentile}%)')
        ax2.axhline(y=self.deep_rest_threshold, color='g', linestyle='--', 
                    label=f'Deep Rest Threshold (10%)')
        
        # Color the background based on labels
        for i, label in enumerate(labels):
            start = window_times[i] - window_size/2000  # Adjust for window size
            end = start + window_size/1000
            if label == 'active':
                ax2.axvspan(start, end, alpha=0.2, color='green')
            else:
                ax2.axvspan(start, end, alpha=0.1, color='red')
        
        ax2.set_title('Window RMS and Activity Detection')
        ax2.set_ylabel('RMS')
        ax2.legend()
        
        # Plot 3: Derivative values
        ax3.plot(window_times, derivative_values, label='RMS Change')
        ax3.axhline(y=self.derivative_threshold, color='r', linestyle='--', 
                    label=f'Derivative Threshold')
        ax3.set_title('RMS Change Between Windows')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Absolute Change')
        ax3.legend()
        
        plt.tight_layout()
        return fig
    
    def get_primary_channel(self):
        """Get the primary (most active) channel index"""
        if not self.is_initialized:
            raise ValueError("Activity labeler not initialized. Call initialize_from_data first.")
        return self.primary_channel
    

class RestActiveLabeler:
    """
    An EMG activity labeler that identifies rest periods based on local minima and flat regions,
    with everything else considered active to properly capture gesture transitions.
    """
    def __init__(self, 
                 rest_threshold_percentile=25,
                 local_minima_margin=0.1,     # How much lower a local minimum must be compared to neighbors
                 flatness_window=7,           # Number of windows to check for flatness
                 max_slope_factor=0.3,        # Maximum slope relative to mean RMS for rest
                 max_rest_rms_percentile=65,  # Upper percentile limit for rest RMS values
                 rest_expansion=2,            # Number of windows to expand around detected rest regions
                 window_size=250,
                 history_size=10):
        """
        Initialize the Rest/Active labeler.
        
        Args:
            rest_threshold_percentile: Percentile for setting rest threshold (0-100)
            local_minima_margin: How much lower a local minimum must be compared to neighbors
            flatness_window: How many windows to consider when checking for flatness
            max_slope_factor: Maximum allowed slope as a fraction of mean RMS for rest periods
            max_rest_rms_percentile: Percentile for upper limit of rest RMS values (0-100)
            rest_expansion: Number of windows to expand around detected rest regions
            window_size: Size of EMG windows
            history_size: Number of previous windows to consider
        """
        self.channel_rms = None
        self.rest_threshold = None
        self.primary_channel = None
        self.rest_threshold_percentile = rest_threshold_percentile
        self.local_minima_margin = local_minima_margin
        self.flatness_window = flatness_window
        self.max_slope_factor = max_slope_factor
        self.max_rest_rms_percentile = max_rest_rms_percentile
        self.rest_expansion = rest_expansion
        self.window_size = window_size
        self.history_size = history_size
        self.is_initialized = False
        
        # State variables
        self.rms_history = []
        self.slope_history = []
        self.last_state = 'rest'
        self.window_labels = []
        
    def initialize_from_data(self, data: np.ndarray):
        """
        Initialize thresholds from complete dataset.
        
        Args:
            data: EMG data of shape (n_channels, n_samples)
        """
        n_channels, n_samples = data.shape
        
        # Calculate RMS for each channel over entire dataset
        self.channel_rms = np.sqrt(np.mean(np.square(data), axis=1))
        
        # Identify most active channel(s)
        self.primary_channel = np.argmax(self.channel_rms)
        
        # Calculate RMS for each window in the dataset using primary channel
        window_size = self.window_size
        stride = window_size // 2  # 50% overlap
        rms_values = []
        slope_values = []
        prev_rms = None
        
        for i in range(0, n_samples - window_size + 1, stride):
            # Calculate RMS for this window
            window_rms = np.sqrt(np.mean(
                np.square(data[self.primary_channel, i:i+window_size])
            ))
            rms_values.append(window_rms)
            
            # Calculate slope (change in RMS)
            if prev_rms is not None:
                slope_values.append(abs(window_rms - prev_rms))
            prev_rms = window_rms
        
        # Set thresholds
        self.rest_threshold = np.percentile(rms_values, self.rest_threshold_percentile)
        
        # Calculate mean RMS and other statistics
        self.mean_rms = np.mean(rms_values)
        self.median_rms = np.median(rms_values)
        self.min_rms = np.min(rms_values)
        self.max_rms = np.max(rms_values)
        
        # Calculate thresholds for slope-based detection
        self.mean_slope = np.mean(slope_values)
        self.max_slope_threshold = self.max_slope_factor * self.mean_rms
        
        # Maximum RMS for rest - this helps identify high flat periods that should be active
        self.max_rest_rms = np.percentile(rms_values, self.max_rest_rms_percentile)
        
        print(f"Initialized Rest/Active labeler:")
        print(f"- Primary channel: {self.primary_channel}")
        print(f"- Rest threshold: {self.rest_threshold:.6f} (percentile: {self.rest_threshold_percentile}%)")
        print(f"- Mean RMS: {self.mean_rms:.6f}")
        print(f"- Min RMS: {self.min_rms:.6f}")
        print(f"- Max RMS: {self.max_rms:.6f}")
        print(f"- Max rest RMS: {self.max_rest_rms:.6f} (percentile: {self.max_rest_rms_percentile}%)")
        print(f"- Mean slope: {self.mean_slope:.6f}")
        print(f"- Max slope threshold: {self.max_slope_threshold:.6f}")
        
        self.is_initialized = True
        self.rms_history = []
        self.slope_history = []
        
    def is_local_minimum(self, index, rms_values=None):
        """
        Check if the RMS at index is a local minimum within the surrounding window.
        
        Args:
            index: Index to check
            rms_values: Optional array of RMS values (uses self.rms_history if None)
            
        Returns:
            Boolean indicating if it's a local minimum
        """
        values = rms_values if rms_values is not None else self.rms_history
        
        if len(values) < 3:
            return False
            
        # Ensure we have enough data on both sides
        if index < 1 or index >= len(values) - 1:
            return False
        
        # Check if center is lower than immediate neighbors
        center_value = values[index]
        left_neighbor = values[index - 1]
        right_neighbor = values[index + 1]
        
        # Must be lower than both neighbors
        if not (center_value < left_neighbor and center_value < right_neighbor):
            return False
            
        # Calculate how much lower the center is compared to neighbors
        avg_neighbor = (left_neighbor + right_neighbor) / 2
        if avg_neighbor > 0:  # Avoid division by zero
            relative_depth = (avg_neighbor - center_value) / avg_neighbor
            return relative_depth > self.local_minima_margin
        return False
        
    def is_flat_region(self, index, rms_values=None):
        """
        Check if the region around a given index is flat (no sharp changes).
        
        Args:
            index: Index to check
            rms_values: Optional array of RMS values (uses self.rms_history if None)
            
        Returns:
            Boolean indicating if it's a flat region
        """
        values = rms_values if rms_values is not None else self.rms_history
        
        if len(values) < self.flatness_window:
            return False
            
        half_window = self.flatness_window // 2
        
        # Ensure we have enough data on both sides
        if index < half_window or index >= len(values) - half_window:
            return False
            
        # Get the window of RMS values centered at index
        window_start = max(0, index - half_window)
        window_end = min(len(values), index + half_window + 1)
        window = values[window_start:window_end]
        
        # Calculate slopes between consecutive points
        slopes = [abs(window[i] - window[i-1]) for i in range(1, len(window))]
        max_slope = max(slopes) if slopes else 0
        
        # Check if slopes are all below threshold
        is_flat = max_slope < self.max_slope_threshold
        
        # Also check that RMS values are not too high (not active)
        center_value = values[index]
        is_low_enough = center_value < self.max_rest_rms
        
        return is_flat and is_low_enough
    
    def is_rest(self, index, rms_values=None):
        """
        Determine if a window should be labeled as rest based on local minima and flatness.
        
        Args:
            index: Index to check
            rms_values: Optional array of RMS values (uses self.rms_history if None)
            
        Returns:
            Boolean indicating if it should be labeled as rest
        """
        # Rest is either a local minimum or a flat region with appropriate RMS level
        return (self.is_local_minimum(index, rms_values) or 
                self.is_flat_region(index, rms_values))
    
    def label_window(self, window: np.ndarray) -> str:
        """
        Label a single window as 'active' or 'rest'.
        
        Args:
            window: EMG window data of shape (n_channels, n_samples)
            
        Returns:
            'active' or 'rest'
        """
        if not self.is_initialized:
            raise ValueError("Labeler not initialized. Call initialize_from_data first.")
        
        # Calculate RMS for this window
        window_rms = np.sqrt(np.mean(np.square(window[self.primary_channel])))
        
        # Update history
        self.rms_history.append(window_rms)
        if len(self.rms_history) > self.history_size + self.flatness_window:
            self.rms_history.pop(0)
        
        # Calculate slope if we have history
        current_slope = 0
        if len(self.rms_history) >= 2:
            current_slope = abs(self.rms_history[-1] - self.rms_history[-2])
            self.slope_history.append(current_slope)
            if len(self.slope_history) > self.history_size:
                self.slope_history.pop(0)
        
        # Default to active
        label = 'active'
        
        # Check for rest conditions
        
        # 1. Check if RMS is too high (never rest)
        if window_rms >= self.max_rest_rms:
            return 'active'
            
        # 2. Check if it's a local minimum or flat region
        if len(self.rms_history) >= 3:  # Need at least 3 windows for local minimum
            idx = len(self.rms_history) - 1  # Most recent window
            
            if self.is_local_minimum(idx) or self.is_flat_region(idx):
                label = 'rest'
                
        return label
        
    def label_windows(self, windows: List[np.ndarray]) -> List[str]:
        """
        Label multiple windows, considering only local minima and flat regions as rest.
        
        Args:
            windows: List of EMG windows
            
        Returns:
            List of labels ('active' or 'rest')
        """
        # Reset state
        self.rms_history = []
        self.slope_history = []
        
        # Calculate RMS values for all windows
        rms_values = [np.sqrt(np.mean(np.square(window[self.primary_channel]))) 
                      for window in windows]
        
        # Initialize all windows as active by default
        labels = ['active'] * len(windows)
        
        # Step 1: Identify all rest periods (local minima and flat regions)
        rest_indices = []
        for i in range(1, len(rms_values) - 1):  # Skip first and last for safety
            if self.is_rest(i, rms_values):
                rest_indices.append(i)
        
        # Step 2: Mark rest periods
        for idx in rest_indices:
            # Mark the window itself as rest
            labels[idx] = 'rest'
            
            # Expand rest to neighboring windows if they're within expansion range
            for j in range(max(0, idx - self.rest_expansion), 
                          min(len(labels), idx + self.rest_expansion + 1)):
                # Only expand to neighbors if they're not above the maximum rest RMS
                if rms_values[j] < self.max_rest_rms:
                    labels[j] = 'rest'
        
        # Step 3: Ensure high RMS windows are never labeled as rest
        for i in range(len(labels)):
            if rms_values[i] >= self.max_rest_rms:
                labels[i] = 'active'
        
        # Step 4: Remove isolated windows (smoothing)
        smoothed_labels = labels.copy()
        for i in range(2, len(smoothed_labels) - 2):
            # Get the surrounding 4 windows (2 on each side)
            surroundings = [smoothed_labels[i-2], smoothed_labels[i-1], 
                           smoothed_labels[i+1], smoothed_labels[i+2]]
            
            # If surrounded mostly by the opposite label, change to match surroundings
            if smoothed_labels[i] == 'rest' and surroundings.count('active') >= 3:
                smoothed_labels[i] = 'active'
            elif smoothed_labels[i] == 'active' and surroundings.count('rest') >= 3:
                smoothed_labels[i] = 'rest'
                
        return smoothed_labels
    
    def visualize(self, data: np.ndarray, n_windows: int = 100):
        """
        Visualize the activity detection on a portion of the data.
        
        Args:
            data: EMG data of shape (n_channels, n_samples)
            n_windows: Number of windows to visualize
        """
        if not self.is_initialized:
            raise ValueError("Labeler not initialized. Call initialize_from_data first.")
        
        # Create windows from the data
        n_channels, n_samples = data.shape
        window_size = self.window_size
        stride = window_size // 2
        
        windows = []
        for i in range(0, min(n_samples - window_size, n_windows * stride), stride):
            windows.append(data[:, i:i+window_size])
        
        # Calculate RMS for visualization
        rms_values = [np.sqrt(np.mean(np.square(window[self.primary_channel]))) 
                      for window in windows[:n_windows]]
        
        # Calculate slopes with zero-padding for first element
        slope_values = [0] + [abs(rms_values[i] - rms_values[i-1]) 
                             for i in range(1, len(rms_values))]
        
        # Find flat regions and local minima for visualization
        flat_regions = []
        local_minima = []
        
        for i in range(1, len(rms_values) - 1):
            if self.is_local_minimum(i, rms_values):
                local_minima.append(i)
                
        for i in range(self.flatness_window, len(rms_values) - self.flatness_window):
            if self.is_flat_region(i, rms_values):
                flat_regions.append(i)
        
        # Label windows
        labels = self.label_windows(windows[:n_windows])
        
        # Print statistics about labeling
        active_count = labels.count('active')
        rest_count = labels.count('rest')
        print(f"Found {len(flat_regions)} flat regions and {len(local_minima)} local minima")
        print(f"Labeling statistics: {active_count} active windows, {rest_count} rest windows")
        
        # Plot results with distinct colors
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Raw signal
        time = np.arange(n_windows * stride + window_size) / 1000  # Assuming 1000 Hz
        signal_to_plot = data[self.primary_channel, :n_windows * stride + window_size]
        ax1.plot(time, signal_to_plot)
        ax1.set_title(f'Raw EMG Signal (Channel {self.primary_channel})')
        ax1.set_ylabel('Amplitude')
        
        # Plot 2: RMS values and thresholds
        window_times = np.arange(window_size/2, window_size/2 + n_windows * stride, stride) / 1000
        ax2.plot(window_times, rms_values, label='Window RMS')
        
        # Mark special regions
        for idx in flat_regions:
            ax2.plot(window_times[idx], rms_values[idx], 'go', markersize=6, alpha=0.7)
            
        for idx in local_minima:
            ax2.plot(window_times[idx], rms_values[idx], 'ro', markersize=6)
            
        # Show thresholds
        ax2.axhline(y=self.rest_threshold, color='r', linestyle='--', 
                    label=f'Rest Threshold ({self.rest_threshold_percentile}%)')
        ax2.axhline(y=self.max_rest_rms, color='b', linestyle='--', 
                    label=f'Max Rest RMS ({self.max_rest_rms_percentile}%)')
        
        # Color the background based on labels - make more visible
        for i, label in enumerate(labels):
            start = window_times[i] - window_size/2000  # Adjust for window size
            end = start + window_size/1000
            if label == 'active':
                ax2.axvspan(start, end, alpha=0.3, color='green')
            else:
                ax2.axvspan(start, end, alpha=0.3, color='red')  # Increased alpha for visibility
        
        ax2.set_title('Window RMS and Activity Detection')
        ax2.set_ylabel('RMS')
        
        # Add a legend for the markers and thresholds
        green_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Flat Region')
        red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Local Minimum')
        handles = [green_dot, red_dot]
        
        # Add the threshold lines to the legend
        rest_line = plt.Line2D([0], [0], color='r', linestyle='--', label=f'Rest Threshold ({self.rest_threshold_percentile}%)')
        max_rest_line = plt.Line2D([0], [0], color='b', linestyle='--', label=f'Max Rest RMS ({self.max_rest_rms_percentile}%)')
        handles.extend([rest_line, max_rest_line])
        
        ax2.legend(handles=handles, loc='upper right')
        
        # Plot 3: Slope values
        ax3.plot(window_times, slope_values, label='RMS Change')
        ax3.axhline(y=self.max_slope_threshold, color='r', linestyle='--', 
                    label=f'Max Slope Threshold for Rest')
        ax3.set_title('RMS Change Between Windows')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Absolute Change')
        ax3.legend()
        
        plt.tight_layout()
        return fig, labels