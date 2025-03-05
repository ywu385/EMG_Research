import matplotlib
# Force matplotlib to use a GUI backend
matplotlib.use('TkAgg')  # You can try 'Qt5Agg' instead if TkAgg doesn't work

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

class FeaturePlotter:
    """
    Real-time plotter for EMG features using matplotlib
    with correct time axis for 1000 Hz data, 250 sample windows
    """
    def __init__(self, feature_names=None, max_points=100, num_channels=4, window_size=250, sampling_rate=1000):
        """
        Initialize the feature plotter
        
        Args:
            feature_names: List of features to plot (default: ['RMS'])
            max_points: Maximum number of points to display
            num_channels: Number of EMG channels
            window_size: Number of samples per window
            sampling_rate: Sampling rate in Hz
        """
        # Set default to just RMS if no features specified
        self.feature_names = feature_names if feature_names is not None else ['RMS']
        self.max_points = max_points
        self.num_channels = num_channels
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        
        # Calculate window duration in seconds
        self.window_duration = window_size / sampling_rate
        
        # Set up data storage
        self.data = {}
        for feature in self.feature_names:
            self.data[feature] = {}
            for channel in range(num_channels):
                self.data[feature][channel] = deque(maxlen=max_points)
                
        self.time_points = deque(maxlen=max_points)
        self.window_index = 0
        
        # Create subplots
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(len(self.feature_names), 1, 
                                         figsize=(10, 3*len(self.feature_names)),
                                         sharex=True)
        
        # Handle the case where we have only one feature
        if len(self.feature_names) == 1:
            self.axes = [self.axes]
            
        # Set up the figure
        self.fig.suptitle(f"EMG Features Over Time (Window Duration: {self.window_duration:.3f}s)", fontsize=16)
        
        # Configure each subplot
        self.lines = {}
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
        
        for i, feature in enumerate(self.feature_names):
            # Set up subplot
            self.axes[i].set_title(f"{feature}")
            self.axes[i].set_ylabel("Amplitude")
            if i == len(self.feature_names) - 1:
                self.axes[i].set_xlabel("Time (s)")
            
            # Create a line for each channel
            self.lines[feature] = []
            for ch in range(num_channels):
                line, = self.axes[i].plot([], [], 
                                        label=f"Channel {ch}",
                                        color=colors[ch % len(colors)])
                self.lines[feature].append(line)
            
            # Add legend
            self.axes[i].legend(loc='upper right')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
        plt.draw()
        plt.pause(0.1)  # Small pause to render the figure
        
    def update(self, features):
        """
        Update the plot with new feature data
        
        Args:
            features: Dictionary of features from FeatureExtractor
        """
        # Update time - accurately based on window duration
        current_time = self.window_index * self.window_duration
        self.time_points.append(current_time)
        self.window_index += 1
        
        # Update data for each feature and channel
        for feature_name in self.feature_names:
            for ch in range(self.num_channels):
                if ch in features and feature_name in features[ch]:
                    self.data[feature_name][ch].append(features[ch][feature_name])
        
        # Update plot lines
        for feature_name in self.feature_names:
            for ch in range(self.num_channels):
                if self.data[feature_name][ch]:  # If we have data for this channel
                    self.lines[feature_name][ch].set_data(list(self.time_points), 
                                                        list(self.data[feature_name][ch]))
        
        # Adjust axes limits if needed
        for i, feature_name in enumerate(self.feature_names):
            # X-axis: show the last N seconds of data
            display_seconds = 10  # Display last 10 seconds
            if self.time_points:
                current_time = self.time_points[-1]
                self.axes[i].set_xlim(max(0, current_time - display_seconds), 
                                    max(display_seconds, current_time + 0.5))
            
            # Y-axis: auto-scale based on data
            all_values = []
            for ch in range(self.num_channels):
                if self.data[feature_name][ch]:
                    all_values.extend(list(self.data[feature_name][ch]))
            
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
                self.axes[i].set_ylim(min_val - margin, max_val + margin)
        
        # Redraw the figure
        self.fig.canvas.draw()
        plt.pause(0.001)  # Small pause to render the updated plot

# Simple test to make sure the plotter works
if __name__ == "__main__":
    # Create a plotter with correct window size and sampling rate
    plotter = FeaturePlotter(
        feature_names=['RMS'], 
        num_channels=4,
        window_size=250,  # 250 samples per window
        sampling_rate=1000  # 1000 Hz sampling rate
    )
    
    # Generate some random data and update the plot
    for i in range(100):
        # Create mock features
        features = {}
        for ch in range(4):
            features[ch] = {'RMS': np.sin(i/4 + ch) + np.random.random() * 0.2}
        
        # Update the plot
        plotter.update(features)
        
        # Sleep to simulate real-time
        time.sleep(0.25)  # 1/4 second per window