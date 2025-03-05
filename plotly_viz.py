import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any
import plotly.io as pio

# Set Plotly to render in browser by default
pio.renderers.default = "browser"

class PlotlyFeatureVisualizer:
    """
    Visualize EMG features using Plotly
    
    This class maintains a history of feature values and provides
    interactive visualization using Plotly.
    """
    
    def __init__(self, max_history=500, update_interval=10):
        """
        Initialize the feature visualizer
        
        Args:
            max_history: Maximum number of data points to keep in history
            update_interval: How often to update the plot (in terms of number of data points)
        """
        self.max_history = max_history
        self.update_interval = update_interval
        self.feature_history = {}
        self.time_points = []
        self.current_time = 0
        self.update_count = 0
        
        # Create Plotly figure
        self.fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=("EMG RMS Values Over Time", "All Features for Selected Channel"),
            vertical_spacing=0.1,
            specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
        )
        
        # Style the figure
        self.fig.update_layout(
            height=800,
            showlegend=True,
            title_text="EMG Feature Visualization",
            title_x=0.5,
            template="plotly_white",
            uirevision=True  # Keep zoom level when updating
        )
        
        # Store trace indices for updates
        self.rms_traces = {}  # Channel -> trace_idx mapping for RMS
        self.feature_traces = {}  # Feature -> trace_idx mapping
        
        # Currently selected channel for detailed view
        self.selected_channel = 0
        
    def _initialize_plot(self, features: Dict[int, Dict[str, Any]]):
        """
        Initialize the plot with the first data point
        
        Args:
            features: Dictionary of features by channel
        """
        # Get available channels and features
        channels = list(features.keys())
        feature_names = list(features[channels[0]].keys()) if channels else []
        
        # Initialize feature history
        for channel in channels:
            if channel not in self.feature_history:
                self.feature_history[channel] = {feature: [] for feature in feature_names}
        
        # Add RMS traces for each channel (top plot)
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        for i, channel in enumerate(channels):
            color = colors[i % len(colors)]
            trace = self.fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[features[channel]['RMS']],
                    mode='lines',
                    name=f'Channel {channel} RMS',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
            self.rms_traces[channel] = len(self.fig.data) - 1
            
        # Add traces for all features of the first channel (bottom plot)
        self.selected_channel = channels[0] if channels else 0
        for feature in feature_names:
            trace = self.fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[features[self.selected_channel][feature]],
                    mode='lines',
                    name=f'{feature}',
                    visible='legendonly' if feature != 'RMS' else True
                ),
                row=2, col=1
            )
            self.feature_traces[feature] = len(self.fig.data) - 1
            
        # Initialize time points
        self.time_points = [0]
        self.current_time = 0
        
        # Set axis labels
        self.fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        self.fig.update_yaxes(title_text="RMS Value", row=1, col=1)
        self.fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        self.fig.update_yaxes(title_text="Feature Value", row=2, col=1)
        
    def update(self, features: Dict[int, Dict[str, Any]], time_step=0.01, force_update=False):
        """
        Update the feature history and plot
        
        Args:
            features: Dictionary of features by channel
            time_step: Time step between data points (seconds)
            force_update: Force plot update regardless of update_interval
            
        Returns:
            The figure object if an update occurred, None otherwise
        """
        # Initialize plot if this is the first update
        if not self.feature_history:
            self._initialize_plot(features)
            
        # Update time
        self.current_time += time_step
        self.time_points.append(self.current_time)
        
        # Limit history length
        if len(self.time_points) > self.max_history:
            self.time_points.pop(0)
            
        # Update feature history
        for channel, channel_features in features.items():
            for feature, value in channel_features.items():
                if channel in self.feature_history and feature in self.feature_history[channel]:
                    self.feature_history[channel][feature].append(value)
                    if len(self.feature_history[channel][feature]) > self.max_history:
                        self.feature_history[channel][feature].pop(0)
        
        # Update counter
        self.update_count += 1
        
        # Only update the figure periodically or when forced
        if self.update_count % self.update_interval == 0 or force_update:
            # Update RMS traces for each channel (top plot)
            for channel, trace_idx in self.rms_traces.items():
                self.fig.data[trace_idx].x = self.time_points
                self.fig.data[trace_idx].y = self.feature_history[channel]['RMS']
                
            # Update feature traces for selected channel (bottom plot)
            for feature, trace_idx in self.feature_traces.items():
                self.fig.data[trace_idx].x = self.time_points
                self.fig.data[trace_idx].y = self.feature_history[self.selected_channel][feature]
            
            # Return the figure for display
            return self.fig
            
        return None
    
    def set_selected_channel(self, channel):
        """
        Change the channel for detailed feature view in the bottom plot
        
        Args:
            channel: Channel index to display
        """
        if channel in self.feature_history:
            self.selected_channel = channel
            # Update the bottom plot with the new channel's data
            for feature, trace_idx in self.feature_traces.items():
                self.fig.data[trace_idx].y = self.feature_history[channel][feature]
            
            # Update the subplot title
            self.fig.layout.annotations[1].text = f"All Features for Channel {channel}"
            return self.fig
            
        return None
    
    def show(self):
        """Show the current figure"""
        self.fig.show()