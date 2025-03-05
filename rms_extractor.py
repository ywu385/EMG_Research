import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def extract_rms_values(feature_list):
    """
    Extract RMS values from the nested feature list
    
    Args:
        feature_list: List of dictionaries containing features per channel
        
    Returns:
        Dictionary mapping channel numbers to lists of RMS values
    """
    channels = {}
    
    # First, determine what channels exist in the data
    for window_features in feature_list:
        for channel in window_features.keys():
            if channel not in channels:
                channels[channel] = []
    
    # Now extract RMS values for each channel across all windows
    for window_features in feature_list:
        for channel in channels.keys():
            if channel in window_features:
                channels[channel].append(window_features[channel]['RMS'])
            else:
                # Handle missing channels (though this shouldn't happen)
                channels[channel].append(None)
    
    return channels

def plot_rms_values(rms_by_channel, sampling_rate=1000, window_size=250):
    """
    Plot RMS values for each channel
    
    Args:
        rms_by_channel: Dictionary mapping channel numbers to lists of RMS values
        sampling_rate: Original sampling rate in Hz
        window_size: Window size in samples
    
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Calculate time points (in seconds)
    # Each window is window_size/sampling_rate seconds
    # Windows are non-overlapping, so each window advances by window_size/sampling_rate
    window_duration = window_size / sampling_rate
    
    # Define colors for different channels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add traces for each channel
    for i, (channel, rms_values) in enumerate(rms_by_channel.items()):
        # Calculate time points
        time_points = [j * window_duration for j in range(len(rms_values))]
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=time_points,
            y=rms_values,
            mode='lines+markers',
            name=f'Channel {channel}',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=5)
        ))
    
    # Update layout
    fig.update_layout(
        title='RMS Values Over Time by Channel',
        xaxis_title='Time (seconds)',
        yaxis_title='RMS Value',
        legend_title='Channel',
        template='plotly_white',
        height=600,
        width=1000,
        hovermode='x unified'
    )
    
    # Add vertical lines every 1 second
    max_rms = max([max(values) for values in rms_by_channel.values()])
    for i in range(1, int(np.ceil(len(list(rms_by_channel.values())[0]) * window_duration))):
        fig.add_shape(
            type="line",
            x0=i,
            y0=0,
            x1=i,
            y1=max_rms * 1.1,
            line=dict(color="gray", width=1, dash="dash"),
        )
        # Add annotation
        fig.add_annotation(
            x=i,
            y=max_rms * 1.05,
            text=f"{i}s",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Configure axes for better zooming
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    
    return fig

# Example usage:
# feature_list = [...] # Your feature list
# rms_by_channel = extract_rms_values(feature_list)
# fig = plot_rms_values(rms_by_channel)
# fig.show()