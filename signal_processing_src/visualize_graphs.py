import multiprocessing
import numpy as np
import time
import collections
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

def plotting_process(metrics_queue, stop_event, max_points=500):
    """Process to plot metrics from the metrics queue using PyQtGraph"""
    # Create the application
    app = QtGui.QApplication([])
    
    # Create window with GraphicsLayoutWidget
    win = pg.GraphicsLayoutWidget(show=True, title="EMG Metrics")
    win.resize(1000, 600)
    
    # Set up plots for RMS, Mean Absolute Value, and Variance
    rms_plot = win.addPlot(row=0, col=0, title="Root Mean Square (RMS)")
    win.nextRow()
    mean_abs_plot = win.addPlot(row=1, col=0, title="Mean Absolute Value")
    win.nextRow()
    var_plot = win.addPlot(row=2, col=0, title="Variance")
    
    # Add legends
    rms_plot.addLegend()
    mean_abs_plot.addLegend()
    var_plot.addLegend()
    
    # Add axis labels
    var_plot.setLabel('bottom', "Time", "s")
    rms_plot.setLabel('left', "Amplitude")
    mean_abs_plot.setLabel('left', "Amplitude")
    var_plot.setLabel('left', "Amplitude")
    
    # Set up colors for different channels
    colors = ['r', 'g', 'b', 'c']
    num_channels = 4  # Adjust based on your setup
    
    # Set up data structures
    timestamps = collections.deque(maxlen=max_points)
    
    # Data for each channel
    rms_data = [collections.deque(maxlen=max_points) for _ in range(num_channels)]
    mean_abs_data = [collections.deque(maxlen=max_points) for _ in range(num_channels)]
    variance_data = [collections.deque(maxlen=max_points) for _ in range(num_channels)]
    
    # Create curve objects for each channel
    rms_curves = []
    mean_abs_curves = []
    var_curves = []
    
    for i in range(num_channels):
        rms_curves.append(rms_plot.plot(pen=colors[i % len(colors)], name=f"Channel {i+1}"))
        mean_abs_curves.append(mean_abs_plot.plot(pen=colors[i % len(colors)], name=f"Channel {i+1}"))
        var_curves.append(var_plot.plot(pen=colors[i % len(colors)], name=f"Channel {i+1}"))
    
    # Initialize x-axis for auto-ranging
    rms_plot.enableAutoRange(axis='x')
    mean_abs_plot.enableAutoRange(axis='x')
    var_plot.enableAutoRange(axis='x')
    
    # Function to update plots
    def update():
        # Try to get all available metrics from the queue (non-blocking)
        while not metrics_queue.empty():
            try:
                metrics = metrics_queue.get_nowait()
                
                # Add the timestamp
                timestamps.append(metrics['time'])
                
                # Add data for each channel
                for i in range(min(num_channels, len(metrics['rms']))):
                    rms_data[i].append(metrics['rms'][i])
                    mean_abs_data[i].append(metrics['mean_abs'][i])
                    variance_data[i].append(metrics['variance'][i])
            except:
                break
        
        # Update the curve data
        for i in range(num_channels):
            # Only update if we have data for this channel
            if rms_data[i]:
                rms_curves[i].setData(list(timestamps), list(rms_data[i]))
            if mean_abs_data[i]:
                mean_abs_curves[i].setData(list(timestamps), list(mean_abs_data[i]))
            if variance_data[i]:
                var_curves[i].setData(list(timestamps), list(variance_data[i]))
        
        # Check if we should stop
        if stop_event.is_set():
            app.quit()
    
    # Set up a timer for updating the plots
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)  # Update every 50 ms
    
    # Start the Qt event loop
    app.exec_()
    print("PyQtGraph plotting process terminated")