#%%
import multiprocessing
import time
import numpy as np
from collections import deque
import os

# Dash and Plotly imports
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

try:
    from stream_processor_bit import *
    from processors import *
    from revolution_api.bitalino import *
    from post_processing import *
    EMG_MODULES_AVAILABLE = True
    print("All EMG modules loaded successfully")
except ImportError as e:
    print(f"Error importing EMG modules: {e}")
    EMG_MODULES_AVAILABLE = False

# Class to handle EMG data visualization
class EMGVisualizer:
    def __init__(self, data_queue, buffer_size=1000):
        self.data_queue = data_queue
        self.buffer_size = buffer_size
        self.signal_buffers = {}
        self.recording_data = []
        self.is_recording = False
        self.filename = "emg_recording.npy"
        self.app = self._create_app()
        
    def _create_app(self):
        """Create the Dash application"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("EMG Signal Monitor", style={'textAlign': 'center'}),
            
            # Display area for the graphs
            html.Div([
                dcc.Graph(id='emg-graph', style={'height': '70vh'}),
                dcc.Interval(
                    id='interval-component',
                    interval=50,  # in milliseconds (20 Hz update rate)
                    n_intervals=0
                )
            ]),
            
            # Controls for recording
            html.Div([
                html.Button('Start Recording', id='record-button', n_clicks=0, 
                            style={'backgroundColor': 'green', 'color': 'white', 'margin': '10px'}),
                html.Button('Save Recording', id='save-button', n_clicks=0,
                            style={'backgroundColor': 'blue', 'color': 'white', 'margin': '10px'}),
                dcc.Input(id='filename-input', type='text', value='emg_recording.npy', 
                          style={'margin': '10px', 'width': '200px'}),
                html.Div(id='status-message', style={'margin': '10px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
        ])
        
        # Register callbacks
        self._register_callbacks(app)
        
        return app
    
    def _register_callbacks(self, app):
        """Register the Dash callbacks"""
        
        # Callback to update the graph
        @app.callback(
            Output('emg-graph', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_graph(n_intervals):
            # Process all available data in the queue
            self._process_queue_data()
            
            # Create the figure with a subplot for each channel
            fig = go.Figure()
            
            if not self.signal_buffers:
                # No data yet
                fig.update_layout(
                    title="Waiting for EMG data...",
                    xaxis_title="Samples",
                    yaxis_title="Amplitude"
                )
                return fig
            
            # Create time axis (sample numbers)
            x_data = list(range(len(next(iter(self.signal_buffers.values())))))
            
            # Add each channel as a trace
            for i, buffer in self.signal_buffers.items():
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=list(buffer),
                    mode='lines',
                    name=f'Channel {i}'
                ))
            
            # Update layout
            fig.update_layout(
                title="EMG Signal Monitor",
                xaxis_title="Samples",
                yaxis_title="Amplitude",
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
                uirevision='constant'  # Keeps zoom level consistent between updates
            )
            
            return fig
        
        # Callback for recording buttons
        @app.callback(
            [Output('record-button', 'children'),
             Output('record-button', 'style'),
             Output('status-message', 'children')],
            [Input('record-button', 'n_clicks'),
             Input('save-button', 'n_clicks')],
            [State('record-button', 'children'),
             State('filename-input', 'value')]
        )
        def update_recording(rec_clicks, save_clicks, button_text, filename_value):
            # Get the ID of the component that triggered the callback
            ctx = dash.callback_context
            if not ctx.triggered:
                # No clicks yet
                return button_text, {'backgroundColor': 'green', 'color': 'white', 'margin': '10px'}, ""
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'record-button':
                # Toggle recording state
                self.is_recording = not self.is_recording
                
                if self.is_recording:
                    # Started recording
                    self.recording_data = []
                    return "Stop Recording", {'backgroundColor': 'red', 'color': 'white', 'margin': '10px'}, "Recording..."
                else:
                    # Stopped recording
                    return "Start Recording", {'backgroundColor': 'green', 'color': 'white', 'margin': '10px'}, f"Recorded {len(self.recording_data)} chunks"
            
            elif trigger_id == 'save-button':
                # Save the recording
                if not self.recording_data:
                    return button_text, {'backgroundColor': 'green' if button_text == "Start Recording" else 'red', 'color': 'white', 'margin': '10px'}, "No data to save!"
                
                # Update filename
                self.filename = filename_value
                if not self.filename.endswith('.npy'):
                    self.filename += '.npy'
                
                try:
                    # Concatenate all chunks
                    all_data = np.hstack(self.recording_data)
                    np.save(self.filename, all_data)
                    return button_text, {'backgroundColor': 'green' if button_text == "Start Recording" else 'red', 'color': 'white', 'margin': '10px'}, f"Saved to {self.filename} ({all_data.shape})"
                except Exception as e:
                    return button_text, {'backgroundColor': 'green' if button_text == "Start Recording" else 'red', 'color': 'white', 'margin': '10px'}, f"Error saving: {str(e)}"
            
            # Default return
            return button_text, {'backgroundColor': 'green', 'color': 'white', 'margin': '10px'}, ""
    
    def _process_queue_data(self):
        """Process all available data in the queue"""
        queue_empty = False
        while not queue_empty:
            try:
                # Get data from queue without blocking
                chunk = self.data_queue.get(block=False)
                
                # Initialize buffers if needed
                for i in range(chunk.shape[0]):
                    if i not in self.signal_buffers:
                        self.signal_buffers[i] = deque(maxlen=self.buffer_size)
                        # Fill buffer with zeros initially
                        self.signal_buffers[i].extend([0] * self.buffer_size)
                
                # Add new data to buffers
                for i in range(chunk.shape[0]):
                    for sample in chunk[i]:
                        self.signal_buffers[i].append(sample)
                
                # Add to recording if recording is active
                if self.is_recording:
                    self.recording_data.append(chunk.copy())
                    
            except:
                # Queue is empty
                queue_empty = True
    
    def run(self, debug=False, host='127.0.0.1', port=8050):
        """Run the Dash application"""
        self.app.run_server(debug=debug, host=host, port=port)

def output_queue(chunk_queue, streamer):
    """Function to stream data from BITalino to the queue"""
    print("Starting data stream process...")
    try:
        for chunk in streamer.stream_processed():
            try:
                # Put the chunk in the queue without blocking
                chunk_queue.put(chunk, block=False)
            except:
                # Queue is full, just continue
                pass
    except Exception as e:
        print(f"Streaming error: {e}")
    finally:
        print("Stream ended")

def main():
    """Main function to run the application"""
    print('Loading Bitalino Device')
    mac_address = "/dev/tty.BITalino-3C-C2"
    device = BITalino(mac_address)
    device.battery(10)
    
    streamer = BitaStreamer(device)
    pipeline = EMGPipeline()
    pipeline.add_processor(ZeroChannelRemover())
    pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
    streamer.add_pipeline(pipeline)
    
    # Create a multiprocessing queue for communication
    emg_queue = multiprocessing.Queue(maxsize=100)
    
    # Start the data acquisition process
    stream_process = multiprocessing.Process(
        target=output_queue, 
        args=(emg_queue, streamer)
    )
    # stream_process.daemon = True
    stream_process.start()
    
    # Create and run the visualizer
    visualizer = EMGVisualizer(emg_queue, buffer_size=1000)
    try:
        visualizer.run(debug=False)
    finally:
        # Clean up
        stream_process.terminate()
        stream_process.join()
        print("Application closed")

if __name__ == "__main__":
    # Set multiprocessing method to 'fork' on Unix or 'spawn' on Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()