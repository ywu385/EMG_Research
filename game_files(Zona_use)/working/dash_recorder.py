import multiprocessing
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import time
import argparse
import os

# Flag to track data source
USE_BITALINO = True
INPUT_FILE = None
DEFAULT_DATA_FILE = "data/combined.txt"  # Default file path

# Import your existing modules if using BiTalino
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

# Maximum number of points to display
max_points = 500

# Function to stream processed data to queue - DEFINE THIS OUTSIDE MAIN
def output_queue(chunk_queue, streamer):
    for chunk in streamer.stream_processed():
        chunk_queue.put((chunk), block=False)

# Function to continuously pull data from the queue - DEFINE THIS OUTSIDE MAIN
def queue_listener(queue):
    # Initialize empty array for each channel
    # Assuming 4 channels as mentioned
    num_channels = 4
    data_array = [np.array([]) for _ in range(num_channels)]
    
    # Temporary file for data exchange
    temp_data_file = 'temp_emg_data.npy'
    
    while True:
        try:
            if not queue.empty():
                chunk = queue.get(block=False)
                
                # Check if chunk is properly shaped
                if isinstance(chunk, np.ndarray) and chunk.ndim == 2:
                    # Process each channel separately
                    for i in range(min(num_channels, chunk.shape[0])):
                        # Update channel data array
                        data_array[i] = np.append(data_array[i], chunk[i])
                        
                        # Keep only the most recent data points
                        if len(data_array[i]) > max_points:
                            data_array[i] = data_array[i][-max_points:]
                else:
                    # Flat chunk - assume it's a single channel or handle as needed
                    chunk_flat = np.array(chunk).flatten()
                    data_array[0] = np.append(data_array[0], chunk_flat)
                    if len(data_array[0]) > max_points:
                        data_array[0] = data_array[0][-max_points:]
                    
                    print(f"Warning: Unexpected chunk shape. Expected 2D array, got {type(chunk)} with shape {getattr(chunk, 'shape', 'unknown')}")
                
                # Save to temp file for dash to read
                np.save(temp_data_file, data_array)
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"Error getting data from queue: {e}")
            time.sleep(0.01)

# Parse command line arguments
def parse_args():
    global USE_BITALINO, INPUT_FILE
    
    parser = argparse.ArgumentParser(description='EMG Recorder')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--bitalino', action='store_true', help='Use BiTalino device (default)')
    group.add_argument('--file', type=str, help='Use text file as input source')
    group.add_argument('--default-file', action='store_true', help='Use default text file')
    
    args = parser.parse_args()
    
    if args.file:
        USE_BITALINO = False
        INPUT_FILE = args.file
        print(f"Using specified text file as data source: {INPUT_FILE}")
    elif args.default_file:
        USE_BITALINO = False
        INPUT_FILE = DEFAULT_DATA_FILE
        if not os.path.exists(DEFAULT_DATA_FILE):
            print(f"Warning: Default file {DEFAULT_DATA_FILE} does not exist!")
            print("Please check the path or create this file.")
            exit(1)
        print(f"Using default text file as data source: {INPUT_FILE}")
    else:
        USE_BITALINO = True
        print("Using BiTalino as data source")

def init_streamer():
    # Setup the appropriate streamer based on source configuration
    if USE_BITALINO and EMG_MODULES_AVAILABLE:
        print('Loading Bitalino Device')
        try:
            mac_address = "/dev/tty.BITalino-3C-C2"
            device = BITalino(mac_address)
            device.battery(10)
            
            # Setup the pipeline in global scope
            streamer = BitaStreamer(device)
        except Exception as e:
            print(f"Error initializing BiTalino: {e}")
            print("Falling back to text streamer mode")
            file_to_use = DEFAULT_DATA_FILE
            streamer = TXTStreamer(file_to_use)
    else:
        print('Loading text streamer')
        file_to_use = INPUT_FILE if INPUT_FILE else DEFAULT_DATA_FILE
        streamer = TXTStreamer(file_to_use)

    # Setup common pipeline
    pipeline = EMGPipeline()
    pipeline.add_processor(ZeroChannelRemover())
    pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
    streamer.add_pipeline(pipeline)
    
    return streamer

def create_app():
    # Initialize Dash components
    app = dash.Dash(__name__)
    
    # Initialize with empty array for each channel
    num_channels = 4
    np.save('temp_emg_data.npy', [np.array([]) for _ in range(num_channels)])

    app.layout = html.Div([
        html.H1("EMG Monitor"),
        html.Div([
            html.Div([
                html.H3("Data Source:"),
                html.P(id="data-source-display")
            ], style={'margin-bottom': '20px'}),
            html.Div([
                html.Label("Select Channel:"),
                dcc.Dropdown(
                    id='channel-selector',
                    options=[
                        {'label': f'Channel {i+1}', 'value': i} for i in range(4)
                    ],
                    value=0,  # Default to first channel
                    clearable=False
                ),
            ], style={'width': '250px', 'margin-bottom': '20px'}),
            dcc.Graph(id='live-emg-graph', animate=True),
            dcc.Interval(
                id='interval-component',
                interval=100,  # in milliseconds
                n_intervals=0
            )
        ])
    ])

    # Callback to update the graph
    @app.callback(
        Output('live-emg-graph', 'figure'),
        [Input('interval-component', 'n_intervals'),
         Input('channel-selector', 'value')]
    )
    def update_graph(n, selected_channel):
        try:
            # Read data from temporary file
            if os.path.exists('temp_emg_data.npy'):
                all_channel_data = np.load('temp_emg_data.npy', allow_pickle=True)
                if selected_channel < len(all_channel_data):
                    current_data = all_channel_data[selected_channel]
                else:
                    current_data = np.array([])
            else:
                current_data = np.array([])
        except Exception as e:
            print(f"Error reading data: {e}")
            current_data = np.array([])
        
        # Create a time array for x-axis
        time_array = np.arange(len(current_data))
        
        # Create the graph
        trace = go.Scatter(
            x=time_array,
            y=current_data,
            mode='lines',
            name=f'Channel {selected_channel + 1}'
        )
        
        return {
            'data': [trace],
            'layout': go.Layout(
                title=f'EMG Signal - Channel {selected_channel + 1}',
                xaxis=dict(title='Samples', range=[max(0, len(current_data)-max_points), len(current_data)]),
                yaxis=dict(title='Amplitude')
            )
        }

    # Callback to display the data source
    @app.callback(
        Output('data-source-display', 'children'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_data_source(n):
        if USE_BITALINO:
            return "BiTalino Device"
        else:
            source_file = INPUT_FILE if INPUT_FILE else DEFAULT_DATA_FILE
            return f"Text File: {os.path.basename(source_file)}"
    
    return app

# Main function
def main():
    # Parse arguments
    parse_args()
    
    # Initialize streamer
    streamer = init_streamer()
    
    # Setup multiprocessing queue
    emg_queue = multiprocessing.Queue()
    
    # Create dash app
    app = create_app()
    
    # Start the listener process
    listener_process = multiprocessing.Process(
        target=queue_listener, 
        args=(emg_queue,),
        daemon=True
    )
    listener_process.start()
    
    # Start the data streaming process
    output_process = multiprocessing.Process(
        target=output_queue, 
        args=(emg_queue, streamer), 
        daemon=True
    )
    output_process.start()
    
    # Run the Dash app
    app.run_server(debug=False)

if __name__ == '__main__':
    # Add multiprocessing freeze_support for Windows compatibility
    multiprocessing.freeze_support()
    
    # Call main function
    main()