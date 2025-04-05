import multiprocessing
import numpy as np
import time
import os
import signal
from datetime import datetime
from stream_processor_bit import *
from processors import *
from post_processing import *

# Try importing the EMG modules at global scope
try:
    from revolution_api.bitalino import *
    EMG_MODULES_AVAILABLE = True
    print("All EMG modules loaded successfully")
except ImportError as e:
    print(f"Error importing EMG modules: {e}")
    EMG_MODULES_AVAILABLE = False

# Initialize BiTalino at the global scope - "Start the car"
print('Loading Bitalino Device')
mac_address = "/dev/tty.BITalino-3C-C2"
device = BITalino(mac_address)
device.battery(10)
streamer = BitaStreamer(device)
# streamer = TXTStreamer('./data/combined.txt')  # for testing

pipeline = EMGPipeline()
pipeline.add_processor(ZeroChannelRemover())
pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
streamer.add_pipeline(pipeline)

# Global queue for data transfer
emg_queue = multiprocessing.Queue()

# Global flags for control
recording_flag = multiprocessing.Value('i', 0)  # 0 = not recording, 1 = recording
stop_program = multiprocessing.Value('i', 0)    # Flag to stop the program

# List to keep track of recorded files
recorded_files = []

# Function to convert NPY to TXT
def convert_npy_to_txt(input_file):
    """
    Convert a .npy file to a .txt file
    
    Parameters:
    -----------
    input_file : str
        Path to the .npy file
    
    Returns:
    --------
    str
        Path to the output txt file
    """
    # Determine output file name
    output_file = os.path.splitext(input_file)[0] + '.txt'
    
    # Load the numpy array
    data = np.load(input_file, allow_pickle=True)
    
    # Handle different data shapes appropriately
    if data.ndim == 1:
        # For 1D data, save as a single column
        np.savetxt(output_file, data, fmt='%.6f')
        print(f"Converted 1D array with {data.shape[0]} samples")
    elif data.ndim == 2:
        # For 2D data (like multi-channel EMG), save with channels as columns
        # Check orientation - if more rows than columns, transpose
        if data.shape[0] > data.shape[1]:
            # This might be time series data with time as rows and channels as columns
            np.savetxt(output_file, data, fmt='%.6f', delimiter='\t')
            print(f"Converted 2D array with shape {data.shape} (rows as time points)")
        else:
            # This is likely channels as rows, time as columns (common EMG format)
            # Transpose so each channel becomes a column
            np.savetxt(output_file, data.T, fmt='%.6f', delimiter='\t')
            print(f"Converted 2D array with {data.shape[0]} channels, {data.shape[1]} samples each")
    else:
        # For higher dimensions, reshape to 2D
        print(f"Warning: Array has {data.ndim} dimensions with shape {data.shape}")
        print("Flattening to 2D for text export")
        
        # Try to intelligently reshape based on the data structure
        reshaped = data.reshape(-1, data.shape[-1])
        np.savetxt(output_file, reshaped, fmt='%.6f', delimiter='\t')
    
    print(f"Saved to: {output_file}")
    return output_file

# Function to stream processed data to queue - defined at global scope
def output_queue(chunk_queue, streamer_obj):
    try:
        print("Starting data streaming process")
        for chunk in streamer_obj.stream_processed():
            try:
                # Always put data in the queue
                chunk_queue.put((chunk), block=False)
            except Exception as e:
                print(f"Error putting chunk in queue: {e}")
                time.sleep(0.1)
    except Exception as e:
        print(f"Stream processing error: {e}")
        print("Stream processing stopped.")

# Function to record data for a specific gesture
def record_gesture(queue, duration, output_file):
    print(f"Recording gesture for {duration} seconds to {output_file}")
    
    # Store data for all channels
    all_data = []
    
    # Recording start time
    start_time = time.time()
    end_time = start_time + duration
    
    # Clear any backlog in the queue
    while not queue.empty():
        queue.get(block=False)
    
    print("Queue cleared, starting to record fresh data...")
    
    # Record until duration is reached
    try:
        while time.time() < end_time:
            try:
                if not queue.empty():
                    # Get data chunk from queue
                    chunk = queue.get(block=False)
                    
                    # Append to our data collection
                    all_data.append(chunk)
                    
                    # Print progress indicator
                    elapsed = time.time() - start_time
                    print(f"\rRecording: {elapsed:.1f}/{duration} seconds - {int(elapsed/duration*100)}% complete", end="")
                else:
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.001)
            except Exception as e:
                print(f"\nError during recording: {e}")
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    
    print("\nRecording complete. Processing data...")
    
    # Process and save collected data
    try:
        # Convert to numpy array for easier processing
        if len(all_data) > 0:
            # Check the format of the data
            if isinstance(all_data[0], np.ndarray):
                # If the first chunk is already a numpy array
                if all_data[0].ndim == 2:
                    # For 2D arrays (multiple channels)
                    # Determine the number of channels
                    num_channels = all_data[0].shape[0]
                    
                    # Concatenate all chunks
                    concat_data = np.hstack(all_data)
                    
                    # Save to file
                    np.save(output_file, concat_data)
                    
                    print(f"Saved {concat_data.shape[1]} samples for {num_channels} channels to {output_file}")
                    print(f"Data shape: {concat_data.shape}")
                else:
                    # For 1D arrays (single channel or flattened data)
                    concat_data = np.concatenate(all_data)
                    np.save(output_file, concat_data)
                    print(f"Saved {len(concat_data)} samples to {output_file}")
            else:
                # If data is in some other format, convert to numpy array first
                data_array = np.array(all_data)
                np.save(output_file, data_array)
                print(f"Saved data with shape {data_array.shape} to {output_file}")
            
            # Add file to our list of recorded files
            recorded_files.append(output_file)
            
        else:
            print("No data collected during recording!")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    # Create output directory if it doesn't exist
    output_dir = "emg_recordings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start the data streaming process using the globally defined streamer
    # This is just "sitting in the car" - the car was already running
    output_process = multiprocessing.Process(
        target=output_queue, 
        args=(emg_queue, streamer), 
        daemon=True
    )
    output_process.start()
    
    print("\n" + "="*50)
    print("Car is already running (streamer was started globally)")
    print("="*50 + "\n")
    
    # Wait a moment for streaming process to initialize
    time.sleep(1)
    
    # Pre-defined gestures to record
    gestures = ["Upward", "Downward", "Left", "Right"]
    
    # Get recording parameters
    try:
        # Ask for duration once (assuming same duration for all gestures)
        duration = float(input("Enter recording duration in seconds for each gesture: "))
        if duration <= 0:
            raise ValueError("Duration must be positive")
        
        # Loop through each gesture
        for i, gesture in enumerate(gestures):
            print(f"\n--- Preparing to record {gesture} (Rider {i+1}) ---")
            
            # Wait for user to be ready
            input(f"Press Enter when ready to record {gesture}...")
            
            # Generate filename with timestamp and gesture name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/{gesture}_{timestamp}.npy"
            
            # Start recording this gesture
            record_gesture(emg_queue, duration, filename)
            
            print(f"Finished recording {gesture}")
            
            # Check if user wants to continue to the next gesture
            if i < len(gestures) - 1:
                cont = input(f"Continue to {gestures[i+1]}? (y/n): ").strip().lower()
                if cont != 'y':
                    print("Gesture recording session ended early by user")
                    break
        
        print("\nAll gestures recorded successfully!")
        
    except ValueError as e:
        print(f"Invalid input: {e}")
    
    # Ask if user wants to record additional gestures before stopping
    while True:
        another = input("\nRecord additional gestures? (y/n): ").strip().lower()
        if another == 'y':
            try:
                # Get gesture name
                gesture_name = input("Enter gesture name: ").strip()
                if not gesture_name:
                    gesture_name = "Custom_Gesture"
                
                # Get duration for this custom gesture
                duration = float(input(f"Enter recording duration for {gesture_name} in seconds: "))
                if duration <= 0:
                    raise ValueError("Duration must be positive")
                
                # Wait for user to be ready
                input(f"Press Enter when ready to record {gesture_name}...")
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{output_dir}/{gesture_name}_{timestamp}.npy"
                
                # Start recording
                record_gesture(emg_queue, duration, filename)
                
            except ValueError as e:
                print(f"Invalid input: {e}")
        elif another == 'n':
            break
        else:
            print("Please enter 'y' or 'n'")
    
    # Convert all recorded NPY files to TXT format
    if recorded_files:
        print("\n" + "="*50)
        print("CONVERTING NPY FILES TO TXT FORMAT")
        print("="*50)
        
        for i, npy_file in enumerate(recorded_files):
            print(f"[{i+1}/{len(recorded_files)}] Converting {os.path.basename(npy_file)}")
            txt_file = convert_npy_to_txt(npy_file)
        
        print("\nAll files converted successfully!")
    
    print("\nRecording session ended")
    print("Note: The BiTalino streamer remains initialized (car is still running)")
    print("You can run this program again without reinitializing the device")

if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    main()