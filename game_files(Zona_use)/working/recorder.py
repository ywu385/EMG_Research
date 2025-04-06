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

# Global variables for configuration
USE_BITALINO = True
INPUT_FILE = './data/combined.txt'  # Default file if not using BiTalino
mac_address = "/dev/tty.BITalino-3C-C2"  # BITalino MAC address

# Global queue for data transfer
emg_queue = multiprocessing.Queue()

# List to keep track of recorded files
recorded_files = []

# Function to convert NPY to TXT and delete the NPY file
def convert_npy_to_txt(input_file, delete_npy=True):
    """
    Convert a .npy file to a .txt file and optionally delete the NPY file
    
    Parameters:
    -----------
    input_file : str
        Path to the .npy file
    delete_npy : bool
        Whether to delete the NPY file after conversion
    
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
    
    # Delete the NPY file if requested
    if delete_npy:
        try:
            os.remove(input_file)
            print(f"Deleted NPY file: {input_file}")
        except Exception as e:
            print(f"Error deleting NPY file: {e}")
    
    return output_file

# IMPORTANT: Modified function to create and use the streamer inside the process
def stream_emg_data(chunk_queue, use_bitalino=True, input_file=None):
    """
    Stream EMG data from either BiTalino or a text file.
    This function creates its own streamer instance.
    """
    try:
        # Create the streamer inside this process
        if use_bitalino and EMG_MODULES_AVAILABLE:
            print('Creating BiTalino streamer within process')
            try:
                # Connect to device
                device = BITalino(mac_address)
                device.battery(10)
                print("BiTalino device connected")
                
                # Setup streamer
                streamer = BitaStreamer(device)
                print("BiTalino streamer created")
            except Exception as e:
                print(f"Error initializing BiTalino: {e}")
                print("Falling back to text streamer")
                streamer = TXTStreamer(input_file, simple=False)
        else:
            print('Creating text streamer within process')
            streamer = TXTStreamer(input_file, simple=False)
        
        # Setup pipeline
        # pipeline = EMGPipeline()
        # pipeline.add_processor(ZeroChannelRemover())
        # # pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
        # streamer.add_pipeline(pipeline)

        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        # pipeline.add_processor(NotchFilter([60], sampling_rate=1000)) 
        # pipeline.add_processor(DCRemover())
        # emg_bandpass = RealTimeButterFilter(
        #                     cutoff=[20, 450],  # Target the 20-450 Hz frequency range for EMG
        #                     sampling_rate=1000,  # Assuming 1000 Hz sampling rate
        #                     filter_type='bandpass',
        #                     order=4  # 4th order provides good balance between sharpness and stability
        #                 )
        # pipeline.add_processor(emg_bandpass)
        # pipeline.add_processor(AdaptiveMaxNormalizer())
        # pipeline.add_processor(MaxNormalizer())
        streamer.add_pipeline(pipeline)
        
        # Stream the data
        print("Starting data streaming process")
        for chunk in streamer.stream_processed():
            try:
                # Put data in the queue
                chunk_queue.put((chunk), block=False)
            except Exception as e:
                print(f"Error putting chunk in queue: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Stream processing error: {e}")
    finally:
        print("Stream processing stopped.")
        # Close device if it was created in this process
        if use_bitalino and EMG_MODULES_AVAILABLE and 'device' in locals():
            try:
                device.close()
                print("BiTalino device closed")
            except:
                print("Error closing BiTalino device")

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
    
    # Create and start the streaming process - now it creates its own streamer
    streamer_process = multiprocessing.Process(
        target=stream_emg_data, 
        args=(emg_queue, USE_BITALINO, INPUT_FILE),
        daemon=True
    )
    streamer_process.start()
    
    print("\n" + "="*50)
    print("Starting EMG recording system")
    print("="*50 + "\n")
    
    # Wait a moment for streaming process to initialize
    time.sleep(2)  # Give it a bit more time to initialize
    
    # Pre-defined gestures to record
    gestures = ["upward", "downward", "left", "right"]
    
    # Get recording parameters
    try:
        # Ask for duration once (assuming same duration for all gestures)
        duration = float(input("Enter recording duration in seconds for each gesture: "))
        if duration <= 0:
            raise ValueError("Duration must be positive")
        name = str(input("Enter Name of participant: "))
        
        # Loop through each gesture
        for i, gesture in enumerate(gestures):
            print(f"\n--- Preparing to record {gesture} ({name}) ---")
            
            # Wait for user to be ready
            input(f"Press Enter when ready to record {gesture}...")
            
            # Generate filename with timestamp and gesture name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/{name}_{timestamp}_{gesture}.npy"
            
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
                filename = f"{output_dir}/{name}_{gesture_name}_{timestamp}.npy"
                
                # Start recording
                record_gesture(emg_queue, duration, filename)
                
            except ValueError as e:
                print(f"Invalid input: {e}")
        elif another == 'n':
            break
        else:
            print("Please enter 'y' or 'n'")
    
    # Convert all recorded NPY files to TXT format and delete NPY files
    if recorded_files:
        print("\n" + "="*50)
        print("CONVERTING NPY FILES TO TXT FORMAT")
        print("="*50)
        
        for i, npy_file in enumerate(recorded_files):
            print(f"[{i+1}/{len(recorded_files)}] Converting {os.path.basename(npy_file)}")
            txt_file = convert_npy_to_txt(npy_file, delete_npy=True)
        
        print("\nAll files converted successfully and NPY files deleted!")
    
    print("\nRecording session ended")
    
    # Clean up
    if streamer_process.is_alive():
        print("Terminating streamer process...")
        streamer_process.terminate()
        streamer_process.join(timeout=1.0)
        print("Streamer process terminated")

if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    main()