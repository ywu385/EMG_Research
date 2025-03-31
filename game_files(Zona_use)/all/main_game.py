import atexit
import glob
import pickle
import multiprocessing
import numpy as np
import time
from game_manager import GameManager

# Import necessary modules for EMG processing
try:
    from stream_processor_bit import *
    print('stream processor imported')
    from processors import *
    print('processors imported')
    from post_processing import *
    print('post processing imported')
    from revolution_api.bitalino import *
    print("Bitalino module is available")
    EMG_MODULES_AVAILABLE = True
except ImportError:
    print("EMG modules not available. Running in keyboard-only mode.")
    EMG_MODULES_AVAILABLE = False

# Global variables for EMG processing
emg_queue = multiprocessing.Queue()
emg_process = None
latest_prediction = "rest"
latest_intensity = 0.1

# Function to calculate intensity value from normalized RMS
def intensity_calc(norm_rms, min_speed=0, max_speed=10):
    """Calculate intensity value from normalized RMS"""
    return min_speed + (norm_rms * (max_speed - min_speed))

# EMG prediction function to run in a separate process
def output_predictions(model_processor, chunk_queue):
    """Process EMG data continuously and put results in the queue"""
    print("EMG processing thread started")
    counter = 0
    
    try:
        while True:
            for chunk in streamer.stream_processed():
                # Process for prediction
                windows = buffer.add_chunk(chunk)
                intensity_value = None
                prediction = None
                
                for w in windows:
                    prediction = model_processor.process(w)
                    i_metrics = intensity_processor.process(w)
                    
                    if i_metrics['rms_values'] is not None and len(i_metrics['rms_values']) > 0:
                        norm_rms = np.array(i_metrics['rms_values']).max() / i_metrics['max_rms_ever']
                        intensity_value = intensity_calc(norm_rms)
                
                # Only when model buffer has enough data
                if prediction is not None:
                    print(f"Prediction: {prediction}")
                    chunk_queue.put((prediction, intensity_value))
                    counter += 1
                    if counter % 100 == 0:
                        print(f"Processed {counter} EMG chunks")
    except Exception as e:
        print(f"Error in EMG processing thread: {e}")
        import traceback
        traceback.print_exc()

# Function to start EMG processing
def start_emg_processing():
    """Start the EMG processing in a separate process"""
    global emg_process
    
    # Start EMG processing in a separate process
    emg_process = multiprocessing.Process(
        target=output_predictions,
        args=(model_processor, emg_queue)
    )
    emg_process.daemon = True
    emg_process.start()
    
    # Short delay to ensure process has started
    time.sleep(1)
    
    if not emg_process.is_alive():
        print("EMG process failed to start")
        return False
    
    print("EMG processing initialized successfully")
    return True

# Function to update EMG state
def update_emg_state():
    """Update the latest EMG prediction and intensity"""
    global latest_prediction, latest_intensity
    
    if not emg_queue.empty():
        try:
            prediction, intensity = emg_queue.get_nowait()
            latest_prediction = prediction
            
            if intensity is not None:
                # Apply smoothing
                latest_intensity = latest_intensity * 0.85 + intensity * 0.15
        except Exception as e:
            print(f"Error updating EMG state: {e}")
    
    # Return current state
    return latest_prediction, latest_intensity

# Function to shutdown EMG processing
def shutdown_emg_processing():
    """Shutdown EMG processing"""
    global emg_process, device
    
    if emg_process is not None and emg_process.is_alive():
        print("Shutting down EMG processing thread...")
        emg_process.terminate()
        emg_process.join(timeout=1.0)
        print("EMG processing thread terminated")
    
    # Close the device connection if it exists
    if 'device' in globals():
        try:
            device.close()
            print("BITalino device closed")
        except Exception as e:
            print(f"Error closing BITalino device: {e}")

def main():
    global device, streamer, model_processor, buffer, intensity_processor
    
    # Initialize the EMG processor first (before pygame)
    use_emg = True
    
    if use_emg and EMG_MODULES_AVAILABLE:
        try:
            # BITalino configuration
            batteryThreshold = 30
            macAddress = "/dev/tty.BITalino-3C-C2"
            
            # Connect to the device
            print(f"Connecting to BITalino device at {macAddress}...")
            device = BITalino(macAddress)
            battery = device.battery(batteryThreshold)
            print(f"Connected to BITalino device. Battery: {battery}%")
            
            # Create streamer
            streamer = BitaStreamer(device)
            print("BITalino streamer created")
            
            # Setup pipeline
            pipeline = EMGPipeline()
            pipeline.add_processor(ZeroChannelRemover())
            pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
            pipeline.add_processor(DCRemover())
            
            bandpass = ButterFilter(cutoff=[20, 450], sampling_rate=1000, filter_type='bandpass', order=4)
            pipeline.add_processor(bandpass)
            pipeline.add_processor(AdaptiveMaxNormalizer())
            streamer.add_pipeline(pipeline)
            print("EMG pipeline created and added to streamer")
            
            # Load model
            model_path = glob.glob('models/*')
            if not model_path:
                print("No model files found")
                use_emg = False
            else:
                with open(model_path[0], 'rb') as file:
                    model, label_encoder = pickle.load(file)
                print("Model loaded successfully")
                
                # Setup model processor
                model_processor = WideModelProcessor(
                    model=model,
                    window_size=250,
                    overlap=0.5,
                    sampling_rate=1000,
                    n_predictions=5,
                    label_encoder=label_encoder
                )
                print("Model processor created")
                
                # Setup buffer and intensity processor
                buffer = SignalBuffer(window_size=250, overlap=0.5)
                intensity_processor = IntensityProcessor(scaling_factor=1.5)
                print("Buffer and intensity processor created")
                
                # Start the processing
                emg_initialized = start_emg_processing()
                if not emg_initialized:
                    print("Warning: EMG processing failed to start, falling back to keyboard mode")
                    use_emg = False
                else:
                    print("EMG processing running in background")
                    # Register cleanup for EMG
                    atexit.register(shutdown_emg_processing)
        except Exception as e:
            print(f"Error initializing EMG: {e}")
            import traceback
            traceback.print_exc()
            use_emg = False
    else:
        use_emg = False
        print("EMG modules not available, running in keyboard-only mode")
    
    # Create and run the game manager
    game_manager = GameManager(1280, 720, use_emg=use_emg)
    
    # If using EMG, update the game manager to use our update function
    if use_emg:
        game_manager.update_emg_state = update_emg_state
    
    # Run the game
    game_manager.run()
    
    # Clean up (in addition to atexit handlers)
    if use_emg:
        shutdown_emg_processing()

if __name__ == "__main__":
    main()