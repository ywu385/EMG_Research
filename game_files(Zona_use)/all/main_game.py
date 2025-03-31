import atexit
import glob
import pickle
import multiprocessing
import numpy as np
import time
import traceback
import pygame
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

# Queue for EMG communication
emg_queue = multiprocessing.Queue()

# EMG state variables
emg_process = None
latest_prediction = "rest"
latest_intensity = 0.1

def process_emg_data(mac_address, model_path, queue):
    """Process EMG data in a separate process"""
    print("EMG processing thread started")
    
    try:
        # Setup device
        device = BITalino(mac_address)
        device.battery(30)
        print("BITalino connected in processing thread")
        
        # Setup streamer
        streamer = BitaStreamer(device)
        print("Created BITalino streamer in processing thread")
        
        # Setup pipeline
        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
        pipeline.add_processor(DCRemover())
        pipeline.add_processor(AdaptiveMaxNormalizer())
        streamer.add_pipeline(pipeline)
        print("Pipeline added to streamer in processing thread")
        
        # Load model in this process
        with open(model_path, 'rb') as file:
            model, label_encoder = pickle.load(file)
        print(f"Model loaded from {model_path}")
        
        # Setup model processor
        model_processor = WideModelProcessor(
            model=model,
            window_size=250,
            overlap=0.5,
            sampling_rate=1000,
            n_predictions=5,
            label_encoder=label_encoder
        )
        
        # Setup buffer and intensity processor
        buffer = SignalBuffer(window_size=250, overlap=0.5)
        intensity_processor = IntensityProcessor(scaling_factor=1.5)
        
        print("Starting EMG data processing loop...")
        
        # Process EMG data continuously
        for chunk in streamer.stream_processed():
            # Process for prediction
            windows = buffer.add_chunk(chunk)
            intensity_value = None
            prediction = None
            
            for w in windows:
                prediction = model_processor.process(w)
                i_metrics = intensity_processor.process(w)
                
                if i_metrics['rms_values'] is not None and len(i_metrics['rms_values']) > 0:
                    min_speed, max_speed = 0, 10  # Define min/max speed range
                    norm_rms = np.array(i_metrics['rms_values']).max() / i_metrics['max_rms_ever']
                    intensity_value = min_speed + (norm_rms * (max_speed - min_speed))
            
            # Only when model buffer has enough data
            if prediction is not None:
                queue.put((prediction, intensity_value))
    
    except Exception as e:
        print(f"Error in EMG processing thread: {e}")
        traceback.print_exc()
    finally:
        # Ensure device is properly closed
        if 'device' in locals():
            try:
                device.close()
                print("BITalino device closed in processing thread")
            except:
                print("Error closing BITalino device in processing thread")

def start_emg_processing():
    """Start the EMG processing in a separate process"""
    global emg_process
    
    try:
        # Find model path
        model_path = glob.glob('models/*')
        if not model_path:
            print("No model files found")
            return False
        
        mac_address = "/dev/tty.BITalino-3C-C2"  # BITalino MAC address
        
        # Start EMG processing in a separate process
        emg_process = multiprocessing.Process(
            target=process_emg_data,
            args=(mac_address, model_path[0], emg_queue)
        )
        emg_process.daemon = True
        emg_process.start()
        
        # Short delay to ensure process has started
        time.sleep(2)
        
        if not emg_process.is_alive():
            print("EMG process failed to start")
            return False
        
        print("EMG processing initialized successfully")
        return True
    
    except Exception as e:
        print(f"Error starting EMG processing: {e}")
        traceback.print_exc()
        return False

def update_emg_state():
    """Update and return the latest EMG prediction and intensity"""
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

def shutdown_emg_processing():
    """Shutdown EMG processing"""
    global emg_process
    
    if emg_process is not None and emg_process.is_alive():
        print("Shutting down EMG processing thread...")
        emg_process.terminate()
        emg_process.join(timeout=1.0)
        print("EMG processing thread terminated")

def main():
    # Use EMG if available
    use_emg = EMG_MODULES_AVAILABLE
    
    # Start EMG processing if available
    if use_emg:
        print("Starting EMG processing...")
        emg_initialized = start_emg_processing()
        if not emg_initialized:
            print("Warning: EMG processing failed to start, falling back to keyboard mode")
            use_emg = False
        else:
            print("EMG processing running in background")
            # Register cleanup for EMG
            atexit.register(shutdown_emg_processing)
    else:
        print("EMG not available, running in keyboard-only mode")
    
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