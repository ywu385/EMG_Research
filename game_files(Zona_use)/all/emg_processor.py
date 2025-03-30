import numpy as np
import multiprocessing
import glob
import pickle
import time
import sys

# Import EMG processing modules
try:
    from stream_processor_bit import *
    print('stream processor imported')
    from processors import *
    print('processors imported')
    from post_processing import *
    print('post processing imported')
    EMG_MODULES_AVAILABLE = True
except ImportError:
    print("EMG modules not available. Running in keyboard-only mode.")
    EMG_MODULES_AVAILABLE = False

# Check if bitalino is available
BITALINO_AVAILABLE = True  # forcing bitalino to try
if BITALINO_AVAILABLE:
    try:
        from revolution_api.bitalino import *
        print("Bitalino module is available")
    except ImportError:
        BITALINO_AVAILABLE = False
        print("Failed to import bitalino module")

# Global queue for EMG predictions
emg_queue = multiprocessing.Queue()
emg_process = None
latest_prediction = "rest"
latest_intensity = 0.1

def intensity_calc(norm_rms, min_speed=0, max_speed=10):
    """Calculate intensity value from normalized RMS"""
    return min_speed + (norm_rms * (max_speed - min_speed))

def process_emg_data(macAddress, battery_threshold, output_queue):
    """Process EMG data in a separate process - entire connection is managed within this process"""
    counter = 0
    
    try:
        print("EMG processing thread started - connecting to device...")
        # Create the device connection INSIDE this process
        device = BITalino(macAddress)
        device.battery(battery_threshold)
        print(f"Connected to BITalino device, battery level: {device.battery(battery_threshold)}%")
        
        # Create streamer
        streamer = BitaStreamer(device)
        
        # Setup pipeline
        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
        pipeline.add_processor(DCRemover())
        
        bandpass = ButterFilter(cutoff=[20, 450], sampling_rate=1000, filter_type='bandpass', order=4)
        pipeline.add_processor(bandpass)
        pipeline.add_processor(AdaptiveMaxNormalizer())
        streamer.add_pipeline(pipeline)
        
        # Load model
        model_path = glob.glob('models/*')
        if not model_path:
            print("No model files found")
            return
            
        with open(model_path[0], 'rb') as file:
            model, label_encoder = pickle.load(file)
        
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
        
        # Main processing loop
        print("Starting EMG processing loop...")
        for chunk in streamer.stream_processed():
            try:
                # Process for prediction and intensity
                windows = buffer.add_chunk(chunk)
                intensity_value = None
                
                for w in windows:
                    prediction = model_processor.process(w)
                    i_metrics = intensity_processor.process(w)
                    
                    if i_metrics['rms_values'] is not None and len(i_metrics['rms_values']) > 0:
                        norm_rms = np.array(i_metrics['rms_values']).max() / i_metrics['max_rms_ever']
                        intensity_value = intensity_calc(norm_rms)
                
                # Only when model buffer has enough data
                if prediction is not None:
                    output_queue.put((prediction, intensity_value))
                    counter += 1
                    if counter % 100 == 0:
                        print(f"Processed {counter} EMG chunks")
            except Exception as e:
                print(f"Error processing chunk: {e}")
                # Continue with next chunk rather than breaking the loop
    
    except Exception as e:
        print(f"Error in EMG processing thread: {e}")
        import traceback
        traceback.print_exc()
        
        # Always try to close the device gracefully
        if 'device' in locals():
            try:
                device.close()
                print("BITalino device closed")
            except:
                pass

def initialize_emg_processing(bitalino=True):
    """Initialize EMG processing in a separate process"""
    global emg_process, EMG_MODULES_AVAILABLE
    
    if not EMG_MODULES_AVAILABLE:
        print("EMG modules not available, skipping initialization")
        return False
    
    try:
        if bitalino and BITALINO_AVAILABLE:
            # Bitalino configuration
            macAddress = "/dev/tty.BITalino-3C-C2"
            battery_threshold = 30
            
            # Create and start the process
            # Note: We're NOT creating the device here, just passing the address
            emg_process = multiprocessing.Process(
                target=process_emg_data,
                args=(macAddress, battery_threshold, emg_queue)
            )
            
            emg_process.daemon = True
            emg_process.start()
            
            # Give the process time to initialize
            time.sleep(1)
            
            if not emg_process.is_alive():
                print("EMG process failed to start")
                return False
                
            print("EMG processing initialized successfully")
            return True
            
        else:
            # Fall back to test data
            print("Bitalino not available or not requested, would initialize test data here")
            # Implementation for test data would go here
            return False
            
    except Exception as e:
        print(f"Error initializing EMG: {e}")
        import traceback
        traceback.print_exc()
        return False

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

def shutdown_emg_processing():
    """Shutdown EMG processing"""
    if emg_process is not None and emg_process.is_alive():
        print("Shutting down EMG processing thread...")
        emg_process.terminate()
        emg_process.join(timeout=1.0)
        print("EMG processing thread terminated")