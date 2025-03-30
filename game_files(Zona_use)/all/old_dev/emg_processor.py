#%%
import numpy as np
import multiprocessing
import glob
import pickle
import time
# import importlib.util
import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#%%
# Import EMG processing modules
try:
    from stream_processor_bit import *
    print('stream proceesor imported')
    from processors import *
    print('processors imported')
    from post_processing import *
    print('post processing imported')
    EMG_MODULES_AVAILABLE = True
except ImportError:
    print("EMG modules not available. Running in keyboard-only mode.")
    EMG_MODULES_AVAILABLE = False

# Check if bitalino is available
# BITALINO_AVAILABLE = importlib.util.find_spec('bitalino') is not None
BITALINO_AVAILABLE = True  # forcing bitalino to try and import from folder directory
if BITALINO_AVAILABLE:
    try:
        from revolution_api.bitalino import *
        print("Bitalino module is available")
    except ImportError:
        BITALINO_AVAILABLE = False
        print("Failed to import bitalino module")
#%%
# # Global queue for EMG predictions
emg_queue = multiprocessing.Queue()
emg_process = None
latest_prediction = "rest"
latest_intensity = 0.1

# def process_emg_data(streamer, model_processor, buffer, intensity_processor, output_queue):
#     """Process EMG data continuously and put results in the queue"""
#     print("EMG processing thread started")
#     counter = 0
    
#     try:
#         while True:
#             for chunk in streamer.stream_processed():
#                 # Process for prediction and intensity
#                 windows = buffer.add_chunk(chunk)
#                 intensity_value = None
                
#                 for w in windows:
#                     prediction = model_processor.process(w)
#                     i_metrics = intensity_processor.process(w)
                    
#                     if i_metrics['rms_values'] is not None and len(i_metrics['rms_values']) > 0:
#                         norm_rms = np.array(i_metrics['rms_values']).max() / i_metrics['max_rms_ever']
#                         intensity_value = intensity_calc(norm_rms)
                
#                 # Only when model buffer has enough data
#                 if prediction is not None:
#                     output_queue.put((prediction, intensity_value))
#                     counter += 1
#                     if counter % 100 == 0:
#                         print(f"Processed {counter} EMG chunks")
#     except Exception as e:
#         print(f"Error in EMG processing thread: {e}")
#         import traceback
#         traceback.print_exc()
 
def process_emg_data(streamer, model_processor, buffer, intensity_processor, output_queue):
    """Process EMG data continuously and put results in the queue"""
    print("EMG processing thread started")
    counter = 0
    try:
        print("Entering main processing loop")
        while True:
            try:
                print("Waiting for next data chunk...")
                for chunk in streamer.stream_processed():
                    print(f"Processing chunk {counter+1}")
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

            except Exception as inner_e:
                print(f"Error in streaming loop: {inner_e}")
                import traceback
                traceback.print_exc()
                # Consider adding a small delay before retrying
                time.sleep(1)
    except Exception as e:
        print(f"Fatal error in EMG processing thread: {e}")
        import traceback
        traceback.print_exc()

def intensity_calc(norm_rms, min_speed=0, max_speed=10):
    """Calculate intensity value from normalized RMS"""
    return min_speed + (norm_rms * (max_speed - min_speed))

def initialize_emg_processing(bitalino = True):
    """Initialize EMG processing in a separate process"""
    global emg_process, EMG_MODULES_AVAILABLE
    
    if not EMG_MODULES_AVAILABLE:
        print("EMG modules not available, skipping initialization")
        return False
    
    try:
        if bitalino and BITALINO_AVAILABLE:
            # Bitalino is available, use it
            running_time = 10000
            batteryThreshold = 30
            macAddress = "/dev/tty.BITalino-3C-C2"

            # Setup of the Device and Streamer
            device = BITalino(macAddress)
            device.battery(batteryThreshold)
            streamer = BitaStreamer(device)  # BitaStreamer
            print('Bitalino Loaded')
        else:
            # Either bitalino is not requested or not available
            # Fall back to test data
            if bitalino and not BITALINO_AVAILABLE:
                print("Bitalino package not available. Falling back to test data.")
                
            # Find data file for testing
            data_path = glob.glob('./data/combined/*')
            if not data_path:
                print("No EMG data files found")
                return False
                
            print(f'Data loaded is {data_path[0]}')
            streamer = TXTStreamer(data_path[0])

        
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
            return False
            
        with open(model_path[0], 'rb') as file:
            model, label_encoder = pickle.load(file)
        
        # Setup model processor
        model_processor = WideModelProcessor(
            model=model,
            window_size=250,
            overlap=0.5,
            sampling_rate=1000,
            n_predictions=5,
            label_encoder = label_encoder
        )
        
        # Setup buffer and intensity processor
        buffer = SignalBuffer(window_size=250, overlap=0.5)
        intensity_processor = IntensityProcessor(scaling_factor=1.5)
        
        # Start EMG processing in a separate process
        emg_process = multiprocessing.Process(
            target=process_emg_data,
            args=(streamer, model_processor, buffer, intensity_processor, emg_queue)
        )
        emg_process.daemon = True  # Process will exit when main program exits
        emg_process.start()
        
        print("EMG processing initialized successfully")
        return True
        
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