import numpy as np
import multiprocessing
import time

# Global variables for processing
emg_queue = multiprocessing.Queue()
emg_process = None
latest_prediction = "rest"
latest_intensity = 0.1

def intensity_calc(norm_rms, min_speed=0, max_speed=10):
    """Calculate intensity value from normalized RMS"""
    return min_speed + (norm_rms * (max_speed - min_speed))

def output_predictions(streamer, model_processor, buffer, intensity_processor, output_queue):
    """Process EMG data continuously and put results in the queue"""
    print("EMG processing thread started")
    counter = 0
    
    try:
        while True:
            for chunk in streamer.stream_processed():
                # Process for prediction and intensity
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
                    output_queue.put((prediction, intensity_value))
                    counter += 1
                    if counter % 100 == 0:
                        print(f"Processed {counter} EMG chunks")
                        
    except Exception as e:
        print(f"Error in EMG processing thread: {e}")
        import traceback
        traceback.print_exc()

def start_processing(streamer, model_processor, buffer, intensity_processor):
    """Start the EMG processing in a separate process"""
    global emg_process
    
    # If there's already a process running, terminate it
    if emg_process is not None and emg_process.is_alive():
        shutdown_emg_processing()
    
    # Start EMG processing in a separate process
    emg_process = multiprocessing.Process(
        target=output_predictions,
        args=(streamer, model_processor, buffer, intensity_processor, emg_queue)
    )
    emg_process.daemon = True
    emg_process.start()
    print("EMG processing thread started")
    
    # Short delay to ensure process has started
    time.sleep(1)
    
    if not emg_process.is_alive():
        print("EMG process failed to start")
        return False
    
    print("EMG processing initialized successfully")
    return True

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
    global emg_process
    
    if emg_process is not None and emg_process.is_alive():
        print("Shutting down EMG processing thread...")
        emg_process.terminate()
        emg_process.join(timeout=1.0)
        print("EMG processing thread terminated")