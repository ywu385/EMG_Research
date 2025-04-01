#%%
import multiprocessing
import time
import traceback
import numpy as np
import pickle
import glob
import atexit

# Import your custom modules
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

# Global variables and initialization
print("Initializing EMG components at global level...")
#%%



# Find the model path
model_paths = glob.glob('./models/*.pkl')
if model_paths:
    model_path = model_paths[0]
    print(f"Found model: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as file:
        model, label_encoder = pickle.load(file)
    print("Model loaded at global level")
else:
    print("No model files found")
    model = None

# BITalino MAC address
mac_address = "/dev/tty.BITalino-3C-C2"  # Update with your device's address

# Initialize device and streamer
if EMG_MODULES_AVAILABLE:
    try:
        # Setup device
        device = BITalino(mac_address)
        device.battery(30)
        print("BITalino connected at global level")
        
        # Setup streamer
        streamer = BitaStreamer(device)
        print("Created BITalino streamer at global level")
        
        # Setup pipeline
        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
        pipeline.add_processor(DCRemover())
        streamer.add_pipeline(pipeline)
        print("Pipeline added to streamer at global level")
        
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
        
        print("All EMG components initialized at global level")
        
        # Flag to indicate if EMG is initialized
        emg_initialized = True
    except Exception as e:
        print(f"Error initializing EMG components: {e}")
        traceback.print_exc()
        emg_initialized = False
else:
    emg_initialized = False

# # Function to calculate intensity value from normalized RMS
# def intensity_calc(norm_rms, min_speed=0.1, max_speed=1.0):
#     return min_speed + (norm_rms * (max_speed - min_speed))

# ### Defines a process that outputs the prediction and puts it in the queue ###
# def output_predictions(model_processor, chunk_queue):
#     counter = 0
#     while True:
#         for chunk in streamer.stream_processed():
#             # Process for prediction
#             # prediction = model_processor.process(chunk)
            
#             # Process for intensity
#             windows = buffer.add_chunk(chunk)
#             intensity_value = None
#             prediction = None
            
#             for w in windows:
#                 prediction = model_processor.process(w) #processes and outputs predictions
#                 i_metrics = intensity_processor.process(w) # outputs dict with other values
#                 norm_rms = np.array(i_metrics['rms_values']).max()/i_metrics['max_rms_ever']
#                 intensity_value = intensity_calc(norm_rms)
            
#             # Only when model buffer has enough data
#             if prediction is not None:
#                 print(f"Prediction: {prediction}")
#                 # Send both prediction and intensity value to the main process
#                 chunk_queue.put((prediction, intensity_value))
#                 print(counter)
#                 counter += 1

# Small queue for real-time communication - only keeps most recent predictions
emg_queue = multiprocessing.Queue(maxsize=4)

# Function to process EMG data and put into queue
def process_emg_data(model_processor, chunk_queue):
    # Using global components from main process
    # global streamer, buffer, intensity_processor
    
    counter = 0
    print("Starting to process EMG data...")
    
    while True:  # Add this outer loop
        try:
            # Process EMG data continuously
            print("Connecting to stream...")
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
                    chunk_queue.put((prediction, intensity_value))
                    print(f"Prediction {counter}: {prediction}, intensity={intensity_value:.2f}")
                    counter += 1
                    
        except Exception as e:
            print(f"Error processing EMG data: {e}")
            traceback.print_exc()
            print("Will attempt to reconnect in 3 seconds...")
            time.sleep(3)  # Wait before retrying

# Process for running EMG
emg_process = None

# Function to start the prediction thread
def start_predictions():
    global emg_process
    
    if not emg_initialized:
        print("EMG components not initialized. Cannot start processing.")
        return False
    
    try:
        # Start prediction thread
        emg_process = multiprocessing.Process(
            target=process_emg_data,
            args=(model_processor,emg_queue)
        )
        # emg_process.daemon = True
        emg_process.start()
        
        # Short delay to ensure process has started
        time.sleep(1)
        
        if not emg_process.is_alive():
            print("EMG process failed to start")
            return False
        
        print("EMG processing started successfully")
        return True
    
    except Exception as e:
        print(f"Error starting EMG processing: {e}")
        traceback.print_exc()
        return False

# Function to shutdown EMG processing
def shutdown_emg():
    global emg_process, device
    
    if emg_process is not None and emg_process.is_alive():
        print("Shutting down EMG processing...")
        emg_process.terminate()
        emg_process.join(timeout=1.0)
        print("EMG processing terminated")
    
    if 'device' in globals():
        try:
            device.close()
            print("BITalino device closed")
        except:
            print("Error closing BITalino device")

# Register the shutdown function
atexit.register(shutdown_emg)

# Main function
def main():
    if start_predictions():
        print("EMG predictions started. Reading from queue...")
        
        start_time = time.time()
        last_time_check = start_time
        
        try:
            # Simple loop to read and print EMG data
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Print time update every 10 seconds
                if current_time - last_time_check >= 10:
                    print(f"Time elapsed: >>>>>>>>>>>>>>>>>>>> {elapsed_time:.1f} seconds")
                    last_time_check = current_time
                
                if not emg_queue.empty():
                    prediction, intensity = emg_queue.get_nowait()
                    print(f"Received: Prediction={prediction}, Intensity={intensity:.2f}")
                
                # time.sleep(0.1)  # Small delay to prevent CPU hogging
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            total_time = time.time() - start_time
            print(f"Total runtime: {total_time:.1f} seconds")
    else:
        print("Failed to start EMG predictions")

if __name__ == "__main__":
    main()