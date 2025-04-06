from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle
from collections import Counter
from post_processing import * 
# Below are dataset prep stuff
from train_helpers import *
from data_labeling import *
# pipline features
from post_processing import *
from processors import *
from stream_processor_bit import *
import multiprocessing
import traceback
import glob


try:
    from revolution_api.bitalino import *
    EMG_MODULES_AVAILABLE = True
    print("All EMG modules loaded successfully")
except ImportError as e:
    print(f"Error importing EMG modules: {e}")
    EMG_MODULES_AVAILABLE = False


emg_queue = multiprocessing.Queue(maxsize=5)


########################################################  LOAD MODEL HERE ######################################################################
model_paths = glob.glob('./working_models/LGBM.pkl')
# model_paths = glob.glob('./working_models/lgb.pkl')
if model_paths:
    model_path = model_paths[0]
    print(f"Found model: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as file:
        # model, label_encoder = pickle.load(file)
        models = pickle.load(file)
    print("Model loaded at global level")
else:
    print("No model files found")
    model = None

######################################################## Bitalino Features ######################################################################
mac_address = "/dev/tty.BITalino-3C-C2"  # Update with your device's address
if EMG_MODULES_AVAILABLE:
    try:
        # Setup device
        device = BITalino(mac_address)
        device.battery(10)
        print("BITalino connected at global level")
        
        # Setup streamer
        streamer = BitaStreamer(device)
        print("Created BITalino streamer at global level")
        
        # Setup pipeline
        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        pipeline.add_processor(NotchFilter([60], sampling_rate=1000)) 
        pipeline.add_processor(DCRemover())
        # emg_bandpass = RealTimeButterFilter(
        #                     cutoff=[20, 450],  # Target the 20-450 Hz frequency range for EMG
        #                     sampling_rate=1000,  # Assuming 1000 Hz sampling rate
        #                     filter_type='bandpass',
        #                     order=4  # 4th order provides good balance between sharpness and stability
        #                 )
        # pipeline.add_processor(emg_bandpass)
        # pipeline.add_processor(AdaptiveMaxNormalizer())
        pipeline.add_processor(MaxNormalizer())
        streamer.add_pipeline(pipeline)
        print("Pipeline added to streamer at global level")
        
        # Setup model processor
        model_processor = LGBMProcessor(
            models=models,
            window_size=250,
            overlap=0.5,
            sampling_rate=1000,
            n_predictions=5,
            # label_encoder=label_encoder
        )
        # model_processor = ModelProcessor(
        #     model = model,
        #     window_size = 250,
        #     overlap=0.5,

        # )

        
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

# Function to process EMG data and put into queue
def process_emg_data(model_processor, chunk_queue):
    counter = 0
    print("Starting to process EMG data...")
    
    while True:  # This outer loop is crucial for reconnection
        try:
            # Process EMG data continuously
            print("Connecting to stream...")
            for chunk in streamer.stream_processed():
                # Process for prediction
                windows = buffer.add_chunk(chunk)
                intensity_value = None
                prediction = None
                
                for w in windows:
                    prediction = model_processor.process_with_metadata(w)
                    i_metrics = intensity_processor.process(w)
                    print(f'Prediction from model: {prediction}')
                    if i_metrics['rms_values'] is not None and len(i_metrics['rms_values']) > 0:
                        min_speed, max_speed = 0, 10  # Define min/max speed range
                        norm_rms = np.array(i_metrics['rms_values']).max() / i_metrics['max_rms_ever']
                        intensity_value = min_speed + (norm_rms * (max_speed - min_speed))
                
                    # Only when model buffer has enough data
                    if prediction is not None:
                        # Handle full queue by making space for new data
                        if chunk_queue.full():
                            try:
                                # Remove oldest item to make space
                                chunk_queue.get_nowait()
                            except:
                                pass  # Ignore any errors
                                
                        # Add newest prediction
                        chunk_queue.put((prediction, intensity_value), block=False)
                        print(f"Prediction from processor {counter}: {prediction}, intensity={intensity_value:.2f}")
                        counter += 1
                    
        except Exception as e:
            print(f"Error processing EMG data: {e}")
            traceback.print_exc()
            print("Will attempt to reconnect in 3 seconds...")
            time.sleep(3)  # Wait before retrying

def main():
    """
    Main function to start the EMG processing and print predictions to console.
    """
    print("Starting EMG prediction system...")
    
    if not emg_initialized:
        print("EMG system not initialized. Please check your device connection.")
        return
    
    try:
        # Create a dedicated process for EMG processing
        emg_process = multiprocessing.Process(
            target=process_emg_data,
            args=(model_processor, emg_queue)
        )
        emg_process.daemon = True  # Process will exit when main program exits
        emg_process.start()
        
        print("EMG processing started in background. Predictions will display below:")
        print("-" * 50)
        
        # In the main thread, continuously get and print predictions
        while True:
            try:
                # Get prediction from queue with timeout
                prediction, intensity = emg_queue.get(timeout=1)
                
                # Print formatted prediction information
                print("\nPrediction Details:")
                print(f"  → Prediction: {prediction}")
                print(f"  → Confidence: {prediction['confidence']:.2f}")
                print(f"  → Intensity: {intensity:.2f}")
                
                # If there are probabilities for each class, print them
                if 'probabilities' in prediction:
                    print("  → Class Probabilities:")
                    for class_name, prob in prediction['probabilities'].items():
                        print(f"     - {class_name}: {prob:.4f}")
                
                print("-" * 50)
                
            except multiprocessing.queues.Empty:
                # Queue is empty, just continue
                pass
            except KeyboardInterrupt:
                print("\nStopping EMG prediction system...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(1)  # Prevent high CPU usage in case of repeated errors
    
    finally:
        # Clean up resources
        if 'emg_process' in locals() and emg_process.is_alive():
            emg_process.terminate()
            emg_process.join(timeout=1)
        
        if EMG_MODULES_AVAILABLE:
            try:
                streamer.stop()
                device.close()
                print("BITalino device disconnected")
            except:
                pass
        
        print("EMG prediction system stopped")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()

