#%%
import pygame
import multiprocessing
import time
import traceback
import numpy as np
import pickle
import glob
import atexit
import sys
import random
from stream_processor_bit import *
from processors import *
from post_processing import *

# Import game classes
from target_game import TargetGame
from gamemanager2 import GameManager
from spriralgame import GridSpiralChallenge

#%%
# Importing Args
import argparse

def parse_arguments():
    """Parse command line arguments for model selection and intensity scaling"""
    parser = argparse.ArgumentParser(description='EMG Processing with model selection')
    
    # Add model selection group
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model1', action='store_true', 
                        help='Use the first model (LGBM.pkl)')
    model_group.add_argument('--model2', action='store_true', 
                        help='Use the second model (lgb.pkl)')
    model_group.add_argument('--model3', action='store_true', 
                        help='Use the third model (lgb.pkl)')
    
    # Add intensity scaling
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scaling factor for movement intensity (default: 1.0)')
    
    # Add optional max intensity cap
    parser.add_argument('--max-intensity', type=float, default=3.0,
                        help='Maximum intensity value (default: 3.0)')
    
    return parser.parse_args()

# grabbing arguments
args = parse_arguments()

########################################################  Adding speed scaling ######################################################################

manual_intensity_scale = args.scale
max_allowed_intensity = args.max_intensity
print(f"Using intensity scaling factor: {manual_intensity_scale}")
print(f"Maximum intensity capped at: {max_allowed_intensity}")
#%%
# BITA = True
# Flag for model switching
if args.model1: 
    model_path = './working_models/LGBM_simple.pkl'
    print('Base Model loaded as {model_path}')
elif args.model2:
    model_path = './working_models/LGBM.pkl'
    print('Experimental Model Loaded {model_path}') 
elif args.model3:
    model_path ='./working_models/LGBM_model3.pkl'
    print('Experimental (zona) Model Loaded {model_path}')  

# Import your custom EMG modules
try:    
    from revolution_api.bitalino import *
    EMG_MODULES_AVAILABLE = True
    print("All EMG modules loaded successfully")
except ImportError as e:
    print(f"Error importing EMG modules: {e}")
    EMG_MODULES_AVAILABLE = False

# Define queue at the top BEFORE it's used elsewhere
# Small queue for real-time communication - only keeps most recent predictions
emg_queue = multiprocessing.Queue(maxsize=5)

# Global variables and initialization
print("Initializing EMG components at global level...")
#%%
# Find the model path
# model_paths = glob.glob('./working_models/LGBM.pkl')
# model_paths = glob.glob('./working_models/lgb.pkl')
if model_path:
    # model_path = model_path[0]
    # print(f"Found model: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as file:
        # model, label_encoder = pickle.load(file)
        models = pickle.load(file)
    print("Model loaded at global level")
else:
    print("No model files found")
    model = None
#%%
# BITalino MAC address
mac_address = "/dev/tty.BITalino-3C-C2"  # Update with your device's address


# Initialize device and streamer
if EMG_MODULES_AVAILABLE:
    try:
        # if BITA:
        # Setup device
        device = BITalino(mac_address)
        device.battery(10)
        print("BITalino connected at global level")
        streamer = BitaStreamer(device)

        # Setup streamer
        
        print("Created BITalino streamer at global level")
        # else:
        # import glob
        # files = glob.glob('./data/zona*')
        # streamer = TXTStreamer(filepath = files[0])
        
        # Setup pipeline
        # pipeline = EMGPipeline()
        # pipeline.add_processor(ZeroChannelRemover())
        # pipeline.add_processor(NotchFilter([60], sampling_rate=1000))
        # pipeline.add_processor(DCRemover())
        # # bandpass = ButterFilter(cutoff=[20, 450], sampling_rate=1000, filter_type='bandpass', order=4)
        # # pipeline.add_processor(bandpass)
        # pipeline.add_processor(AdaptiveMaxNormalizer())
        # streamer.add_pipeline(pipeline)
        pipeline = EMGPipeline()
        pipeline.add_processor(ZeroChannelRemover())
        pipeline.add_processor(NotchFilter([60], sampling_rate=1000)) 
        pipeline.add_processor(DCRemover())
        emg_bandpass = RealTimeButterFilter(
                            cutoff=[20, 450],  # Target the 20-450 Hz frequency range for EMG
                            sampling_rate=1000,  # Assuming 1000 Hz sampling rate
                            filter_type='bandpass',
                            order=4  # 4th order provides good balance between sharpness and stability
                        )
        pipeline.add_processor(emg_bandpass)
        # pipeline.add_processor(AdaptiveMaxNormalizer())
        # pipeline.add_processor(MaxNormalizer())
        streamer.add_pipeline(pipeline)
        print("Pipeline added to streamer at global level")
        
        # Setup model processor
        if args.model2 or args.model3:
            model_processor = LGBMProcessor(
                models=models,
                window_size=250,
                overlap=0.5,
                sampling_rate=1000,
                n_predictions=5,
                wavelets  = ['sym5']
                # label_encoder=label_encoder
            )
        elif args.model1:
            model_processor = LGBMProcessor(
                models=models,
                window_size=250,
                overlap=0.5,
                sampling_rate=1000,
                n_predictions=5,
                # wavelets  = ['sym5']
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
    
     # Add this rate limiting variable 
    current_intensity = 0
    # How quickly intensity can increase (smaller = more gradual)
    ramp_factor = 0.1
    # Maximum intensity allowed (cap)
    max_intensity_limit = 3.5  # Adjust this value to control the ramp speed (lower = more gradual)

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
                    prediction = model_processor.process(w)
                    i_metrics = intensity_processor.process(w)
                    print(f'Model output prediction: {prediction}')
                    
                    metric_att = 'smoothed_rms'
                    metric_att = 'rms_values'
                    if i_metrics[metric_att] is not None and len(i_metrics[metric_att]) > 0:
                        min_speed, max_speed = 0, 1  # Define min/max speed range
                        # norm_rms = np.array(i_metrics['rms_values']).max() / i_metrics['max_rms_ever']
                        norm_rms = i_metrics['overall_normalized_rms']
                        intensity_value = min_speed + (norm_rms * (max_speed - min_speed))
                        intensity_value = intensity_value * manual_intensity_scale

                        intensity_value = min(intensity_value, max_allowed_intensity)
        ########################################################  New Intensity limit ######################################################################
                    # if i_metrics[metric_att] is not None and len(i_metrics[metric_att]) > 0:
                    #     min_speed, max_speed = 0, 3  # Define min/max speed range
                    #     norm_rms = i_metrics['overall_normalized_rms']
                        
                    #     # Calculate raw target intensity
                    #     raw_intensity = min_speed + (norm_rms * (max_speed - min_speed))
                        
                    #     # Apply ramp factor for gradual increase
                    #     # Move only part of the way toward target
                    #     current_intensity += (raw_intensity - current_intensity) * ramp_factor
                        
                    #     # Apply the cap (maximum limit)
                    #     intensity_value = min(current_intensity, max_intensity_limit)
                ########################################################  New Intensity Limit (Above) ######################################################################
                    # Only when model buffer has enough data
                    if prediction is not None:
                        # Handle full queue by making space for new data
                        # Print raw prediction probabilities if available
                        if hasattr(model_processor, 'latest_probabilities') and model_processor.latest_probabilities is not None:
                            print(f"Raw probabilities: {model_processor.latest_probabilities}")
                        # Print the actual prediction
                        print(f"Final prediction: {prediction}")


                        if chunk_queue.full():
                            try:
                                # Remove oldest item to make space
                                chunk_queue.get_nowait()
                            except:
                                pass  # Ignore any errors
                                
                        # Add newest prediction
                        chunk_queue.put((prediction, intensity_value), block=False)
                        print(f"Prediction {counter}: {prediction}, intensity={intensity_value:.2f}")
                        counter += 1
                    
        except Exception as e:
            print(f"Error processing EMG data: {e}")
            traceback.print_exc()
            print("Will attempt to reconnect in 3 seconds...")
            time.sleep(3)  # Wait before retrying

# Process for running EMG
emg_process = None

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

# Helper function to safely clear the queue
def clear_queue():
    """Clear all items from the queue"""
    count = 0
    while not emg_queue.empty():
        try:
            emg_queue.get_nowait()
            count += 1
        except:
            break
    if count > 0:
        print(f"Cleared {count} items from the queue")

# Main function to run the game
def main():
    # Initialize pygame
    pygame.init()
    
    # Create game manager
    screen_width, screen_height = 800, 600
    game_manager = GameManager(screen_width, screen_height)
    
    # Start EMG processing using your existing code
    global emg_process
    
    if emg_initialized:
        print("Starting EMG processing")
        # Clear the queue before starting
        clear_queue()
        
        # Start the EMG process using your existing code
        emg_process = multiprocessing.Process(
            target=process_emg_data,
            args=(model_processor, emg_queue)
        )
        emg_process.start()
    else:
        print("EMG components not initialized. Using keyboard controls only.")
    
    # Mapping of EMG predictions to game directions
    # Customize based on your model's output classes
    prediction_mapping = {
        'upward': 'up',
        'downward': 'down',
        'inward': 'left',
        'outward': 'right',
        'rest':'rest',
        'Upward':'up',
        'Downward':'down',
        'Left':'right',
        'Right':'left',
        # 'left':'right',
        # 'right':'left'
        # Add mappings for your specific model outputs
    }
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Check for new EMG data from your global queue
        if not emg_queue.empty():
            try:
                # Get prediction and intensity from the queue
                prediction, intensity = emg_queue.get_nowait()
                
                # Map the prediction to a game direction if needed
                game_direction = prediction_mapping.get(prediction, prediction)
                
                # Pass data directly to the game manager
                game_manager.latest_prediction = game_direction
                game_manager.latest_intensity = intensity
                
                print(f"Game using: {game_direction}, intensity={intensity:.2f}")
            except Exception as e:
                print(f"Error processing EMG data in game: {e}")
        
        # Update and render the game
        game_manager.handle_input()
        game_manager.update()
        game_manager.render()
        
        # Cap the frame rate
        clock.tick(60)
    
    # Clean up when exiting
    pygame.quit()
    shutdown_emg()
    sys.exit()

if __name__ == '__main__':
    main()