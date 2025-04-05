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
from post_processing import BaggedRF

# Import game classes
from target_game import TargetGame
from gamemanager2 import GameManager
from spriralgame import GridSpiralChallenge
#%%
# Import your custom EMG modules
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

# Define queue at the top BEFORE it's used elsewhere
# Small queue for real-time communication - only keeps most recent predictions
emg_queue = multiprocessing.Queue(maxsize=5)

# Global variables and initialization
print("Initializing EMG components at global level...")
#%%
# Find the model path
model_paths = glob.glob('./working_models/LGBM.pkl')
# model_paths = glob.glob('./working_models/lgb.pkl')
if model_paths:
    model_path = model_paths[0]
    print(f"Found model: {model_path}")
    
    # Load model
    with open(model_path, 'rb') as file:
        # model, label_encoder = pickle.load(file)
        model = pickle.load(file)
        models = model['models']
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
        # bandpass = ButterFilter(cutoff=[20, 450], sampling_rate=1000, filter_type='bandpass', order=4)
        # pipeline.add_processor(bandpass)
        pipeline.add_processor(AdaptiveMaxNormalizer())
        streamer.add_pipeline(pipeline)
        print("Pipeline added to streamer at global level")
        
        # Setup model processor
        model_processor = LGBMProcessor(
            model=models,
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
                    prediction = model_processor.process(w)
                    i_metrics = intensity_processor.process(w)
                    
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

# Simulation function for testing without actual EMG hardware
def simulate_emg_data(chunk_queue):
    """Simulate EMG data for testing"""
    counter = 0
    print("Starting EMG simulation...")
    
    # Possible prediction classes - update to match your model's output
    prediction_classes = ["up", "down", "left", "right"]
    
    while True:
        try:
            # Generate mock prediction and intensity
            prediction = random.choice(prediction_classes)
            
            # Generate intensity that varies sinusoidally (more natural)
            intensity_value = 5.0 + 4.5 * math.sin(time.time() * 0.2)
            intensity_value = max(0.5, min(10.0, intensity_value))
            
            # Handle full queue by making space for new data
            if chunk_queue.full():
                try:
                    chunk_queue.get_nowait()
                except:
                    pass
            
            # Add newest prediction
            chunk_queue.put((prediction, intensity_value), block=False)
            print(f"Simulated {counter}: {prediction}, intensity={intensity_value:.2f}")
            counter += 1
            
            # Sleep to simulate realistic update rate
            time.sleep(0.2)
                
        except Exception as e:
            print(f"Error in EMG simulation: {e}")
            time.sleep(1)

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
        'rest':'rest'
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