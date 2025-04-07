#%%
from stream_processor_bit import *
from processors import *
from revolution_api.bitalino import *
import numpy as np
import pygame
import multiprocessing
from post_processing import *
import glob

# #%%
model_path = "xgboost_rest.pkl"
models = glob.glob(model_path)

print(models)

#%%
### Bitalino Constants ###
running_time = 60
batteryThreshold = 30
acqChannels = [0, 1, 2, 3, 4, 5]
samplingRate = 1000
nSamples = 10

macAddress = "/dev/tty.BITalino-3C-C2"

### Setup of the Device and Streamer ###
# device = BITalino(macAddress)

# device.battery(batteryThreshold)

# streamer = BitaStreamer(device) # BitaStreamer 

######################################################## TXT STREAMER ######################################################################
import glob 
data_path = glob.glob('./data/zona*')

print(f'Data loaded is {data_path[0]}')
streamer = TXTStreamer(data_path[0])

######################################################## END TXT STREAMER ######################################################################
#%%
pipeline = EMGPipeline()
pipeline.add_processor(ZeroChannelRemover())
# pipeline.add_processor(FiveChannels())   # Only use this if model is trained on 5 channels
pipeline.add_processor(NotchFilter([60],sampling_rate = 1000))
pipeline.add_processor(DCRemover())
# bandpass = ButterFilter(cutoff=[20, 450], sampling_rate=1000, filter_type='bandpass', order=4)
# pipeline.add_processor(bandpass)
pipeline.add_processor(MaxNormalizer())


streamer.add_pipeline(pipeline)



# model_path =  '/Users/adampochobut/Desktop/Res/revolution-python-api/rfc.pkl' # Adam's path

#%%


#Changed 1 to 0
# import pickle
# with open(models[0], 'rb') as file:
#     model_path = pickle.load(file)

# model_processor = ModelProcessor(
#     model=model_path,
#     window_size=250,  # 250ms window
#     overlap=0.5,      # 50% overlap
#     sampling_rate=1000,
#     n_predictions=5
# )

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

model_processor = LGBMProcessor(
            models=models,
            window_size=250,
            overlap=0.5,
            sampling_rate=1000,
            n_predictions=5,
            # label_encoder=label_encoder
        )

#%%
# Initialize buffer and intensity processor
buffer = SignalBuffer(window_size=250, overlap=0.5)
intensity_processor = IntensityProcessor(scaling_factor=1.5)

# Function to calculate intensity value from normalized RMS
def intensity_calc(norm_rms, min_speed=0.1, max_speed=1.0):
    return min_speed + (norm_rms * (max_speed - min_speed))

### Defines a process that outputs the prediction and puts it in the queue ###
def output_predictions(model_processor, chunk_queue):
    counter = 0
    while True:
        for chunk in streamer.stream_processed():
            # Process for prediction
            # prediction = model_processor.process(chunk)
            
            # Process for intensity
            windows = buffer.add_chunk(chunk)
            intensity_value = None
            prediction = None
            
            for w in windows:
                prediction = model_processor.process(w) #processes and outputs predictions
                i_metrics = intensity_processor.process(w) # outputs dict with other values
                norm_rms = np.array(i_metrics['rms_values']).max()/i_metrics['max_rms_ever']
                intensity_value = intensity_calc(norm_rms)
            
            # Only when model buffer has enough data
            if prediction is not None:
                print(f"Prediction: {prediction}")
                # Send both prediction and intensity value to the main process
                chunk_queue.put((prediction, intensity_value))
                print(counter)
                counter += 1

# Queue to pass chunks between threads
chunk_queue = multiprocessing.Queue()

# Main setup function
def main():

    prediction_thread = multiprocessing.Process(
        target=output_predictions,
        args=(model_processor, chunk_queue)
    )

    prediction_thread.start()
    
    latest_prediction = "none"
    latest_intensity = 0.1  # Default intensity value
    
    # Smoothing parameters for intensity
    target_intensity = 0.1  # Target intensity (where we want to go)
    intensity_smoothing = 0.15  # How fast we reach the target (0.0-1.0)

    ### PYGAME SETUP ###
    pygame.init()

    # Screen dimensions
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("10x10 Grid Carousel Menu with Smooth Warping")

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GRAY = (200, 200, 200)

    # Font
    font = pygame.font.Font(None, 36)

    # Grid settings
    grid_size = 30  # 10x10 grid
    cell_size = 100  # Size of each cell in pixels

    # Selector position
    selector_x = 0  # Column index (0 to 9)
    selector_y = 0  # Row index (0 to 9)

    # Camera offset (to center the selected tile)
    camera_offset_x = 0
    camera_offset_y = 0
    target_camera_offset_x = 0
    target_camera_offset_y = 0

    # Smoothing factor (0.0 to 1.0, higher is smoother)
    smoothing_factor = 0.1

    # Scroll speed (pixels per frame)
    base_scroll_speed = 0.1  # Base speed before intensity is applied

    # Main loop
    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(BLACK)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Wrap the selector position smoothly
        if selector_x < 0:
            selector_x += grid_size
            camera_offset_x += grid_size * cell_size  # Adjust camera offset to avoid jump
        elif selector_x >= grid_size:
            selector_x -= grid_size
            camera_offset_x -= grid_size * cell_size  # Adjust camera offset to avoid jump
        if selector_y < 0:
            selector_y += grid_size
            camera_offset_y += grid_size * cell_size  # Adjust camera offset to avoid jump
        elif selector_y >= grid_size:
            selector_y -= grid_size
            camera_offset_y -= grid_size * cell_size  # Adjust camera offset to avoid jump

        # Update target camera offset to center the selected tile
        target_camera_offset_x = (WIDTH // 2) - (selector_x * cell_size + cell_size // 2)
        target_camera_offset_y = (HEIGHT // 2) - (selector_y * cell_size + cell_size // 2)

        # Smoothly interpolate camera offsets toward the target
        camera_offset_x += (target_camera_offset_x - camera_offset_x) * smoothing_factor
        camera_offset_y += (target_camera_offset_y - camera_offset_y) * smoothing_factor

        # Draw the grid with additional copies to create a seamless effect
        for dx in range(-1, 2):  # Render 3 copies horizontally (-1, 0, 1)
            for dy in range(-1, 2):  # Render 3 copies vertically (-1, 0, 1)
                for row in range(grid_size):
                    for col in range(grid_size):
                        # Calculate the position of the cell (adjusted for camera offset and grid copies)
                        x = (col + dx * grid_size) * cell_size + camera_offset_x
                        y = (row + dy * grid_size) * cell_size + camera_offset_y

                        # Draw the cell if it is within the screen bounds
                        if -cell_size < x < WIDTH + cell_size and -cell_size < y < HEIGHT + cell_size:
                            # Draw the cell
                            color = GRAY
                            if (row + dy * grid_size) % grid_size == int(selector_y) and (col + dx * grid_size) % grid_size == int(selector_x):
                                color = RED  # Highlight the selected cell
                            pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))

                            # Draw the cell label (optional)
                            label = f"{row * grid_size + col + 1}"  # Label as 1-100
                            text_surface = font.render(label, True, BLACK)
                            text_rect = text_surface.get_rect(center=(x + cell_size // 2, y + cell_size // 2))
                            screen.blit(text_surface, text_rect)

        # Check for new prediction and intensity data
        if not chunk_queue.empty():
            prediction_data = chunk_queue.get_nowait()
            latest_prediction = str(prediction_data[0])
            
            # Update target intensity if available - this is what we're smoothing toward
            if prediction_data[1] is not None:
                target_intensity = prediction_data[1]

        # Smooth the intensity - gradually move current value toward target
        latest_intensity += (target_intensity - latest_intensity) * intensity_smoothing
        
        # Calculate scroll speed based on smoothed intensity
        scroll_speed = base_scroll_speed * latest_intensity
        
        # Display prediction and intensity information
        prediction_text = f"Prediction: {latest_prediction} | Intensity: {latest_intensity:.2f}"
        text_surface = font.render(prediction_text, True, (255, 255, 0))  # Yellow text
        text_rect = text_surface.get_rect(midtop=(WIDTH // 2, 10))
        screen.blit(text_surface, text_rect)

        # Apply movement based on prediction and smoothed intensity
        # if latest_prediction == "outward":
        #     selector_x -= scroll_speed  # Move left
        # if latest_prediction == "inward":
        #     selector_x += scroll_speed  # Move right
        # if latest_prediction == "upward":
        #     selector_y -= scroll_speed  # Move up
        # if latest_prediction == "0":
        #     selector_y += scroll_speed
        # if latest_prediction == "1":
        #     selector_y += 0
        # if latest_prediction == "2":
        #     selector_y -= scroll_speed 


        if latest_prediction == "right":
            selector_x -= scroll_speed  # Move left
        if latest_prediction == "left":
            selector_x += scroll_speed  # Move right
        if latest_prediction == "upward":
            selector_y -= scroll_speed  # Move up
        if latest_prediction == "downward":
            selector_y += scroll_speed
        if latest_prediction == "1":
            selector_y += 0
        if latest_prediction == "2":
            selector_y -= scroll_speed 

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Quit Pygame when the loop ends
    pygame.quit()


# Entry point for the program
if __name__ == "__main__":
    main()