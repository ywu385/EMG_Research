#%%
from stream_processor_bit import *
from processors import *
# from bitalino import *
import numpy as np
import pygame
import multiprocessing
from post_processing import *


#%%

### Bitalino Constants ###
# running_time = 60
# batteryThreshold = 30
# acqChannels = [0, 1, 2, 3, 4, 5]
# samplingRate = 1000
# nSamples = 10

# macAddress = "/dev/tty.BITalino-3C-C2"

# ### Setup of the Device and Streamer ###
# device = BITalino(macAddress)
# device.battery(batteryThreshold)

# streamer = BitaStreamer(device) # BitaStreamer 

#%%
######################################################## TXT STREAMER ######################################################################
import glob 
data_path = glob.glob('../data/newest data/*')

streamer = TXTStreamer(data_path[0])

######################################################## END TXT STREAMER ######################################################################

pipeline = EMGPipeline()
pipeline.add_processor(FiveChannels())   # Only use this if model is trained on 5 channels
pipeline.add_processor(NotchFilter([60],sampling_rate = 1000))
pipeline.add_processor(DCRemover())

streamer.add_pipeline(pipeline)

# model_path =  '/Users/adampochobut/Desktop/Res/revolution-python-api/rfc.pkl' # Adam's path

# #%%
model_path = 'models/*'
models = glob.glob(model_path)

import pickle
with open(models[0], 'rb') as file:
    model_path = pickle.load(file)

#%%
model_processor = ModelProcessor(
    model= model_path,
    window_size=250,  # 250ms window
    overlap=0.5,      # 50% overlap
    sampling_rate=1000
)
#%%
######################################################## JOHNs Additions ######################################################################

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
            prediction = model_processor.process(chunk)
            
            # Process for intensity
            windows = buffer.add_chunk(chunk)
            intensity_value = None
            
            for w in windows:
                intensity_metrics = intensity_processor.process(w)
                norm_rms = np.array(intensity_metrics['rms_values']).max()/intensity_processor.max_rms
                intensity_value = intensity_calc(norm_rms)
                print(f"Intensity value: {intensity_value}")
            
            # Only when model buffer has enough data
            if prediction is not None:
                print(f"Prediction: {prediction}")
                # Send both prediction and intensity value to the main process
                chunk_queue.put((prediction, intensity_value))
                print(counter)
                counter += 1

# def output_predictions(model_processor, chunk_queue):
#     counter = 0
#     while True:
#         for chunk in streamer.stream_processed():
#             prediction = model_processor.process(chunk)
#             # Add a piece here that takes extractions and displays it as a graph per time  (RMS) #TODO
#             # RMS4 for normalized RMS
#             if prediction is not None:  # Only when model buffer has enough data
#                 print(f"Prediction: {prediction}")
#                 chunk_queue.put(prediction)
#                 print(counter)
#                 counter += 1

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

    ### PYGAME SETUP ###
    #Pygame must be run in the mainloop on its own.
    #It cannot be made into a process
    #All processes must be started before Pygame is initialized 
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
    grid_size = 10  # 10x10 grid
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
    scroll_speed = 0.1  # Slower speed for smoother scrolling

    # Main loop
    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(BLACK)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # We can later replace this part so that instead of getting keys it just gets the predicition
        # We can multiply scroll_speed by the magnituted of force of the EMG signals
        # keys = pygame.key.get_pressed()

        # # Move the selector continuously if a key is held down
        # if keys[pygame.K_LEFT]:
        #     selector_x -= scroll_speed  # Move left
        # if keys[pygame.K_RIGHT]:
        #     selector_x += scroll_speed  # Move right
        # if keys[pygame.K_UP]:
        #     selector_y -= scroll_speed  # Move up
        # if keys[pygame.K_DOWN]:
        #     selector_y += scroll_speed  # Move down

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

        ######################################################## ADDED by JOHN ######################################################################
        #If there is a new prediction, it will be put on screen
        if(not chunk_queue.empty()):
            prediction_data = chunk_queue.get_nowait()
            latest_prediction = prediction_data[0]
            # Update intensity if available
            if prediction_data[1] is not None:
                latest_intensity = prediction_data[1]
        

        prediction_text = f"Latest Prediction: {latest_prediction} | Intensity: {latest_intensity:.2f}"
        text_surface = font.render(prediction_text, True, (255, 255, 0))  # Yellow text
        text_rect = text_surface.get_rect(midtop=(WIDTH // 2, 10))
        screen.blit(text_surface, text_rect)

        base_scroll_speed = 1
        # Apply intensity to scroll speed
        scroll_speed = base_scroll_speed * latest_intensity
        ######################################################## Prev implementation ######################################################################
        #If there is a new prediction, it will be put on screen
        # if(not chunk_queue.empty()):
        #     latest_prediction = chunk_queue.get_nowait()

        # prediction_text = f"Latest Prediction:{latest_prediction}"
        # text_surface = font.render(prediction_text, True, (255, 255, 0))  # Yellow text
        # text_rect = text_surface.get_rect(midtop=(WIDTH // 2, 10))
        # screen.blit(text_surface, text_rect)

        ######################################################## Adding intensity######################################################################
        if latest_prediction=="outward":
            selector_x -= scroll_speed  # Move left
        if latest_prediction=="inward":
            selector_x += scroll_speed  # Move right
        if latest_prediction=="upward":
            selector_y -= scroll_speed  # Move up
        if latest_prediction=="downward":
            selector_y += scroll_speed  # Move down

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Quit Pygame when the loop ends
    pygame.quit()


# Entry point for the program
if __name__ == "__main__":
    main()