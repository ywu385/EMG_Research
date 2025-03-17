from stream_processor_bit import *
from processors import *
from revolution_api.bitalino import *
import numpy as np
import pygame
import multiprocessing
import configparser
import argparse
import ast


def parse_arguments():
    """Parse command line arguments to get the INI file path."""
    parser = argparse.ArgumentParser(description='BITalino EMG Processing and Visualization')
    
    parser.add_argument('--ini_file', type=str, default='config.ini',
                        help='Path to the INI configuration file (default: config.ini)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from the specified INI file."""
    config = configparser.ConfigParser()
    
    # Try to read the specified config file
    if not config.read(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")
    
    print(f"Loaded configuration from: {config_path}")
    return config


### Defines a process that outputs the prediction and puts it in the queue ###
def output_predictions(model_processor, chunk_queue, streamer):
    counter = 0
    while True:
        for chunk in streamer.stream_processed():
            prediction = model_processor.process(chunk)
            # Add a piece here that takes extractions and displays it as a graph per time  (RMS) #TODO
            if prediction is not None:  # Only when model buffer has enough data
                print(f"Prediction: {prediction}")
                chunk_queue.put(prediction)
                print(counter)
                counter += 1


# Main setup function
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Load configuration from INI file
        config = load_config(args.ini_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please provide a valid INI file path using --ini_file argument")
        print("Using default values instead...")
        
        # Create a default configuration
        config = configparser.ConfigParser()
        
        # Set default Bitalino settings
        config['bitalino'] = {
            'running_time': '60',
            'battery_threshold': '30',
            'acq_channels': '0,1,2,3,4,5',
            'sampling_rate': '1000',
            'n_samples': '10',
            'mac_address': '/dev/tty.BITalino-3C-C2'
        }
        
        # Set default pipeline settings
        config['pipeline'] = {
            'use_five_channels': 'true',
            'notch_filter_hz': '60',
            'enable_dc_remover': 'true'
        }
        
        # Set default model settings
        config['model'] = {
            'model_path': '/Users/adampochobut/Desktop/Models/rfc.pkl',
            'window_size': '250',
            'overlap': '0.5',
            'sampling_rate': '1000'
        }
        
        # Set default pygame settings
        config['pygame'] = {
            'width': '800',
            'height': '600',
            'grid_size': '10',
            'cell_size': '100',
            'smoothing_factor': '0.1',
            'scroll_speed': '0.1'
        }
        
        print("Default configuration created.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    ### Bitalino Configuration ###
    running_time = int(config['bitalino']['running_time'])
    battery_threshold = int(config['bitalino']['battery_threshold'])
    acq_channels = [int(ch) for ch in config['bitalino']['acq_channels'].split(',')]
    sampling_rate = int(config['bitalino']['sampling_rate'])
    n_samples = int(config['bitalino']['n_samples'])
    mac_address = config['bitalino']['mac_address']

    ### Setup of the Device and Streamer ###
    device = BITalino(mac_address)
    device.battery(battery_threshold)

    streamer = BitaStreamer(device)

    ### Pipeline Configuration ###
    pipeline = EMGPipeline()
    
    if config.getboolean('pipeline', 'use_five_channels'):
        pipeline.add_processor(FiveChannels())   # Only use this if model is trained on 5 channels
    
    notch_filter_hz = [int(hz) for hz in config['pipeline']['notch_filter_hz'].split(',')]
    pipeline.add_processor(NotchFilter(notch_filter_hz, sampling_rate=sampling_rate))
    
    if config.getboolean('pipeline', 'enable_dc_remover'):
        pipeline.add_processor(DCRemover())

    if config.getboolean('pipeline', 'enable_zero_channel_remover'):
        pipeline.add_processor(ZeroChannelRemover())

    streamer.add_pipeline(pipeline)

    ### Model Configuration ###
    model_processor = ModelProcessor(
        model_path=config['model']['model_path'],
        window_size=int(config['model']['window_size']),
        overlap=float(config['model']['overlap']),
        sampling_rate=int(config['model']['sampling_rate'])
    )

    # Queue to pass chunks between threads
    chunk_queue = multiprocessing.Queue()

    # Start prediction thread
    prediction_thread = multiprocessing.Process(
        target=output_predictions,
        args=(model_processor, chunk_queue, streamer)
    )

    prediction_thread.start()
    
    latest_prediction = "none"

    ### PYGAME SETUP ###
    # Pygame must be run in the mainloop on its own.
    # It cannot be made into a process
    # All processes must be started before Pygame is initialized 
    pygame.init()

    # Screen dimensions
    WIDTH = int(config['pygame']['width'])
    HEIGHT = int(config['pygame']['height'])
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
    grid_size = int(config['pygame']['grid_size'])
    cell_size = int(config['pygame']['cell_size'])

    # Selector position
    selector_x = 0  # Column index (0 to 9)
    selector_y = 0  # Row index (0 to 9)

    # Camera offset (to center the selected tile)
    camera_offset_x = 0
    camera_offset_y = 0
    target_camera_offset_x = 0
    target_camera_offset_y = 0

    # Smoothing factor (0.0 to 1.0, higher is smoother)
    smoothing_factor = float(config['pygame']['smoothing_factor'])

    # Scroll speed (pixels per frame)
    scroll_speed = float(config['pygame']['scroll_speed'])

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
        keys = pygame.key.get_pressed()

        # Move the selector continuously if a key is held down
        if keys[pygame.K_LEFT]:
            selector_x -= scroll_speed  # Move left
        if keys[pygame.K_RIGHT]:
            selector_x += scroll_speed  # Move right
        if keys[pygame.K_UP]:
            selector_y -= scroll_speed  # Move up
        if keys[pygame.K_DOWN]:
            selector_y += scroll_speed  # Move down

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

        #If there is a new prediction, it will be put on screen
        if(not chunk_queue.empty()):
            latest_prediction = chunk_queue.get_nowait()

        prediction_text = f"Latest Prediction:{latest_prediction}"
        text_surface = font.render(prediction_text, True, (255, 255, 0))  # Yellow text
        text_rect = text_surface.get_rect(midtop=(WIDTH // 2, 10))
        screen.blit(text_surface, text_rect)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Quit Pygame when the loop ends
    pygame.quit()


# Entry point for the program
if __name__ == "__main__":
    main()