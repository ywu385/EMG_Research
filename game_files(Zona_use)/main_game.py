import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import atexit
from game_comp.game_manager import GameManager
import emg_comp.emg_processor as emg_processor



def main():
    # Initialize the EMG processor first (at global level, before pygame)
    use_emg = True
    
    if use_emg:
        print("Initializing EMG processing...")
        emg_initialized = emg_processor.initialize_emg_processing(bitalino = True)
        
        if not emg_initialized:
            print("Warning: EMG initialization failed, falling back to keyboard mode")
            use_emg = False
        else:
            print("EMG processing running in background")
            
            # Register cleanup for EMG if it was initialized
            atexit.register(emg_processor.shutdown_emg_processing)
    
    # Create and run the game manager
    # game_manager = GameManager(1280, 720, use_emg=use_emg)
    game_manager = GameManager(800, 600, use_emg=use_emg)
    game_manager.run()

if __name__ == "__main__":
    main()

# Notes
# Option to restart each game (record all records: user1, user1game1, user1game2, ...)
# Spiral coloring: gradient coloring with brightest being away
# Implement error message when they trace outside of the box
# Can we see what is the square that is being selected?
# Can we keep track of the square numbers
# 