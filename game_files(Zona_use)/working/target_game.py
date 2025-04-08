import random
import time
from datetime import datetime
import os

class TargetGame:
    """Manages the target-based navigation game logic with fixed targets and time limit"""
    def __init__(self, grid_size=10, targets_for_next_level=10, time_limit=300):
        self.grid_size = grid_size
        self.score = 0
        self.targets_completed = 0
        self.current_target = None
        self.targets_for_next_level = targets_for_next_level
        self.level_completed = False
        self.time_limit = time_limit  # Time limit in seconds (5 minutes default)
        self.time_expired = False
        
        # Fixed sequence of targets (predefine these for research consistency)
        self.target_sequence = self.generate_fixed_targets()
        
        # Game state
        self.STATE_WAITING = 0   # Waiting to press space to start
        self.STATE_COUNTDOWN = 1 # Countdown before game starts
        self.STATE_PLAYING = 2   # Game is active
        self.STATE_PAUSED = 3    # Game is paused
        self.STATE_LEVEL_COMPLETE = 4  # Level completed
        self.STATE_TIME_EXPIRED = 5  # Time limit reached
        
        self.state = self.STATE_WAITING
        
        # Countdown settings
        self.countdown_duration = 3  # 3 seconds countdown
        self.countdown_start_time = 0
        self.countdown_remaining = 0
        
        # Time tracking
        self.start_time = 0
        self.last_target_time = 0
        self.current_run_time = 0
        self.paused_time = 0  # Time when paused
        self.paused_duration = 0  # Total time spent paused
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (200, 200, 200)
        self.YELLOW = (255, 255, 0)
        self.TARGET_COLOR = (0, 255, 0)  # Green for target

        # Path Tracking
        self.complete_path_history = []  # Tracks entire path throughout the game
        self.current_path_segment = []   # Tracks current path segment (resets after target)
        self.last_position_number = None        # To avoid recording duplicate positions

    ######################################################## Grid and Path functions ######################################################################
    def coords_to_grid_number(self, col, row): # Helper method for finding grid numbers
        """Convert grid coordinates to a grid number (1-100)"""
        return row * self.grid_size + col + 1

    def update_path(self, current_position):
        """Update path tracking with player's current position"""
        if self.state != self.STATE_PLAYING:
            return
        
        # Convert position coordinates to grid number
        col, row = current_position
        position_number = self.coords_to_grid_number(col, row)


        # Avoid adding duplicate consecutive positions
        if position_number != self.last_position_number:
            self.complete_path_history.append(position_number)
            self.current_path_segment.append(position_number)
            self.last_position_number = position_number
 
    def save_path_history_to_file(self, user_id="user", base_dir="game_data", time_id = None):
        """Save the complete path history to a file with timestamp and user ID"""
        try:
            # Create directory structure
            if time_id is None:
                time_id = datetime.now().strftime("%Y%m%d_%H%M%S")[-5:]

            user_id = user_id + '_' + time_id
            user_dir = os.path.join(base_dir, user_id)

            # Create directories if they don't exist
            os.makedirs(user_dir, exist_ok=True)
            
            # Create filename
            base_filename = os.path.join(user_dir, f"{time_id}_target_game_path.txt")
            
            filename = base_filename
            counter = 1
            while os.path.exists(filename):
                # File exists, create a new name with a counter
                filename = os.path.join(user_dir, f"{time_id}_target_game_path_{counter}.txt")
                counter += 1

            with open(filename, 'w') as f:
                # Write header information
                f.write(f"Target Game Path History\n")
                f.write(f"User ID: {user_id}\n")
                f.write(f"Session Time ID: {time_id}\n")
                f.write(f"Targets Completed: {self.targets_completed}\n")
                f.write(f"Total Time: {self.current_run_time:.2f} seconds\n")
                f.write(f"Path (Grid Numbers):\n")
                
                # Write the actual path history
                for i, position in enumerate(self.complete_path_history):
                    f.write(f"{i+1}: {position}\n")
                    
            print(f"Path history saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving path history: {e}")
            return None
    ######################################################## Grid and Path functions ######################################################################
    def generate_fixed_targets(self):
        """
        Generate a fixed sequence of targets within the visible area.
        This ensures consistent target locations for research purposes.
        """
        # Create a list of visible positions (assuming a view of 5x5 from center)
        visible_area_size = 5  # 5x5 grid visible at once
        
        # Calculate visible range (based on center of grid)
        center = self.grid_size // 2
        min_pos = max(0, center - visible_area_size // 2)
        max_pos = min(self.grid_size - 1, center + visible_area_size // 2)
        
        visible_positions = []
        for row in range(min_pos, max_pos + 1):
            for col in range(min_pos, max_pos + 1):
                # Skip the exact center position (starting position)
                if row == center and col == center:
                    continue
                visible_positions.append((col, row))
        
        # Ensure we have enough positions for the required targets
        if len(visible_positions) < self.targets_for_next_level:
            # If not enough positions, allow using some positions twice
            while len(visible_positions) < self.targets_for_next_level:
                visible_positions.append(random.choice(visible_positions))
        
        # Shuffle the positions to create a random but fixed sequence
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(visible_positions)
        
        # Take only the required number of targets
        targets = visible_positions[:self.targets_for_next_level]
        
        return targets
    
    def start_countdown(self):
        """Start the countdown before game begins"""
        if self.state == self.STATE_WAITING or self.state == self.STATE_LEVEL_COMPLETE or self.state == self.STATE_TIME_EXPIRED:
            self.state = self.STATE_COUNTDOWN
            self.countdown_start_time = time.time()
            self.countdown_remaining = self.countdown_duration
            # Reset flags if restarting
            self.level_completed = False
            self.time_expired = False
    
    def start_game(self):
        """Start the actual game after countdown"""
        self.state = self.STATE_PLAYING
        self.start_time = time.time()
        self.targets_completed = 0
        self.paused_duration = 0
        # Set first target
        self.get_next_target()
    
    def toggle_pause(self):
        """Pause or unpause the game"""
        if self.state == self.STATE_PLAYING:
            self.state = self.STATE_PAUSED
            self.paused_time = time.time()
        elif self.state == self.STATE_PAUSED:
            self.state = self.STATE_PLAYING
            # Calculate the time spent paused
            pause_length = time.time() - self.paused_time
            self.paused_duration += pause_length
    
    def get_next_target(self):
        """Get the next target from the sequence"""
        if self.targets_completed < len(self.target_sequence):
            self.current_target = self.target_sequence[self.targets_completed]
        else:
            # Fallback if we somehow exceed the target list
            self.current_target = None
    
    def check_target_reached(self, current_position):
        """Check if the current position has reached the target"""
        if self.state != self.STATE_PLAYING:
            return False
            
        # Update path tracking
        self.update_path(current_position)
        
        if current_position == self.current_target:
            # Target reached!
            self.targets_completed += 1  
            
            # Record completion time
            self.last_target_time = time.time() - self.start_time - self.paused_duration
            
            self.current_path_segment = []

            # Check if level completed
            if self.targets_completed >= self.targets_for_next_level:
                self.level_completed = True
                self.state = self.STATE_LEVEL_COMPLETE
                return True
            
            # Set next target
            self.get_next_target()
            
            # Reset start time for next target (but keep track of total time)
            target_completion_time = time.time()
            
            return True
        return False
    
    def check_time_limit(self):
        """Check if time limit has been reached"""
        if self.state == self.STATE_PLAYING:
            elapsed_time = time.time() - self.start_time - self.paused_duration
            if elapsed_time >= self.time_limit:
                self.time_expired = True
                self.state = self.STATE_TIME_EXPIRED
                return True
        return False
    
    def update(self):
        """Update game state"""
        current_time = time.time()
        
        if self.state == self.STATE_COUNTDOWN:
            # Update countdown
            elapsed = current_time - self.countdown_start_time
            self.countdown_remaining = max(0, self.countdown_duration - elapsed)
            
            # Check if countdown is complete
            if self.countdown_remaining <= 0:
                self.start_game()
                
        elif self.state == self.STATE_PLAYING:
            # Update game time
            self.current_run_time = current_time - self.start_time - self.paused_duration
            
            # Check time limit
            self.check_time_limit()
    
    def get_complete_path_history(self):
        """Return the complete path history for the entire game"""
        return self.complete_path_history
    
    def get_current_path_segment(self):
        """Return the current path segment (since last target)"""
        return self.current_path_segment

    def get_target_info(self):
        """Get current target information"""
        remaining_time = max(0, self.time_limit - self.current_run_time) if self.state == self.STATE_PLAYING else 0
        
        return {
            'target': self.current_target,
            'score': self.targets_completed,
            'current_time': self.current_run_time,
            'last_time': self.last_target_time,
            'state': self.state,
            'countdown': self.countdown_remaining,
            'level_completed': self.level_completed,
            'time_expired': self.time_expired,
            'targets_for_next_level': self.targets_for_next_level,
            'time_limit': self.time_limit,
            'time_remaining': remaining_time
        }
    
    def reset_game(self):
        """Reset the game to start over"""
        self.targets_completed = 0
        self.current_target = None
        self.state = self.STATE_WAITING
        self.level_completed = False
        self.time_expired = False
        self.complete_path_history = []
        self.current_path_segment = []
        self.last_position = None
        
    def reset_for_next_level(self):
        """Reset only what's needed for the next level"""
        self.targets_completed = 0
        self.current_target = None