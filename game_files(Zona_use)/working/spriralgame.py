import pygame
import numpy as np
import time
import math
from datetime import datetime
import os

class GridSpiralChallenge:
    """
    A grid-based spiral navigation challenge where the player navigates through a series
    of tiles arranged in a spiral pattern on the same grid as the target game.
    """
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        
        # Game state
        self.STATE_WAITING = 0   # Waiting to press space to start
        self.STATE_COUNTDOWN = 1 # Countdown before game starts
        self.STATE_PLAYING = 2   # Game is active
        self.STATE_PAUSED = 3    # Game is paused
        self.STATE_COMPLETED = 4 # Challenge completed
        
        self.state = self.STATE_WAITING
        
        # Countdown settings
        self.countdown_duration = 3  # 3 seconds countdown
        self.countdown_start_time = 0
        self.countdown_remaining = 0
        
        # Time tracking
        self.start_time = 0
        self.total_time = 0
        self.paused_time = 0  # Time when paused
        
        # Spiral path - sequence of grid coordinates to follow
        self.spiral_points = []
        self.current_point_index = 0
        self.generate_spiral_path()
        
        # Challenge settings
        self.progress = 0  # 0 to 100 percent
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)
        self.PURPLE = (128, 0, 128)
        self.ORANGE = (255, 165, 0)

        self.complete_path_history = []  
        self.current_path_segment = []   # Tracks current path segment
        self.last_position_number = None  
        

    def coords_to_grid_number(self, col, row):
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
            base_filename = os.path.join(user_dir, f"{time_id}_spiral_game_path.txt")
            
            filename = base_filename
            counter = 1
            while os.path.exists(filename):
                # File exists, create a new name with a counter
                filename = os.path.join(user_dir, f"{time_id}_sprial_game_path_{counter}.txt")
                counter += 1

            if self.state == self.STATE_PLAYING:
                current_time = time.time() - self.start_time
            else:
                current_time = self.total_time

            with open(filename, 'w') as f:
                # Write header information
                f.write(f"Spiral Game Path History\n")
                f.write(f"User ID: {user_id}\n")
                f.write(f"Session Time ID: {time_id}\n")
                f.write(f"Progress: {self.progress:.2f}%\n")
                f.write(f"Points Reached: {self.current_point_index}/{len(self.spiral_points)}\n")
                f.write(f"Total Time: {current_time:.2f} seconds\n")
                f.write(f"Path (Grid Numbers):\n")
                
                # Write the actual path history
                for i, position in enumerate(self.complete_path_history):
                    f.write(f"{i+1}: {position}\n")
                    
            print(f"Path history saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error saving path history: {e}")
            return None

    def generate_spiral_path(self, start_grid_number = 35):
        """Generate a spiral path of grid coordinates"""
        self.spiral_points = []
        
        # Start at the center of the grid
        # center_x = self.grid_size // 2
        # center_y = self.grid_size // 2
        start_col = (start_grid_number - 1) % self.grid_size
        start_row = (start_grid_number - 1) // self.grid_size
        
        # Initialize spiral path with the center point
        self.spiral_points.append((start_col, start_row))
        
        # Define directions: right, down, left, up
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Generate spiral path
        steps = 1  # Number of steps in current direction
        direction_index = 0  # Start moving right
        x, y = start_col, start_row
        
        # Continue until we reach grid boundaries
        max_steps = self.grid_size * 2
        direction_changes = 0
        
        while len(self.spiral_points) < max_steps and direction_changes < max_steps * 2:
            # Take 'steps' steps in current direction
            for _ in range(steps):
                dx, dy = directions[direction_index]
                x, y = x + dx, y + dy
                
                # Check if point is within grid boundaries
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # Avoid adding duplicates
                    if (x, y) not in self.spiral_points:
                        self.spiral_points.append((x, y))
                else:
                    # We've hit a boundary, no need to continue in this direction
                    break
            
            # Change direction
            direction_index = (direction_index + 1) % 4
            direction_changes += 1
            
            # Increase steps every 2 direction changes
            if direction_changes % 2 == 0:
                steps += 1
                
        # Ensure we have a reasonable number of points (not too many, not too few)
        if len(self.spiral_points) > 20:
            # If we have too many points, take a subset
            step = len(self.spiral_points) // 20
            self.spiral_points = self.spiral_points[::step]
            # Always include the last point
            if self.spiral_points[-1] != self.spiral_points[-1]:
                self.spiral_points.append(self.spiral_points[-1])
    
    def start_countdown(self):
        """Start the countdown before challenge begins"""
        if self.state == self.STATE_WAITING:
            self.state = self.STATE_COUNTDOWN
            self.countdown_start_time = time.time()
            self.countdown_remaining = self.countdown_duration
    
    def start_challenge(self):
        """Start the actual challenge after countdown"""
        self.state = self.STATE_PLAYING
        self.start_time = time.time()
        self.current_point_index = 0
        self.progress = 0
    
    def toggle_pause(self):
        """Pause or unpause the challenge"""
        if self.state == self.STATE_PLAYING:
            self.state = self.STATE_PAUSED
            self.paused_time = time.time()
        elif self.state == self.STATE_PAUSED:
            self.state = self.STATE_PLAYING
            # Adjust start time to account for paused duration
            paused_duration = time.time() - self.paused_time
            self.start_time += paused_duration
    
    def check_progress(self, current_position):
        """
        Check player progress along the spiral.
        Returns True if the current checkpoint is reached.
        """
        if self.state != self.STATE_PLAYING:
            return False
        
        self.update_path(current_position)

        if self.current_point_index >= len(self.spiral_points):
            return False
            
        # Get current target point
        target_point = self.spiral_points[self.current_point_index]
        
        # Check if player has reached the current point
        if current_position == target_point:
            self.current_point_index += 1
            
            # Update progress percentage
            self.progress = (self.current_point_index / len(self.spiral_points)) * 100
            
            # Check if all points have been reached
            if self.current_point_index >= len(self.spiral_points):
                self.state = self.STATE_COMPLETED
                self.total_time = time.time() - self.start_time
                return True
                
            return True
            
        return False
    
    def update(self):
        """Update challenge state"""
        current_time = time.time()
        
        if self.state == self.STATE_COUNTDOWN:
            # Update countdown
            elapsed = current_time - self.countdown_start_time
            self.countdown_remaining = max(0, self.countdown_duration - elapsed)
            
            # Check if countdown is complete
            if self.countdown_remaining <= 0:
                self.start_challenge()
    
    def get_current_target(self):
        """Get the current target point coordinates"""
        if self.current_point_index < len(self.spiral_points):
            return self.spiral_points[self.current_point_index]
        return None
    
    def get_completed_points(self):
        """Get list of completed points in the spiral"""
        return self.spiral_points[:self.current_point_index]
    
    def get_upcoming_points(self, look_ahead=3):
        """Get list of upcoming points in the spiral, limited to look_ahead count"""
        start = self.current_point_index
        end = min(start + look_ahead, len(self.spiral_points))
        return self.spiral_points[start:end]
    
    def reset(self):
        """Reset the challenge to start over"""
        self.state = self.STATE_WAITING
        self.current_point_index = 0
        self.progress = 0
        self.complete_path_history = []
        self.current_path_segment = []
        self.last_position_number = None
    
    def get_info(self):
        """Get current challenge information"""
        
        return {
            'state': self.state,
            'progress': self.progress,
            'current_point': self.current_point_index,
            'total_points': len(self.spiral_points),
            'countdown': self.countdown_remaining,
            'current_target': self.get_current_target(),
            'time': time.time() - self.start_time if self.state == self.STATE_PLAYING else self.total_time
        }