import pygame
import numpy as np
import random
import time
import sys
import atexit
import multiprocessing

# Import game modules
from target_game import TargetGame

class GameManager:
    """Manages the game and processes EMG input"""
    
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("EMG Navigation Game")
        
        # Setup game components
        self.grid_size = 10
        self.cell_size = 100
        
        # Create game instance
        self.target_game = TargetGame(self.grid_size, targets_for_next_level=10, time_limit=300)
        
        # Camera position (for grid navigation)
        self.camera_x = 0
        self.camera_y = 0
        
        # Speed settings
        self.base_keyboard_speed = 5      # Speed for keyboard controls
        self.base_emg_speed = 0.1         # Base speed for EMG
        
        # EMG control variables
        self.latest_prediction = "rest"
        self.latest_intensity = 0.1
        
       
        # Setup fonts
        self.font = pygame.font.Font(None, 36)
        self.large_font = pygame.font.Font(None, 96)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (200, 200, 200)
        self.ORANGE = (255, 165, 0)
        self.CYAN = (0, 255, 255)
        self.PURPLE = (128, 0, 128)
        
        # For tracking keypress events
        self.space_pressed = False
        self.p_pressed = False
        self.esc_pressed = False
        self.r_pressed = False
        
        # Register cleanup function
        atexit.register(self.cleanup)
    
    def cleanup(self):
        """Clean up resources when the game exits"""
        pygame.quit()
    
    
    def handle_input(self):
        """Process keyboard and EMG input"""
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Handle game control keys
        if keys[pygame.K_SPACE] and not self.space_pressed:
            if self.target_game.state == self.target_game.STATE_WAITING:
                self.target_game.start_countdown()
            elif self.target_game.state in [self.target_game.STATE_LEVEL_COMPLETE, self.target_game.STATE_TIME_EXPIRED]:
                self.target_game.reset_game()
                self.target_game.start_countdown()
            self.space_pressed = True
        elif not keys[pygame.K_SPACE]:
            self.space_pressed = False
        
        # P to pause/unpause
        if keys[pygame.K_p] and not self.p_pressed:
            if self.target_game.state in [self.target_game.STATE_PLAYING, self.target_game.STATE_PAUSED]:
                self.target_game.toggle_pause()
            self.p_pressed = True
        elif not keys[pygame.K_p]:
            self.p_pressed = False
        
        # R to reset
        if keys[pygame.K_r] and not self.r_pressed:
            self.target_game.reset_game()
            self.r_pressed = True
            print("Game reset by user")
        elif not keys[pygame.K_r]:
            self.r_pressed = False
        
        # ESC to exit
        if keys[pygame.K_ESCAPE] and not self.esc_pressed:
            # Quit the game
            self.cleanup()
            sys.exit()
            self.esc_pressed = True
        elif not keys[pygame.K_ESCAPE]:
            self.esc_pressed = False
        
        # Process EMG data
        self.process_emg_data()
        
        # Handle movement input
        if self.target_game.state == self.target_game.STATE_PLAYING:
            # Keyboard movement
            if keys[pygame.K_LEFT]:
                self.camera_x -= self.base_keyboard_speed
            if keys[pygame.K_RIGHT]:
                self.camera_x += self.base_keyboard_speed
            if keys[pygame.K_UP]:
                self.camera_y -= self.base_keyboard_speed
            if keys[pygame.K_DOWN]:
                self.camera_y += self.base_keyboard_speed
            
            # EMG movement
            # Map predictions to directions
            scroll_speed = self.base_emg_speed * self.latest_intensity * 10
            
            # Customize these mappings based on your EMG prediction classes
            if self.latest_prediction == "left":
                self.camera_x -= scroll_speed
            elif self.latest_prediction == "right":
                self.camera_x += scroll_speed
            elif self.latest_prediction == "up":
                self.camera_y -= scroll_speed
            elif self.latest_prediction == "down":
                self.camera_y += scroll_speed
    
    def update(self):
        """Update game state"""
        # Calculate current position on grid
        center_col = int((self.camera_x + self.screen_width // 2) // self.cell_size)
        center_row = int((self.camera_y + self.screen_height // 2) // self.cell_size)
        
        # Convert to grid coordinates
        center_grid_col = center_col % self.grid_size
        center_grid_row = center_row % self.grid_size
        center_position = (center_grid_col, center_grid_row)
        
        # Update target game
        self.target_game.update()
        
        if self.target_game.state == self.target_game.STATE_PLAYING:
            # Check if target is reached
            self.target_game.check_target_reached(center_position)
    
    def render(self):
        """Render the game"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Calculate the visible range of tiles
        start_col = int(self.camera_x // self.cell_size) - 1
        end_col = start_col + (self.screen_width // self.cell_size) + 3
        start_row = int(self.camera_y // self.cell_size) - 1
        end_row = start_row + (self.screen_height // self.cell_size) + 3

        # Calculate the central tile
        center_col = int((self.camera_x + self.screen_width // 2) // self.cell_size)
        center_row = int((self.camera_y + self.screen_height // 2) // self.cell_size)
        
        # Convert to grid position
        center_grid_col = center_col % self.grid_size
        center_grid_row = center_row % self.grid_size
        center_position = (center_grid_col, center_grid_row)
        
        # Get target game info
        target_info = self.target_game.get_target_info()
        
        # Render grid
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                # Map to grid coordinates using modulo for infinite tiling
                grid_row = row % self.grid_size
                grid_col = col % self.grid_size
                
                # Calculate screen position
                screen_x = col * self.cell_size - self.camera_x
                screen_y = row * self.cell_size - self.camera_y
                
                # Draw tile if it would be visible on screen
                if -self.cell_size < screen_x < self.screen_width and -self.cell_size < screen_y < self.screen_height:
                    # Determine cell color based on state
                    is_center = (col == center_col and row == center_row)
                    is_target = (self.target_game.state == self.target_game.STATE_PLAYING and 
                                (grid_col, grid_row) == target_info['target'])
                    
                    # Choose color based on cell state
                    if is_center:
                        color = self.RED
                    elif is_target:
                        color = self.GREEN
                    else:
                        color = self.GRAY
                    
                    # Draw the cell with appropriate color
                    pygame.draw.rect(self.screen, color, (screen_x, screen_y, self.cell_size, self.cell_size))
                    
                    # Calculate the tile's unique number (1-100) based on its grid position
                    tile_number = grid_row * self.grid_size + grid_col + 1
                    
                    # Draw the cell label
                    text_surface = self.font.render(str(tile_number), True, self.BLACK)
                    text_rect = text_surface.get_rect(center=(screen_x + self.cell_size // 2, screen_y + self.cell_size // 2))
                    self.screen.blit(text_surface, text_rect)
        
        # Display game state specific UI
        if self.target_game.state == self.target_game.STATE_WAITING:
            # Display "Press Space to Start" message
            start_text = "Press SPACE to Start"
            start_surface = self.large_font.render(start_text, True, self.YELLOW)
            start_rect = start_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(start_surface, start_rect)
            
            # Display instructions
            inst_text = "Use arrow keys to navigate to the target tiles"
            inst_surface = self.font.render(inst_text, True, self.WHITE)
            inst_rect = inst_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
            self.screen.blit(inst_surface, inst_rect)
            
        elif self.target_game.state == self.target_game.STATE_COUNTDOWN:
            # Display countdown number
            count_text = str(int(target_info['countdown']) + 1)
            count_surface = self.large_font.render(count_text, True, self.YELLOW)
            count_rect = count_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(count_surface, count_rect)
            
            # Display "Get Ready" message
            ready_text = "Get Ready!"
            ready_surface = self.font.render(ready_text, True, self.WHITE)
            ready_rect = ready_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 80))
            self.screen.blit(ready_surface, ready_rect)
            
        elif self.target_game.state == self.target_game.STATE_PAUSED:
            # Display "PAUSED" message
            pause_text = "PAUSED"
            pause_surface = self.large_font.render(pause_text, True, self.YELLOW)
            pause_rect = pause_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(pause_surface, pause_rect)
            
            # Display "Press P to Resume" message
            resume_text = "Press P to Resume"
            resume_surface = self.font.render(resume_text, True, self.WHITE)
            resume_rect = resume_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
            self.screen.blit(resume_surface, resume_rect)
            
        elif self.target_game.state == self.target_game.STATE_TIME_EXPIRED:
            # Display time expired message
            expired_text = "Time's Up!"
            expired_surface = self.large_font.render(expired_text, True, self.RED)
            expired_rect = expired_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(expired_surface, expired_rect)
            
            # Display restart message
            restart_text = "Press SPACE to Try Again"
            restart_surface = self.font.render(restart_text, True, self.WHITE)
            restart_rect = restart_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
            self.screen.blit(restart_surface, restart_rect)
        
        elif self.target_game.state == self.target_game.STATE_LEVEL_COMPLETE:
            # Display level complete message
            complete_text = f"Level Complete!"
            complete_surface = self.large_font.render(complete_text, True, self.YELLOW)
            complete_rect = complete_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(complete_surface, complete_rect)
            
            # Display score
            score_text = f"Targets: {target_info['score']}/{target_info['targets_for_next_level']}"
            score_surface = self.font.render(score_text, True, self.WHITE)
            score_rect = score_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
            self.screen.blit(score_surface, score_rect)
            
            # Display next level message
            next_text = "Press SPACE to Play Again"
            next_surface = self.font.render(next_text, True, self.WHITE)
            next_rect = next_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 120))
            self.screen.blit(next_surface, next_rect)
        
        # Display score and time information if game has started
        if self.target_game.state in [self.target_game.STATE_PLAYING, self.target_game.STATE_PAUSED]:
            # Display targets progress
            score_text = f"Targets: {target_info['score']}/{target_info['targets_for_next_level']}"
            score_surface = self.font.render(score_text, True, self.YELLOW)
            score_rect = score_surface.get_rect(midtop=(self.screen_width // 2, 10))
            self.screen.blit(score_surface, score_rect)
            
            # Display time information
            time_text = f"Time: {target_info['current_time']:.1f}s / {target_info['time_limit']}s"
            time_surface = self.font.render(time_text, True, self.YELLOW)
            time_rect = time_surface.get_rect(midtop=(self.screen_width // 2, 50))
            self.screen.blit(time_surface, time_rect)
            
            # Display last completion time if available
            if target_info['last_time'] > 0:
                last_text = f"Last Target Time: {target_info['last_time']:.1f}s"
                last_surface = self.font.render(last_text, True, self.YELLOW)
                last_rect = last_surface.get_rect(midtop=(self.screen_width // 2, 90))
                self.screen.blit(last_surface, last_rect)
            
            # Display current position and target information
            position_text = f"Position: ({center_grid_col}, {center_grid_row}) | Target: {target_info['target']}"
            pos_surface = self.font.render(position_text, True, self.YELLOW)
            pos_rect = pos_surface.get_rect(midbottom=(self.screen_width // 2, self.screen_height - 10))
            self.screen.blit(pos_surface, pos_rect)
            
            # Display EMG information
            emg_text = f"EMG: {self.latest_prediction} ({self.latest_intensity:.2f})"
            emg_surface = self.font.render(emg_text, True, self.YELLOW)
            emg_rect = emg_surface.get_rect(topright=(self.screen_width - 20, 10))
            self.screen.blit(emg_surface, emg_rect)
            
            # Draw a crosshair in the center of the screen
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2
            crosshair_size = 15
            
            # Draw crosshair lines
            pygame.draw.line(self.screen, self.YELLOW, (center_x - crosshair_size, center_y), 
                            (center_x + crosshair_size, center_y), 2)
            pygame.draw.line(self.screen, self.YELLOW, (center_x, center_y - crosshair_size), 
                            (center_x, center_y + crosshair_size), 2)

        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Process input
            self.handle_input()
            
            # Update game state
            self.update()
            
            # Render
            self.render()
            
            # Cap the frame rate
            clock.tick(60)
        
        # Clean up when exiting
        self.cleanup()

def main():
    # Create and run the game
    game_manager = GameManager(800, 600)
    
    # Start the game loop
    game_manager.run()

if __name__ == "__main__":
    main()