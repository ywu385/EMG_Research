"""
Game components package for EMG Navigation Game.
Contains the game manager, target game, and spiral game components.
"""

# This allows importing from the package
from game_comp.game_manager import GameManager
from game_comp.target_game import TargetGame
from game_comp.GridSpiralGame import GridSpiralChallenge

# Define what should be imported when using "from game_comp import *"
__all__ = ['GameManager', 'TargetGame', 'GridSpiralChallenge']