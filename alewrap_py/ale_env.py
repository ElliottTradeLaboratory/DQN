import os
import sys
import re
import numpy as np
from .ale_python_interface import ALEInterface

def _game_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "atari_roms")

def get_game_path(game_name):
    return os.path.join(_game_dir(), game_name) + ".bin"

def list_games():
    files = os.listdir(_game_dir())
    return [os.path.basename(f).split(".")[0] for f in files]

class AleEnv(object):

    def __init__(self, args):
    
        self._game_name = args['env']
        game_path = get_game_path(self._game_name)
        if not os.path.exists(game_path):
            raise ValueError('{}.bin not found.({})'.format(self._game_name, game_path))
        
        self.ale = ALEInterface(game_path)
        self.obsShape = self.ale.getScreenDims()
        self.game_over_reward = args.get('game_over_reward', 0)
        
        self.envStart()

        # setup initial observations by playing a no-action command
        self.saveState()
        self.envStep(0)
        self.loadState()
    
    def envStart(self):
        self.ale.resetGame()
        return self._generateObservations()
    
    def _generateObservations(self):
        return self.ale.getGameScreenRGB()

    def saveState(self):
        self.ale.saveState()
    
    def loadState(self):
        self.ale.loadState()
    
    def envStep(self, action):
        if self.isGameOver():
            self.ale.resetGame()
            reward = self.game_over_reward
            # The first screen of the game will be also
            # provided as the observation.
        else:
            reward = self.ale.act(action)
        return self._generateObservations(), reward, self.isGameOver(), self.lives()

    def isGameOver(self):
        return self.ale.isGameOver()
    
    def lives(self):
        return self.ale.lives()
    
    def actions(self):
        return self.ale.getMinimalActionSet()

    @property
    def game_name(self):
        return self._game_name

    @property
    def frame_shape(self):
        return self.obsShape

    def __del__(self):
        if hasattr(self, 'ale'):
            del self.ale


