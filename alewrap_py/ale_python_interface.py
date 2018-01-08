__all__ = ['ALEInterface']

from ctypes import *
import numpy as np
from numpy.ctypeslib import as_ctypes
import os
import six
from time import time

debug = False

libxitari = 'xitari/libxitari.so' if not debug else 'xitari/libxitari_debug.so'
ale_lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),
                                        libxitari))

ale_lib.fillRgbFromPalette.argtypes=[c_void_p, c_void_p, c_int, c_int]
ale_lib.fillRgbFromPalette.restype=None
ale_lib.ALE_new.argtypes = None
ale_lib.ALE_new.restype = c_void_p
ale_lib.ALE_del.argtypes = [c_void_p]
ale_lib.ALE_del.restype = None
ale_lib.act.argtypes = [c_void_p, c_int]
ale_lib.act.restype = c_double
ale_lib.resetGame.argtypes = [c_void_p]
ale_lib.resetGame.restype = None
ale_lib.getScreenRGB.argtypes =[ c_void_p, c_void_p, c_int]
ale_lib.getScreenRGB.restype = None
ale_lib.getScreenWidth.argtypes =[ c_void_p]
ale_lib.getScreenWidth.restype = c_int
ale_lib.getScreenHeight.argtypes = [c_void_p]
ale_lib.getScreenHeight.restype = c_int
ale_lib.isGameOver.argtypes = [c_void_p]
ale_lib.isGameOver.restype = c_bool
ale_lib.loadState.argtypes = [c_void_p]
ale_lib.loadState.restype = c_bool
ale_lib.saveState.argtypes = [c_void_p]
ale_lib.saveState.restype = None
ale_lib.fillObs.argtypes = [c_void_p, c_void_p, c_int]
ale_lib.fillObs.restype = None
ale_lib.fillRamObs.argtypes = [c_void_p, c_void_p, c_int]
ale_lib.fillRamObs.restype = None
ale_lib.numLegalActions.argtypes = [c_void_p]
ale_lib.numLegalActions.restype = c_int
ale_lib.legalActions.argtypes = [c_void_p, c_void_p, c_int]
ale_lib.legalActions.restype = None
ale_lib.livesRemained.argtypes = [c_void_p]
ale_lib.livesRemained.restype = c_int
ale_lib.saveSnapshot.argtypes = [c_void_p, c_void_p, c_int]
ale_lib.saveSnapshot.restype = None
ale_lib.restoreSnapshot.argtypes = [c_void_p, c_void_p, c_int]
ale_lib.restoreSnapshot.restype = None
if not debug:
    ale_lib.maxReward.argtypes = [c_void_p]
    ale_lib.maxReward.restype = c_int

def _as_bytes(s):
    if hasattr(s, 'encode'):
        return s.encode('utf8')
    return s

class ALEInterface(object):
    def __init__(self, rom_file):
        self.obj = ale_lib.ALE_new(_as_bytes(rom_file))
        self.count = 0

    def act(self, action):
        return int(ale_lib.act(self.obj, action))

    def isGameOver(self):
        ret = ale_lib.isGameOver(self.obj)
        return ret
    def resetGame(self):
        ale_lib.resetGame(self.obj)

    def getScreenDims(self):
        w = ale_lib.getScreenWidth(self.obj)
        h = ale_lib.getScreenHeight(self.obj)
        return (h, w)

    def loadState(self):
        return ale_lib.loadState(self.obj)

    def saveState(self):
        ale_lib.saveState(self.obj)
        
    def fillObs(self, obs, obs_size):
        ale_lib.fillObs(self.obj, as_ctypes(obs), obs_size)
        
    def fillRamObs(self, obs, obs_size):
        ale_lib.fillRamObs(self.obj, as_ctypes(obs), obs_size)

    def numMinimalActions(self):
        return ale_lib.numLegalActions(self.obj)
        
    def getMinimalActionSet(self):
        action_size = self.numMinimalActions()
        actions = np.zeros(action_size, dtype=np.intc)
        ale_lib.legalActions(self.obj, as_ctypes(actions), action_size)
        return actions.tolist()

    def getGameScreenRGB(self):
        st = time()
        if not hasattr(self, 'height'):
            self.height, self.width  = self.getScreenDims()
        rgb_size = int(self.height* self.width * 3)
        if not hasattr(self, 'rgb'):
            self.rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        ale_lib.getScreenRGB(self.obj, as_ctypes(self.rgb[:]),rgb_size)
        """
        print('getGameScreenRGB({}) {:0.4f}'.format(self.count, time()-st))
        if self.count == 4:
            import sys
            sys.stdout.flush()
            assert False
        self.count += 1
        """
        return self.rgb

    def lives(self):
        return ale_lib.livesRemained(self.obj)
    
    def saveSnapshot(self, data, lenght):
        ale_lib.saveSnapshot(self.obj, as_ctypes(data), lenght)

    def restoreSnapshot(self, snapshot, size):
        ale_lib.restoreSnapshot(self.obj, as_ctypes(snapshot), size)

    def maxReward(self):
        if debug:
            return 0
        else:
            return ale_lib.maxReward(self.obj)

    def _game_dir(self):
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "atari_roms")

    def _get_game_path(self, game_name):
        return os.path.join(self._game_dir(), game_name) + ".bin"

    def list_games(self):
        files = os.listdir(self._game_dir())
        return [os.path.basename(f).split(".")[0] for f in files]

    def __del__(self):
        ale_lib.ALE_del(self.obj)
