import sys
import re
import numpy as np

sys.path.append('../')

try:
    from utils import get_random
except ImportError:
    class NumpyRandom(object):
        def random(self, a, b=None):
            if b is None:
                return np.random.randint(a)

            else:
                return np.random.randint(a, b)

    def get_random():
        return NumpyRandom()

from .environments import GameEnvironment, GymEnvWrapper
from .game_screen import get_game_screen
from .recorder import RecorderEnv
from .render import Render
from .ale_env import AleEnv
from .ale_python_interface import ALEInterface

def get_env(opt):
    assert 'env' in opt, "'env' is required."
    if re.match(r'^.*-v.*$', opt['env']):
        env = GymEnvWrapper(opt)
    else:
        env = GameEnvironment(opt)
    return RecorderEnv(env, opt)

ACTION_MEANING = np.array([
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
])
