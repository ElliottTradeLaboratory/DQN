from datetime import datetime
import numpy as np
from utils import get_random
from alewrap_py import get_env
from common import get_extensions

def _random_setup(opt):

    get_random(opt.random_type, opt.seed)
    
    if opt.verbose >= 1:
        print('Seed:{0}'.format(opt.seed))

    return opt


def setup(opt, build_agent=True):

    np.set_printoptions(threshold=np.inf)

    opt = _random_setup(opt)

    Ex = get_extensions(opt)
    opt = Ex.setup(opt)

    # load training environment
    gameEnv = get_env(opt)
    gameActions = gameEnv.getActions()
    
    opt.actions   = gameActions
    opt.n_actions   = len(gameActions)

    if build_agent:
        from agents import DQNAgent
        agent = DQNAgent(opt)
    else:
        agent = None

    if opt.verbose >= 1 :
        print('Options:')
        opt.summary()

    return gameEnv, agent, gameActions, opt
