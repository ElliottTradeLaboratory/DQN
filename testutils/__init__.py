import numpy as np

from .assertions import *
from .context_manager import *

def _make_state_one(tables, index):
    sample_s = tables[index:index+4,0]

    state = np.empty((4,84,84))
    
    for i, s in enumerate(sample_s):
        state[i,...] = s

    return state

def make_state(tables, index, batch_size=1):
    
    if batch_size == 1:
        return _make_state_one(tables, index)
    else:
        state = np.empty((batch_size, 4,84,84))
        
        for i in range(batch_size):
            state_one = _make_state_one(tables, index+i)
            state[i,...] = state_one

    return state
