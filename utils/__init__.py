import sys
import re
import numpy as np

from .random import get_random
from .namespace import Namespace

_SESSION = None
def get_session():
    import tensorflow as tf
    global _SESSION

    if _SESSION is None:
        
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        _SESSION = tf.Session(config=config)

    return _SESSION

class Unbuffered(object):
    def __init__(self, stream, f):
        self.stream = stream
        self.f = f
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.f.write(data)
        self.f.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
        self.f.write(datas)
        self.f.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

np.set_printoptions(threshold=np.inf)

