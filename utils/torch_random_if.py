__all__ = ['TorchRandomInterface']

import os
from ctypes import *

debug = False

try:
    _TORCH_INSTALL = os.environ['TORCH_INSTALL']
except KeyError:
    raise RuntimeError('Needs set torch/install path to the environment variable TORCH_INSTALL.')

libTH_so = os.path.join(_TORCH_INSTALL,'lib/libTH.so')
libTH = cdll.LoadLibrary(libTH_so)

libTH.THGenerator_new.argtypes=None
libTH.THGenerator_new.restype=c_void_p
libTH.THGenerator_free.argtypes=[c_void_p]
libTH.THGenerator_free.restype=None
libTH.THRandom_manualSeed.argtypes=[c_void_p, c_ulong]
libTH.THRandom_manualSeed.restype=None
libTH.THRandom_random.argtypes=[c_void_p]
libTH.THRandom_random.restype=c_ulong
libTH.THRandom_uniform.argtypes=[c_void_p, c_double, c_double]
libTH.THRandom_uniform.restype=c_double


class TorchRandomInterface(object):
    def __init__(self):
        self.obj = libTH.THGenerator_new()

    def manualSeed(self, seed):
        libTH.THRandom_manualSeed(self.obj, seed)

    def random(self, a=None, b=None):
        if b is None:
            if a is None:
                return libTH.THRandom_random(self.obj)
            return self._random1(a)
        return self._random2(a, b)

    def _random1(self, b):
        return (libTH.THRandom_random(self.obj) % b) + 1

    def _random2(self, a, b):
        return (libTH.THRandom_random(self.obj) % (b+1-a)) + a

    def uniform(self, a=0, b=1):
        return libTH.THRandom_uniform(self.obj, a, b)

    def __del__(self):
        libTH.THGenerator_free(self.obj)
