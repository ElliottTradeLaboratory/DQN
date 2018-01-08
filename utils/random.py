import numpy as np

class Random(object):
    def manualSeed(self, seed):
        self._manualSeed(seed if seed >= 0 else None)

    def _manualSeed(self, seed):
        raise NotImplementedError

    def random(self, a, b=None):
        raise NotImplementedError

    def uniform(self, a=0, b=1, shape=None):
        raise NotImplementedError

class TorchRandom(Random):
    
    def __init__(self):
        """
        import torch
        import lutorpy as lua
        self.torch = require('torch')
        """
        from .torch_random_if import TorchRandomInterface
        
        self.torch = TorchRandomInterface()
        
    def _manualSeed(self, seed):
        self.torch.manualSeed(seed)
    
    def random(self, a=None, b=None):
        return self.torch.random(a, b) - 1

    def uniform(self, a=0, b=1, shape=None):
        if shape is None:
            return self.torch.uniform(a, b)
        else:
            n_elements = np.prod(shape)
            ary = [self.torch.uniform(a, b) for _ in range(n_elements)]
            return np.array(ary).reshape(shape)

class PytorchRandom(Random):
    
    def __init__(self):
        import torch
        self.torch = torch
        self.lt = torch.LongTensor(1)
        self.ft = torch.FloatTensor(1)
        
    def _manualSeed(self, seed):
        self.torch.manual_seed(seed)
    
    def random(self, a=None, b=None):
        if b is None:
            if a is None:
                return self.lt.random_()[0]
            else:
                return self.lt.random_(a)[0]
        return self.lt.random_(a, b)[0]
    
    def uniform(self, a=0, b=1, shape=None):
        if shape is None:
            return self.ft.uniform_(a, b)[0]
        else:
            r = self.torch.FloatTensor(*shape)
            return r.uniform_(a, b).numpy()

class NumpyRandom(Random):
    def _manualSeed(self, seed):
        np.random.seed(seed)

    def random(self, a, b=None):
        if b is None:
            return np.random.randint(a)

        else:
            return np.random.randint(a, b)
    
    def uniform(self, a=0, b=1, shape=None):
        if shape is None:
            return np.random.uniform(a, b)
        else:
            return np.random.uniform(a, b, size=shape)


_random_factory = {
    'torch':TorchRandom,
    'pytorch':PytorchRandom,
    'numpy':NumpyRandom,
}
def _create_random(random_type):
    if not random_type in _random_factory:
        raise ValueError("random_type expected {} but got '{}'".format(list(_random_factory.keys()), random_type))
    return _random_factory.get(random_type)()

_RANDOM = None
_TYPE = None
def get_random(random_type=None, seed=None):
    global _RANDOM, _TYPE
    if random_type is not None:
        _RANDOM = _create_random(random_type)
        _TYPE = random_type
        if seed is not None:
            _RANDOM.manualSeed(seed)
    elif _RANDOM is None:
        raise ValueError('When using this function for the first time, random_type is required.')

    return _RANDOM
