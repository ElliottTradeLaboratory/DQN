import numpy as np

class GameScreen(object):
    def __init__(self, args):
        self.args = args
        self.verbose = args.get('verbose', 2)
        if self.verbose >= 10:
            print('GameScreen.__init__')
        self._reset(args)
  
    def _reset(self, args):
        self.frameBuffer = None
        self.poolBuffer = None
        self.lastIndex = 0
        pool_frms = args.get('pool_frms', dict(size=2, type='max'))
        self.screen_normalize = args.get('screen_normalize', 'env') == 'env' and args.get('maximization') == 'env'
        self.bufferSize = pool_frms['size']
        self.poolType = pool_frms['type']
        if self.poolType != 'max':
            raise ValueError("got '{}', expected only 'max'.".format(self.poolType))

    def clear(self):
        if self.verbose >= 10:
            print('GameScreen.clear')
        if self.frameBuffer is not None:
            self.frameBuffer.fill(0)
        self.lastIndex = 0

    def grab(self):
        if self.verbose >= 10:
            print('GameScreen._grab')
        self.poolBuffer = self.frameBuffer.max(0)
        if self.verbose >= 10:
            print('self.poolBuffer.shape', self.poolBuffer.shape)
        return self.poolBuffer

    def paint(self, frame):
        if self.verbose >= 10:
            print('GameScreen._paint')
        
        if self.frameBuffer is None:
            self.frameBuffer = np.empty((self.bufferSize,) + frame.shape, dtype=np.float32)
            self.clear()
            
        if self.lastIndex >= self.bufferSize:
            self.lastIndex = 0

        self.frameBuffer[self.lastIndex, :] = frame.astype(np.float32) / 255.0 if self.screen_normalize else \
                                              frame.astype(np.float32)
        self.lastIndex += 1

class NopGameScreen(object):
    def __init__(self, args):
        self.args = args
        self.verbose = args.get('verbose', 2)
        if self.verbose >= 10:
            print('NopGameScreen._grab')
        self._reset(args)
  
    def _reset(self, args):
        self.poolBuffer = None
        self.screen_normalize = args.get('screen_normalize', 'env') == 'env'

    def clear(self):
        if self.verbose >= 10:
            print('NopGameScreen.clear')

    def grab(self):
        if self.verbose >= 10:
            print('NopGameScreen._grab')
        return self.poolBuffer

    def paint(self, frame):
        if self.verbose >= 10:
            print('NopGameScreen._paint')
        
        self.poolBuffer = frame.astype(np.float32) / 255 if self.screen_normalize else \
                          frame.astype(np.float32)


def get_game_screen(args):

    return GameScreen(args) if args.get('maximization', 'env') == 'env' else\
           NopGameScreen(args)
