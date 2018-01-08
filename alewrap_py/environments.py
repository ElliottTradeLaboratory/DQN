import sys
import os
import re
import numpy as np
import cv2
from .ale_env import AleEnv
from .game_screen import get_game_screen

from . import get_random

class BaseEnv(object):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, args):
        self.args = args
        self.verbose = args.get('verbose', 2)
        self.frameskip = args.get('actrep', args.get('frameskip', 1))
        self.random_starts = args.get('random_starts', 1)
        self.random = get_random()
        self.normalize = args.get('screen_normalize', 'env') == 'env'

    def getState(self):
        raise NotImplementedError()

    def step(self, action, training):
        raise NotImplementedError()

    def newGame(self):
        raise NotImplementedError()

    def nextRandomGame(self):
        raise NotImplementedError()

    def getActions(self):
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        print(type(self))
        raise NotImplementedError()
    
    def _frameskipping(self, step, action, current_lives, training):
        num_steps = self.frameskip if isinstance(self.frameskip, int) \
                        else self.random.random(self.frameskip[0], self.frameskip[1])
        
        reward = 0

        for skiped_flames in range(num_steps):
            if self.verbose >= 10:
                print('frameskip',skiped_flames)
            obs, r, term, info = step(action)
            reward += r
            lives = info['ale.lives']
            # Normally, the Lives is greater than 1 and the Game continues until it is zero.
            # This also applies to the evaluation at DQN 3.0.
            # But when the training at DQN3.0, the game is terminate even if get a loss just once.
            if training and lives > 0 and lives < current_lives :
                #print('training', training, 'lives < current_lives',lives , current_lives)
                term = True

            if term:
                #print('break')
                break

        info['frameskip'] = skiped_flames+1
        return obs, reward, term, info

    def _setup_random_starts(self, step, k):

        s, r, t, info = self.newGame()

        k = k if k is not None else self.random.random(self.random_starts) + 1

        if self.verbose >= 10:
            print('BaseEnv._setup_random_starts k=',k)

        for i in range(1, k):
            if self.verbose >= 10:
                print('BaseEnv._setup_random_startse  count=',i+1)
            _, _, term, _ = step(0)
            if term:
                print(string.format('WARNING: Terminal signal received after %d 0-steps', i))

        return step(0)

    @property
    def game_name(self):
        raise NotImplementedError()

    @property
    def frame_shape(self):
        raise NotImplementedError()

    def __del__(self):
        raise NotImplementedError()
    
class GameEnvironment(BaseEnv):

    def __init__(self, args):
        super(GameEnvironment, self).__init__(args)

        if self.verbose >= 10:
            print('GameEnvironment.__init__')

        self._screen = get_game_screen(args)
        self._reset()

    def _updateState(self, frame, reward, terminal, info):
        if self.verbose >= 10:
            print('GameEnvironment._updateState')
        self._state.reward       = reward
        self._state.terminal     = terminal
        #self._state.prev_lives  = self._state.lives or lives
        self._state.info         = info
        self._state.lives        = info['ale.lives']
        return self

    def getState(self):
        if self.verbose >= 10:
            print('GameEnvironment.getState')

        # grab the screen again only if the state has been updated in the meantime
        self._state.observation  = self._screen.grab()

        return self._state.observation, self._state.reward, self._state.terminal, self._state.info

    def _reset(self):
        if self.verbose >= 10:
            print('GameEnvironment._reset')
            
        if not re.match('^.*-v.*$', self.args['env']):
            self.env = AleEnv(self.args)
        else:
            self.env = GymAleEnvWrapper(self.args)
            self.frameskip = self.env.frameskip

        self._actions = self.getActions()
        # start the game
        if self.verbose > 0 :
            print('\nPlaying:', self.env.game_name)
        self._resetState()
        self._updateState(*self._step(0))
        self.getState()
        return self

    def _resetState(self):
        if self.verbose >= 10:
            print('GameEnvironment._resetState')

        self._screen.clear()
        class State:
            pass
        self._state = State()
        return self

    # Function plays `action` in the game and return game state.
    def _step(self, action):
        if self.verbose >= 10:
            print('GameEnvironment._step act:', action)
        s, r, t, lives = self.env.envStep(action)
        self._screen.paint(s)
        info = {'ale.lives':lives}
        return s, r, t, info

    def _randomStep(self):
        if self.verbose >= 10:
            print('GameEnvironment._randomStep')
        act = self._actions[self.random.random(len(self._actions))]
        return self._step(act)

    def step(self, action, training):
        if self.verbose >= 10:
            print('GameEnvironment.step act:', action)

        ret = self._frameskipping(self._step, action, self._state.lives, training)
        return self._updateState(*ret).getState()
        
    """
    Function advances the emulator state until a new game starts and returns
    this state. The new game may be a different one, in the sense that playing back
    the exact same sequence of actions will result in different outcomes.
    """
    def newGame(self):
        if self.verbose >= 10:
            print('GameEnvironment.newGame')
        terminal = self._state.terminal
        if self.verbose >= 10:
            print('terminal', terminal)
        while not terminal:
            obs, reward, terminal, info = self._randomStep()
        self._screen.clear()
        
        return self._updateState(*self._step(0)).getState()

    """
    Function advances the emulator state until a new (random) game starts and
    returns this state.
    """
    def nextRandomGame(self, k=None):
        if self.verbose >= 10:
            print('GameEnvironment.nextRandomGame')
        
        ret = self._setup_random_starts(self._step, k)

        return self._updateState(*ret).getState()

    # Function returns a table with valid actions in the current game.
    def getActions(self):
        if self.verbose > 10:
            print('GameEnvironment.getActions')
        return self.env.actions()

        
    def render(self, mode='human', close=False):
            
        if close:
            cv2.destroyAllWindows()
            return
        img = self._state.observation * 255 if self.normalize else \
              self._state.observation
        img = img.astype(np.uint8)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow('show', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            k = cv2.waitKey(1)

    @property
    def game_name(self):
        return self.env.game_name

    @property
    def frame_shape(self):
        return self.env.frame_shape + (3,)

    def __del__(self):
        if hasattr(self, 'env'):
            del self.env


class GymEnvWrapper(BaseEnv):

    def __init__(self, args):
        super(GymEnvWrapper, self).__init__(args)

        import gym
        from collections import deque
        from gym.envs.atari import AtariEnv

        self._game_name = args['env']
        env = gym.make(self._game_name)

        env_deque = deque([env])
        atari_env = None
        while len(env_deque):
            tmpenv = env_deque.pop()
            if isinstance(tmpenv, AtariEnv):
                atari_env = tmpenv
                break
            if hasattr(tmpenv, 'env'):
                env_deque.append(tmpenv.env)
                
        if atari_env is not None:
            self.env = atari_env
            self._actions = atari_env._action_set.tolist()
        else:
            self.env = env
            self._actions = [ a for a in range(env.action_space.n)]

    def _getState(self, *args):
        if len(args) == 4:
            s, r, t, info = args
        elif len(args) == 1:
            s, r, t, info = *args, 0, False, {}
        else:
            assert False, 'length of args is only 1 or 4.'
        if self.normalize:
            s = s.astype(np.float32) / 255.0
        else:
            s = s.astype(np.float32)
        return s, r, t, info

    def getState(self):
        return self._getState(self.env.reset())

    def step(self, action, training):
        assert self._actions.count(action) == 1, '{} is not in actions {}'.format(action, self._actions)
        act_index = self._actions.index(action)
        return self._getState(*self.env.step(act_index))

    def newGame(self):
        return self._getState(self.env.reset())

    def nextRandomGame(self, k=None):
        self._setup_random_starts(self.env.step, k)
        return self._getState(*self.env.step(0))

    def getActions(self):
        return self._actions

    def render(self, mode='human', close=False):
        return self.env.render(mode, close)

    @property
    def game_name(self):
        return self._game_name

    @property
    def frame_shape(self):
        return self.env.observation_space.shape

    def __del__(self):
        if hasattr(self, 'env'):
            del self.env

