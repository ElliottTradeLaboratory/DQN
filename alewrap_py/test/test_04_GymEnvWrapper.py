import sys
import os
import time
import threading
import unittest
from unittest.mock import *

import numpy as np
import gym
from torch.utils.serialization import load_lua

sys.path.append('../../')
sys.path.append('../../dqn')

from alewrap_py import get_env
from utils import get_random, Namespace


class TestGymEnvWrapper_GameEnvWrapper(unittest.TestCase):

    def setUp(self):
        random = get_random('pytorch', 1)

    def test_00_init0(self):
        from alewrap_py.environments import GymEnvWrapper

        env = GymEnvWrapper(dict(env='Breakout-v0'))
        from gym import Env
        self.assertIsNotNone(env.env)
        self.assertIsInstance(env.env, gym.envs.atari.AtariEnv)
        self.assertEqual(env._actions, [0,1,3,4])

        env = GymEnvWrapper(dict(env='CartPole-v0'))
        self.assertIsInstance(env.env, gym.Env)
        self.assertEqual(env._actions, [0,1])

    def test_01_getActions(self):

        from alewrap_py.environments import GymEnvWrapper
        
        args = dict(env='Breakout-v0')
        env = GymEnvWrapper(args)
        self.assertEqual(env.getActions(), [0,1,3,4])
        
    def test_02_getState(self):

        from alewrap_py.environments import GymEnvWrapper
        
        args = dict(env='Breakout-v0')
        env = GymEnvWrapper(args)

        org_reset = env.env.reset
        reset_s = None
        def inject_reset():
            nonlocal reset_s
            reset_s = org_reset()
            return reset_s

        @patch.object(env.env, "reset", wraps=inject_reset)
        @patch.object(env.env, "step", wraps=env.env.step)
        def test_func(mock_step, mock_reset):

            s, r, t, info = env.getState()

            mock_reset.assert_called_once_with()
            mock_step.assert_not_called()

            # The GameEnvWrapper method returns normalized observations
            #  after receiving from AtariEnv.reset() or step()
            self.assertTrue(np.all(s == reset_s.astype(np.float32)/255))
            self.assertEqual(r, 0)
            self.assertEqual(info, {})

        test_func()

    def test_03_step1(self):

        from alewrap_py.environments import GymEnvWrapper
        
        args = dict(env='Breakout-v0', actrep=4)
        env = GymEnvWrapper(args)
        s, r, t, info = env.getState()

        # the alewrap_py reproduces alewrap on python, 
        # therefore the argument of step() is not index of action vector
        # like gym environment, It is raw action value for ALE.
        # If you choised gym environment like 'Breakout-v0',
        # the GameEnvWrapper will convert action value to
        # action index before call the step() of gym environment.
        with self.assertRaisesRegex(AssertionError, '2 is not in actions \[0, 1, 3, 4\]'):
            s, r, t, info = env.step(2, True)

    def test_03_step2(self):

        from alewrap_py.environments import GymEnvWrapper
        
        args = dict(env='Breakout-v0', actrep=4)
        env = GymEnvWrapper(args)
        s, r, t, info = env.getState()

        org_step = env.env.step
        step_s = None
        step_r = None
        step_t = None
        step_info = None
        def inject_step(action):
            nonlocal step_s, step_r, step_t, step_info

            step_s, step_r, step_t, step_info = org_step(action)
            return step_s, step_r, step_t, step_info

        @patch.object(env.env, "reset", wraps=env.env.reset)
        @patch.object(env.env, "step", wraps=inject_step)
        def test_func(mock_step, mock_reset):

            s, r, t, info = env.step(0, True)

            mock_reset.assert_not_called()

            # both 'actrep' and 'frameskip' has become no meaning.
            mock_step.assert_called_once_with(0)

            self.assertTrue(np.all(s == step_s.astype(np.float32)/255))
            self.assertEqual(r, 0)
            self.assertEqual(info, {'ale.lives':5})

        test_func()

        
    def test_02_newGame(self):

        from alewrap_py.environments import GymEnvWrapper
        
        args = dict(env='Breakout-v0')
        env = GymEnvWrapper(args)

        org_reset = env.env.reset
        reset_s = None
        def inject_reset():
            nonlocal reset_s
            reset_s = org_reset()
            return reset_s

        @patch.object(env.env, "reset", wraps=inject_reset)
        @patch.object(env.env, "step", wraps=env.env.step)
        def test_func(mock_step, mock_reset):

            s, r, t, info = env.newGame()

            mock_reset.assert_called_once_with()
            mock_step.assert_not_called()

            self.assertTrue(np.all(s == reset_s.astype(np.float32)/255))
            self.assertEqual(r, 0)
            self.assertEqual(info, {})

        test_func()

    def test_02_nextRandomGame1(self):

        from alewrap_py.environments import GymEnvWrapper
        
        args = dict(env='Breakout-v0')
        env = GymEnvWrapper(args)

        org_reset = env.env.reset
        reset_s = None
        def inject_reset():
            nonlocal reset_s
            reset_s = org_reset()
            return reset_s

        @patch.object(env.env, "reset", wraps=inject_reset)
        @patch.object(env.env, "step", wraps=env.env.step)
        def test_func(mock_step, mock_reset):

            s, r, t, info = env.nextRandomGame()

            mock_reset.assert_called_once_with()
            self.assertEqual(mock_step.call_count, 2)

            self.assertTrue(np.all(s == reset_s.astype(np.float32)/255))
            self.assertEqual(r, 0)
            self.assertEqual(info, {'ale.lives':5})

        test_func()
