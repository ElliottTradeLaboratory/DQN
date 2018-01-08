import sys
import os
import unittest
from unittest.mock import *

import numpy as np

sys.path.append('../../')

class TestAleEnv(unittest.TestCase):

    def test_00_init0(self):
    
        from alewrap_py.ale_env import AleEnv
        with self.assertRaises(KeyError):
            env = AleEnv({})

    @patch('alewrap_py.ale_env.ALEInterface')
    def test_00_init2(self, mock_ale):

        args = dict(env = 'breakout')

        instance = mock_ale.return_value
        instance.isGameOver.return_value = False
        
        from alewrap_py.ale_env import AleEnv
        env = AleEnv(args)

        bin_path = os.path.abspath('../atari_roms/{}.bin'.format(args['env']))
        """
            The AleEnv of alewrap_py was merged both game(AleLayer.lua) and AleEnv(AleEnv.lua).
            That interface like a AleEnv(AleEnv.lua).
            And in the alewrap_py has simplified methods than alewap.
            In the alewrap, sequence of get the screen size is:
                1) ale.getScreenWidth()
                2) ale.getScreenHeight()
            but, in the alewrap_py is only ale.getScreenDims() that including above calls.
            In the alewrap, sequence of get the game screen is:
                1) ale.getScreenWidth()
                2) ale.getScreenHeight() both calls are get the size of torch tensor 
                   for save the screen data at next step 3).
                3) ale.fillObs() that screen-data set into torch tensor data area. 
                4) if useRGB is true , then call ale.getRgbFromPallet
                   that convert from raw screen data to RGB data.
            but, in the alewrap_py is:
                1) Skip above 1) and 2) because in the alewrap_py, 
                   pre-allocated buffer for save the screen data is unnecessary.
                2) Above 3) and 4) are merge to ale.getGameScreenRGB().
            These approach used the atari_py as reference.
        """
        expected = [
            call(bin_path),
            call().getScreenDims(),
            call().resetGame(),
            call().getGameScreenRGB(),
            call().saveState(),
            call().isGameOver(),
            call().act(0),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
            call().loadState(),
        ]
        self.assertTrue(mock_ale.mock_calls == expected)

    def test_00_init3(self):
        args = dict(env = 'breakout')

        from alewrap_py.ale_env import AleEnv
        env = AleEnv(args)

        self.assertTrue(env.obsShape == (210, 160))


    @patch('alewrap_py.ale_env.ALEInterface.getMinimalActionSet')
    def test_01_actions1(self, mock_getMinimalActionSet):

        """
            In the alewarp, sequence of get the actions that dependence to selected game is:
                1) ale.numActions() is get number of actions that size of buffer for save the action values.
                2) ale.actions is get action values that dependence to selected game.
            but in the alewrap_py is only ale.getMinimalActionSet()
        """

        from alewrap_py.ale_env import AleEnv

        args = dict(env = 'breakout')
        
        instance = mock_ale.return_value
        instance.isGameOver.return_value = False
        
        env = AleEnv(args)

        actions = env.actions()

        mock_getMinimalActionSet.assert_called_once_with()
        
    def test_01_actions1(self):

        from alewrap_py.ale_env import AleEnv

        args = dict(env = 'breakout')

        env = AleEnv(args)

        actions = env.actions()

        self.assertEqual(actions, [0,1,3,4])


    def test_02_envStep1(self):

        # envStep returns:
        # 1) observation: np.ndarray
        # 2) reward     : int
        # 3) termial    : bool
        # 4) lives      : int
        
        from alewrap_py.ale_env import AleEnv

        args = dict(env = 'breakout')

        env = AleEnv(args)
        s, r, t, lives = env.envStep(0)

        self.assertIsInstance(s, np.ndarray)
        self.assertIsInstance(r, int)
        self.assertIsInstance(t, bool)
        self.assertIsInstance(t, int)


    @patch('alewrap_py.ale_env.ALEInterface')
    def test_02_envStep2(self, mock_ale):

        from alewrap_py.ale_env import AleEnv

        args = dict(env = 'breakout')

        # If the ALEInterface.isGameOver returns True during the AleEnv.envStep,
        # it will call the ALEInterface.resetGame to reset the game,
        # and get the first screen image in a new game
        # by the ALEInterface.getGameScreenRGB() and return it.
        instance = mock_ale.return_value
        return_values = [False] * 11
        return_values += [True, True, True] # When the game is over, ale.isGameOver() call 3 times with returns True every time.
        instance.isGameOver.side_effect = return_values

        env = AleEnv(args)

        from collections import deque
        actions = deque([0, 1, 3, 4, 0, 1])
        t = False
        for i in range(100):
            if len(actions) == 0:
                break
            with self.subTest(i=i):
                s, r, t, lives = env.envStep(actions.popleft())


        bin_path = os.path.abspath('../atari_roms/{}.bin'.format(args['env']))
        expected = [
            call(bin_path),
            call().getScreenDims(),
            call().resetGame(),
            call().getGameScreenRGB(),
            call().saveState(),
            call().isGameOver(),
            call().act(0),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
            call().loadState(),
            call().isGameOver(),
            call().act(0),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
            call().isGameOver(),
            call().act(1),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
            call().isGameOver(),
            call().act(3),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
            call().isGameOver(),
            call().act(4),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
            call().isGameOver(),
            call().act(0),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
            call().isGameOver(),
            call().resetGame(),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
        ]
        self.assertTrue(mock_ale.mock_calls == expected)

    def test_03_game_name(self):

        from alewrap_py.ale_env import AleEnv

        args = dict(env = 'breakout')

        env = AleEnv(args)
        self.assertEqual(env.game_name, 'breakout')
