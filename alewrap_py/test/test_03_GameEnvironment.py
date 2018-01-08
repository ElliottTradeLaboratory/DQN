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
from testutils import *

class TestGameEnvironment(unittest.TestCase):

    def setUp(self):
        random = get_random('pytorch', 1)

    def test_00_init0(self):
        from alewrap_py.environments import GameEnvironment

        with self.assertRaises(KeyError):
            env = GameEnvironment({})
        
        env = GameEnvironment(dict(env='breakout'))
        from alewrap_py.environments import GameEnvironment
        from alewrap_py.game_screen import GameScreen
        from alewrap_py.ale_env import AleEnv
        self.assertIsNotNone(env.env)
        self.assertIsInstance(env.env, AleEnv)
        self.assertIsInstance(env._screen, GameScreen)
        self.assertEqual(env.frameskip, 1)
        self.assertEqual(env._actions, [0,1,3,4])

    @patch('alewrap_py.ale_env.ALEInterface')
    def test_00_init2(self, mock_ale):

        instance = mock_ale.return_value
        return_values = [False] * 4
        instance.isGameOver.side_effect = return_values
        instance.getGameScreenRGB.return_value = np.zeros((210,160,3))

        from alewrap_py.environments import GameEnvironment
        
        args = dict(env='breakout', actrep=4, random_starts=30)
        env = GameEnvironment(args)

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
            call().getMinimalActionSet(),
            call().isGameOver(),
            call().act(0),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
        ]

        self.assertTrue(mock_ale.mock_calls == expected)

    @patch('alewrap_py.ale_env.ALEInterface')
    def test_01_getState1(self, mock_ale):

        instance = mock_ale.return_value
        return_values = [False] * 4
        instance.isGameOver.side_effect = return_values
        
        screen = np.empty((210,160,3)).astype(np.uint8)
        screens = [screen+5, screen+4,screen+3,screen+2,screen+1]
        instance.getGameScreenRGB.side_effect = screens

        from alewrap_py.environments import GameEnvironment
        
        args = dict(env='breakout', actrep=4, random_starts=30)
        env = GameEnvironment(args)
        s, r, t, info = env.getState()
        
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
            call().getMinimalActionSet(),
            call().isGameOver(),
            call().act(0),
            call().getGameScreenRGB(),
            call().isGameOver(),
            call().lives(),
        ]

        self.assertTrue(mock_ale.mock_calls == expected)

        # The getGameScreenRGB() called 3 times and get screen image every time,
        # but the env will saved only once that return of final call.
        self.assertTrue(np.all(s*255 == (screen.astype(np.float32) + 3)))

    def test_01_getState2(self):

        from alewrap_py.environments import GameEnvironment
        
        args = dict(env='breakout', actrep=4, random_starts=30)
        env = GameEnvironment(args)
        s, r, t, info = env.getState()

        self.assertEqual(r, 0)
        self.assertFalse(t)
        self.assertEqual(info, {'ale.lives':5})

    @patch('alewrap_py.environments.AleEnv.envStep')
    def test_02_step1(self, mock_envStep):
        class Unbuffered(object):
           def __init__(self, stream):
               self.stream = stream
           def write(self, data):
               self.stream.write(data)
               self.stream.flush()
           def writelines(self, datas):
               self.stream.writelines(datas)
               self.stream.flush()
           def __getattr__(self, attr):
               return getattr(self.stream, attr)
        sys.stdout = Unbuffered(sys.stdout)
        

        return_values = [np.ones((210, 160, 3)), 0, False, 5]
        mock_envStep.return_value = return_values

        import alewrap_py.environments as e
        
        # [frameskip test]
        # If both 'actrep' and 'frameskip' are not specified use default skip counts that 1.
        args = dict(env='breakout', random_starts=30)
        env = e.GameEnvironment(args)
        
        env.getState()
        mock_envStep.reset_mock()
        env.step(0, True)
        
        self.assertEqual(mock_envStep.call_count, 1)

        
        # [frameskip test]
        # If 'actrep' is 2 then Ale.envStep() is called 2 times.
        args = dict(env='breakout', actrep=2, random_starts=30)
        env = e.GameEnvironment(args)
        
        env.getState()
        mock_envStep.reset_mock()
        env.step(0, True)
        
        self.assertEqual(mock_envStep.call_count, 2)

        # [frameskip test]
        # If 'frameskip' is 4 then Ale.envStep() is called 4 times.
        args = dict(env='breakout', frameskip=4, random_starts=30)
        env = e.GameEnvironment(args)
        
        env.getState()
        mock_envStep.reset_mock()
        env.step(0, True)
        
        self.assertEqual(mock_envStep.call_count, 4)

    @patch('alewrap_py.environments.AleEnv.envStep')
    def test_02_step2(self, mock_envStep):

        screen = np.zeros((210, 160, 3), dtype=np.float32)
        return_values = [[screen +50,12, False, 5],
                         [screen +49,11, False, 5],
                         [screen +48,10, False, 4],
                         [screen +47, 9, False, 4],
                         [screen +46, 8, False, 3],
                         [screen +45, 7, False, 3],
                         [screen +44, 6, False, 2],
                         [screen +43, 5, False, 2],
                         [screen +42, 4, False, 2],
                         [screen +41, 3, False, 1],
                         [screen +40, 2, True , 0],
                         [screen +39, 1, False, 5],
                         [screen +38, 0, False, 5],
                         [screen +37, 100, False, 5],
                         [screen +36, 101, False, 5],
                         [screen +35, 102, False, 5],
                         [screen +34, 103, False, 5],
                         ]
        mock_envStep.side_effect = return_values

        args = dict(env='breakout', frameskip=1, random_starts=30)
        import alewrap_py.environments as e
        env = e.GameEnvironment(args)
        
        env.getState()

        #[training mode test]
        # In the training mode, returns the terminal that True every time the lives decrease.
        # The s of return is maximum value between current screen image and pre-screen image.
        mock_envStep.reset_mock()
        mock_envStep.side_effect = return_values

        # the GameScreen.frameBuffer has first screen image that got and saved by the getState().
        # It clearing up for testing.
        env._screen.clear()

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 1)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 5)
        self.assertFalse(t)
        self.assertEqual(r, 12)
        self.assertTrue(np.all(s*255 == screen+50))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 2)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 5)
        self.assertFalse(t)
        self.assertEqual(r, 11)
        self.assertTrue(np.all(s*255 == screen+50))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 3)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 4)
        self.assertTrue(t)
        self.assertEqual(r, 10)
        self.assertTrue(np.all(s*255 == screen+49))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 4)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 4)
        self.assertFalse(t)
        self.assertEqual(r, 9)
        self.assertTrue(np.all(s*255 == screen+48))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 5)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 3)
        self.assertTrue(t)
        self.assertEqual(r, 8)
        self.assertTrue(np.all(s*255 == screen+47))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 6)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 3)
        self.assertFalse(t)
        self.assertEqual(r, 7)
        self.assertTrue(np.all(s*255 == screen+46))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 7)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 2)
        self.assertTrue(t)
        self.assertEqual(r, 6)
        self.assertTrue(np.all(s*255 == screen+45))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 8)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 2)
        self.assertFalse(t)
        self.assertEqual(r, 5)
        self.assertTrue(np.all(s*255 == screen+44))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 9)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 2)
        self.assertFalse(t)
        self.assertEqual(r, 4)
        self.assertTrue(np.all(s*255 == screen+43))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 10)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 1)
        self.assertTrue(t)
        self.assertEqual(r, 3)
        self.assertTrue(np.all(s*255 == screen+42))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 11)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 0)
        self.assertTrue(t)
        self.assertEqual(r, 2)
        self.assertTrue(np.all(s*255 == screen+41))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 12)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 5)
        self.assertFalse(t)
        self.assertEqual(r, 1)
        self.assertTrue(np.all(s*255 == screen+40))



        #[training mode test]
        # If the lives decreases during the frame-skipping, interrupt the frame-skip and return.
        # And reward is sum to reward of each skipped step.
        mock_envStep.side_effect = return_values

        args = dict(env='breakout', frameskip=4, random_starts=30)
        env = e.GameEnvironment(args)
        
        env.getState()
        mock_envStep.reset_mock()
        mock_envStep.side_effect = return_values
        env._screen.clear()

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 3)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 4)
        self.assertTrue(t)
        self.assertEqual(r, 33)
        self.assertTrue(np.all(s*255 == screen+49))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 5)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 3)
        self.assertTrue(t)
        self.assertEqual(r, 17)
        self.assertTrue(np.all(s*255 == screen+47))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 7)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 2)
        self.assertTrue(t)
        self.assertEqual(r, 13)
        self.assertTrue(np.all(s*255 == screen+45))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 10)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 1)
        self.assertTrue(t)
        self.assertEqual(r, 12)
        self.assertTrue(np.all(s*255 == screen+42))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 11)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 0)
        self.assertTrue(t)
        self.assertEqual(r, 2)
        self.assertTrue(np.all(s*255 == screen+41))

        s, r, t, info = env.step(0, True)
        self.assertEqual(mock_envStep.call_count, 15)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 5)
        self.assertFalse(t)
        self.assertEqual(r, 202)
        self.assertTrue(np.all(s*255 == screen+37))




    @patch('alewrap_py.environments.AleEnv.envStep')
    def test_02_step3(self, mock_envStep):
    
        #[non-training mode test]
        # In the non-training mode, It not returns the tarminal that True
        # even if the lives was decreace. when AleEnv.envStep() returns the terminal that True, 
        # also the GameEnv returns terminal that True.
        screen = np.zeros((210, 160, 3), dtype=np.float32)
        return_values = [[screen +50,12, False, 5],
                         [screen +49,11, False, 4],
                         [screen +48,10, False, 3],
                         [screen +47, 9, False, 2],
                         [screen +46, 8, False, 1],
                         [screen +45, 7, True , 0],
                         ]

        mock_envStep.side_effect = return_values

        args = dict(env='breakout', frameskip=1, random_starts=30)
        import alewrap_py.environments as e
        env = e.GameEnvironment(args)

        env.getState()

        mock_envStep.reset_mock()
        mock_envStep.side_effect = return_values
        env._screen.clear()

        s, r, t, info = env.step(0, False)
        self.assertEqual(mock_envStep.call_count, 1)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 5)
        self.assertFalse(t)
        self.assertEqual(r, 12)
        self.assertTrue(np.all(s*255 == screen+50))

        s, r, t, info = env.step(0, False)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 4)
        self.assertFalse(t)
        self.assertEqual(mock_envStep.call_count, 2)
        self.assertEqual(r, 11)
        self.assertTrue(np.all(s*255 == screen+50))

        s, r, t, info = env.step(0, False)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 3)
        self.assertFalse(t)
        self.assertEqual(mock_envStep.call_count, 3)
        self.assertEqual(r, 10)
        self.assertTrue(np.all(s*255 == screen+49))

        s, r, t, info = env.step(0, False)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 2)
        self.assertFalse(t)
        self.assertEqual(mock_envStep.call_count, 4)
        self.assertEqual(r, 9)
        self.assertTrue(np.all(s*255 == screen+48))

        s, r, t, info = env.step(0, False)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 1)
        self.assertFalse(t)
        self.assertEqual(mock_envStep.call_count, 5)
        self.assertEqual(r, 8)
        self.assertTrue(np.all(s*255 == screen+47))

        s, r, t, info = env.step(0, False)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 0)
        self.assertTrue(t)
        self.assertEqual(mock_envStep.call_count, 6)
        self.assertEqual(r, 7)
        self.assertTrue(np.all(s*255 == screen+46))


        #[non-training mode test]
        # If the lives decreases during the frame-skipping, interrupt the frame-skip and return.

        mock_envStep.side_effect = return_values


        args = dict(env='breakout', frameskip=4, random_starts=30)
        env = e.GameEnvironment(args)

        env.getState()

        mock_envStep.reset_mock()
        mock_envStep.side_effect = return_values
        env._screen.clear()


        s, r, t, info = env.step(0, False)
        self.assertEqual(mock_envStep.call_count, 4)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 2)
        self.assertFalse(t)
        self.assertEqual(r, 42)
        self.assertTrue(np.all(s*255 == screen+48))

        s, r, t, info = env.step(0, False)
        self.assertEqual(mock_envStep.call_count, 6)
        self.assertIn('ale.lives', info)
        self.assertEqual(info['ale.lives'], 0)
        self.assertTrue(t)
        self.assertEqual(r, 15)
        self.assertTrue(np.all(s*255 == screen+46))

    @patch('alewrap_py.environments.AleEnv.envStep')
    def test_03_newGame1(self, mock_envStep):

        other_screen = np.ones((210, 160, 3)).astype(np.float32)
        before_screen = np.array([3,4,5] * 210 * 160).reshape((210, 160, 3)).astype(np.float32)
        final_screen = np.array([1,2,3] * 210 * 160).reshape((210, 160, 3)).astype(np.float32)

        envStep_returns = [[other_screen,  0, False, 5],
                           [other_screen,  0, False, 4],
                           [other_screen,  0, False, 3],
                           [other_screen,  0, False, 2],
                           [other_screen,  0, False, 1],
                           [before_screen, 0, True , 0],
                           [final_screen,  0, False, 5]]
        random_returns = [0,2,3,1,1,0]

        with patch.object(get_random(), 'random') as mock_random:

            # If you call the newGame() when the game over condition is not reached,
            # it automatically moves with action that randomly until the game is over
            # and ALEInterface.resetGame() will be called during AleEnv.envStep
            # after the game is over.
            # Then call AleEnv.envStep again with null action to get the first screen image in the new game
            # The newGame() returns it without maximization between pre-screen image
            # because the GameScreen.frameBuffer will cleared before get the first screen image.
            mock_envStep.side_effect = envStep_returns

            args = dict(env='breakout', frameskip=1, random_starts=30)
            import alewrap_py.environments as e
            env = e.GameEnvironment(args)

            env.getState()
            actions = env.getActions()

            mock_envStep.reset_mock()
            mock_envStep.side_effect = envStep_returns
            mock_random.side_effect = random_returns

            s, r, t, info = env.newGame()

            # these are calls with random actions during the _randomSteps()
            expected = [call(actions[ret]) for ret in random_returns]
            # this is null action call after the _randomSteps()
            expected += [call(0)]

            self.assertEqual(mock_envStep.call_count, 7)
            self.assertEqual(mock_envStep.mock_calls, expected)
            self.assertTrue(np.all(env._screen.frameBuffer[0]*255==final_screen))
            self.assertTrue(np.all(env._screen.frameBuffer[1]==np.zeros((210,160,3))))
            self.assertTrue(np.all(s*255 == final_screen))
            self.assertEqual(r, 0)
            self.assertFalse(t)
            self.assertEqual(info, {'ale.lives':5})

    def test_03_newGame2(self):
        # In training mode, if you call the newGame after received
        # the terminal condition from step(), it calls only the envStep()
        # with null action and returns it return values.
        import alewrap_py.environments as e
        args = dict(env='breakout', frameskip=1, random_starts=30)
        env = e.GameEnvironment(args)
        random = get_random()
        actions = env.getActions()
        s, r, t, info = env.getState()
        
        while not t:
            act = actions[random.random(4)]
            s, r, t, info = env.step(act, True)

        @patch.object(env.env.ale, 'getGameScreenRGB')
        @patch.object(env.env.ale, 'act')
        @patch.object(env.env.ale, 'resetGame', wraps=env.env.ale.resetGame)
        @patch.object(env.env, 'envStep', wraps=env.env.envStep)
        def test_func(mock_envStep, mock_resetGame, mock_act, mock_getGameScreenRGB):
            mock_act.return_value = 0
            screen = np.ones((210,160,3), dtype=np.uint8)
            mock_getGameScreenRGB.return_value = screen

            s, r, t, info = env.newGame()

            self.assertEqual(info['ale.lives'], 5)
            mock_envStep.assert_called_once_with(0)
            mock_resetGame.assert_not_called()
            mock_getGameScreenRGB.assert_called_once_with()
            mock_act.assert_called_once_with(0)
            self.assertTrue(np.all(s*255 == screen.astype(np.float32)))


    def test_03_newGame3(self):
        # In non-training mode, if you call the newGame()
        # after received the terminal condition from step(),
        # it will call the ALEInterface.resetGame() to reset to new game,
        # and get the first screen image of new game.
        # At this time,  it will not called ALEInterface.act()
        screen = np.ones((210, 160, 3)).astype(np.uint8)
        getGameScreenRGB_returns = [screen + i for i in range(50)]

        import alewrap_py.environments as e
        args = dict(env='breakout', frameskip=1, random_starts=30)
        env = e.GameEnvironment(args)
        random = get_random()
        actions = env.getActions()
        s, r, t, info = env.getState()
        
        while not t:
            act = actions[random.random(4)]
            s, r, t, info = env.step(act, False)

        @patch.object(env.env, 'envStep', wraps=env.env.envStep)
        @patch.object(env.env.ale, 'getGameScreenRGB', side_effect=getGameScreenRGB_returns)
        @patch.object(env.env.ale, 'act', wraps=env.env.ale.act)
        @patch.object(env.env.ale, 'resetGame', wraps=env.env.ale.resetGame)
        def test_func(mock_resetGame, mock_act, mock_envStep, mock_getGameScreenRGB):

            s, r, t, info = env.newGame()

            self.assertEqual(info['ale.lives'], 5)
            mock_act.assert_not_called()
            mock_resetGame.assert_called_once_with()
            self.assertTrue(np.all(s*255 == screen.astype(np.float32)))
            
        test_func()


    def test_04_nextRandomGame1(self):
        #
        # training mode
        #
        screen = np.zeros((210, 160, 3)).astype(np.uint8)
        getGameScreenRGB_returns = [screen + i for i in range(1, 50)]
        random_returns = [10, 20, 1, 2, 30, 15]

        import alewrap_py.environments as e
        args = dict(env='breakout', frameskip=1, random_starts=30)
        env = e.GameEnvironment(args)
        random = get_random()
        actions = env.getActions()
        s, r, t, info = env.getState()
        
        args = dict(env='breakout', frameskip=1, random_starts=30)
        import alewrap_py.environments as e
        env = e.GameEnvironment(args)
        env.getState()

        for i, random_starts in enumerate(random_returns):
            with self.subTest(i=i):
                t = False
                while not t:
                    act = actions[random.random(4)]
                    s, r, t, info = env.step(act, True)

                @patch.object(get_random(), 'random', return_value=random_starts)
                @patch.object(env.env, 'envStep', wraps=env.env.envStep)
                @patch.object(env.env.ale, 'getGameScreenRGB', side_effect=getGameScreenRGB_returns)
                @patch.object(env.env.ale, 'act', wraps=env.env.ale.act)
                @patch.object(env.env.ale, 'resetGame', wraps=env.env.ale.resetGame)
                def test_func(mock_resetGame, mock_act, mock_envStep, mock_getGameScreenRGB, mock_random):


                    s, r, t, info = env.nextRandomGame()

                    expected_lives = 5 if i > 0 and i % 4 == 0 else 5 - (i + (5 * int(i / 5.0))) % 5 - 1
                    self.assertEqual(info['ale.lives'], expected_lives)

                    # Both the AleEnv.envStep() and ALEInterface.mock_getGameScreenRGB are
                    # called random_starts + 2 times.
                    # Because it is called 1 time during newGame(), next called
                    # random_starts times during skipping for random starts and
                    # finally 1 more time called after skipping for get the first screen image.
                    self.assertEqual(mock_envStep.call_count, random_starts+2)
                    self.assertEqual(mock_getGameScreenRGB.call_count, random_starts+2)

                    if i > 0 and i % 4 == 0:
                        self.assertEqual(mock_act.call_count, random_starts+1)
                        mock_resetGame.assert_called_once_with()
                    else:
                        # In the training mode, if the step() return the terminal condition,
                        # the game may still continuing, because step() will return
                        # the terminal condition when the lives is decreased every time.
                        self.assertEqual(mock_act.call_count, random_starts+2)
                        mock_resetGame.assert_not_called()

                    before_screen = screen.astype(np.float32)+random_starts+1
                    final_screen = screen.astype(np.float32)+random_starts+2
                    idx = abs(env._screen.lastIndex-1)
                    self.assertTrue(np.all(env._screen.frameBuffer[abs(idx-1)]*255==before_screen))
                    self.assertTrue(np.all(env._screen.frameBuffer[idx]*255==final_screen))
                    self.assertTrue(np.all(s*255 == final_screen))
                    self.assertFalse(t)

                test_func()

    def test_04_nextRandomGame2(self):
        #
        # non-training mode
        #
        screen = np.zeros((210, 160, 3)).astype(np.uint8)
        getGameScreenRGB_returns = [screen + i for i in range(1, 50)]
        random_returns = [10, 20, 1, 2]

        import alewrap_py.environments as e
        args = dict(env='breakout', frameskip=1, random_starts=30)
        env = e.GameEnvironment(args)
        random = get_random()
        actions = env.getActions()
        s, r, t, info = env.getState()
        
        args = dict(env='breakout', frameskip=1, random_starts=30)
        import alewrap_py.environments as e
        env = e.GameEnvironment(args)
        env.getState()

        for i, random_starts in enumerate(random_returns):
            with self.subTest(i=i):
                t = False
                while not t:
                    act = actions[random.random(4)]
                    s, r, t, info = env.step(act, False)

                @patch.object(get_random(), 'random', return_value=random_starts)
                @patch.object(env.env, 'envStep', wraps=env.env.envStep)
                @patch.object(env.env.ale, 'getGameScreenRGB', side_effect=getGameScreenRGB_returns)
                @patch.object(env.env.ale, 'act', wraps=env.env.ale.act)
                @patch.object(env.env.ale, 'resetGame', wraps=env.env.ale.resetGame)
                def test_func(mock_resetGame, mock_act, mock_envStep, mock_getGameScreenRGB, mock_random):

                    s, r, t, info = env.nextRandomGame()

                    self.assertEqual(info['ale.lives'], 5)
                    
                    # Both the AleEnv.envStep() and ALEInterface.mock_getGameScreenRGB are
                    # called random_starts + 2 times.
                    # Because it is called 1 time during newGame(), next called
                    # random_starts times during skipping for random starts and
                    # finally 1 more time called after skipping for get the first screen image.
                    self.assertEqual(mock_envStep.call_count, random_starts+2)
                    self.assertEqual(mock_getGameScreenRGB.call_count, random_starts+2)
                    
                    # The same is true for ALEInterface.act(),
                    # but the AleEnv.envStep() will called resetGame() once
                    # instead act() because game was terminated.
                    # Therefore act() is called random_starts + 1 time.
                    self.assertEqual(mock_act.call_count, random_starts+1)
                    mock_resetGame.assert_called_once_with()

                    before_screen = screen.astype(np.float32)+random_starts+1
                    final_screen = screen.astype(np.float32)+random_starts+2
                    idx = abs(env._screen.lastIndex-1)
                    self.assertTrue(np.all(env._screen.frameBuffer[abs(idx-1)]*255==before_screen))
                    self.assertTrue(np.all(env._screen.frameBuffer[idx]*255==final_screen))
                    self.assertTrue(np.all(s*255 == final_screen))
                    self.assertFalse(t)
                    self.assertEqual(info['ale.lives'], 5)

                test_func()

    def test_05_maxmization_option(self):

        expected_s = np.zeros((210,160,3), dtype=np.uint8)
        side_effect = [(expected_s + i, 1, False, 5) for i in range(10, -1, -1)]

        @patch('alewrap_py.environments.AleEnv.envStep')
        def test(mock_envStep):
            for training in [1, 0]:
                mock_envStep.reset_mock()
                mock_envStep.side_effect=side_effect

                args = dict(env='breakout', actrep=1, random_starts=30, maximization='none')

                import alewrap_py.environments as e
                import alewrap_py.game_screen as s
                game_env = e.GameEnvironment(args)
                self.assertIsInstance(game_env._screen, s.NopGameScreen)
                game_actions = game_env.getActions()
                game_env.getState()

                for i, expected in enumerate(side_effect[2:]):
                    s, r, t, info = game_env.step(0, training)
                    self.assertEqual(mock_envStep.call_count, i+3)
                    assert_equal(np.uint8(s * 255), expected[0])
        test()


        @patch('alewrap_py.environments.AleEnv.envStep')
        def test(mock_envStep):
            for training in [1, 0]:
                mock_envStep.reset_mock()
                mock_envStep.side_effect=side_effect

                args = dict(env='breakout', actrep=1, random_starts=30, maximization='agent')

                import alewrap_py.environments as e
                import alewrap_py.game_screen as s
                game_env = e.GameEnvironment(args)
                self.assertIsInstance(game_env._screen, s.NopGameScreen)
                game_actions = game_env.getActions()
                game_env.getState()

                for i, expected in enumerate(side_effect[2:]):
                    s, r, t, info = game_env.step(0, training)
                    self.assertEqual(mock_envStep.call_count, i+3)
                    assert_equal(np.uint8(s * 255), expected[0])
        test()


        @patch('alewrap_py.environments.AleEnv.envStep')
        def test(mock_envStep):
            for training in [1, 0]:
                mock_envStep.reset_mock()
                mock_envStep.side_effect=side_effect

                args = dict(env='breakout', actrep=1, random_starts=30, maximization='env')

                import alewrap_py.environments as e
                import alewrap_py.game_screen as s
                game_env = e.GameEnvironment(args)
                self.assertIsInstance(game_env._screen, s.GameScreen)
                game_actions = game_env.getActions()
                game_env.getState()

                for i, expected in enumerate(side_effect[2:]):
                    s, r, t, info = game_env.step(0, training)
                    self.assertEqual(mock_envStep.call_count, i+3)
                    assert_equal(s, game_env._screen.frameBuffer.max(0), verbose=2, use_float_equal=True)
        test()


    def test_99_random_2000step(self):

        from scale import ScaleCV2
        scale = ScaleCV2(80,80, Namespace(inter='LINEAR'))
        
        np.set_printoptions(threshold=np.inf)

        args = dict(env='breakout', actrep=4, random_starts=30)

        from alewrap_py.environments import GameEnvironment
        game_env = GameEnvironment(args)
        game_actions = game_env.getActions()
        
        dqn3_states = load_lua('./alewrap_dump/random_1000step_training_state.dat').numpy()
        dqn3_rewards = load_lua('./alewrap_dump/random_1000step_training_reward.dat').numpy()
        dqn3_terminals = load_lua('./alewrap_dump/random_1000step_training_terminal.dat').numpy()
        dqn3_actions = load_lua('./alewrap_dump/random_1000step_training_action.dat').numpy()
        
        dqn3_states = np.transpose(dqn3_states, (0,2,3,1))
        
        random = get_random('pytorch', 1)
        random.manualSeed(1)
        s, r, t, info = game_env.getState()

        for i in range(1000):
            #with self.subTest(i=i):

            #if not np.all(dqn3_states[i] == s):
            #    scale.show(s)
            #    scale.show(dqn3_states[i])
            #    print('-------------{}----------------'.format(i))
            #    print(s)
            #    print('-----------------------------')
            #    print(dqn3_states[i])
            #    print('-----------------------------')

            self.assertTrue(np.all(dqn3_states[i] == s))
            self.assertEqual(dqn3_rewards[i], r)
            self.assertEqual(dqn3_terminals[i], t)

            act = game_actions[random.random(0, 4)]

            self.assertEqual(dqn3_actions[i], act)

            if t:
                s, r, t, info = game_env.nextRandomGame()
            else:
                s, r, t, info = game_env.step(act, True)
            sys.stdout.flush()
            sys.stderr.flush()

        dqn3_states = load_lua('./alewrap_dump/random_1000step_not_training_state.dat').numpy()
        dqn3_rewards = load_lua('./alewrap_dump/random_1000step_not_training_reward.dat').numpy()
        dqn3_terminals = load_lua('./alewrap_dump/random_1000step_not_training_terminal.dat').numpy()
        dqn3_actions = load_lua('./alewrap_dump/random_1000step_not_training_action.dat').numpy()
        
        dqn3_states = np.transpose(dqn3_states, (0,2,3,1))
        
        s, r, t, info = game_env.nextRandomGame()
        for i in range(1000):
            with self.subTest(i=i):
                self.assertTrue(np.all(dqn3_states[i] == s))
                self.assertEqual(dqn3_rewards[i], r)
                self.assertEqual(dqn3_terminals[i], t)

                act = game_actions[random.random(0, 4)]

                self.assertEqual(dqn3_actions[i], act)

                if t:
                    s, r, t, info = game_env.newGame()
                else:
                    s, r, t, info = game_env.step(act, False)

