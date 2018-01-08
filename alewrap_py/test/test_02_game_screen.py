import sys
import unittest
import numpy as np

sys.path.append('../../')

from testutils import *

class TestGameScreen(unittest.TestCase):


    def test_01_init(self):
        from alewrap_py.game_screen import GameScreen
        
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2)
        
        gs = GameScreen(args)

        self.assertIsNone(gs.frameBuffer)
        self.assertIsNone(gs.poolBuffer)
        self.assertEqual(gs.lastIndex, 0)
        self.assertEqual(gs.bufferSize, 2)
        self.assertEqual(gs.poolType, 'max')
        
        pool_frms = dict(type = 'min', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2)
        
        # In the alewrap, the poolType accepts 'max', 'min' or 'mean', but DQN3.0 uses only 'max' eventually.
        # In the alewrap_py implements only 'max' because it aimes to simple implementation than alewrap.
        with self.assertRaisesRegex(ValueError, "got 'min', expected only 'max'."):
            gs = GameScreen(args)

        gs = GameScreen({})

        self.assertIsNone(gs.frameBuffer)
        self.assertIsNone(gs.poolBuffer)
        self.assertEqual(gs.lastIndex, 0)
        self.assertEqual(gs.bufferSize, 2)
        self.assertEqual(gs.poolType, 'max')

    def test_01_paint(self):
        
        from alewrap_py.game_screen import GameScreen
        
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2)
        
        gs = GameScreen(args)

        frame1 = np.array([[101,102,103],[111,112,113]]).astype(np.uint8)
        gs.paint(frame1)

        self.assertIsNotNone(gs.frameBuffer)
        self.assertIsNone(gs.poolBuffer)
        self.assertEqual(gs.frameBuffer.shape, (2,) + frame1.shape)
        self.assertEqual(gs.lastIndex, 1)

        # The paint method must normalize the frame before save into the frameBuffer.
        self.assertTrue(np.all(gs.frameBuffer[0] == frame1.astype(np.float32) / 255.0))
        self.assertTrue(np.all(gs.frameBuffer[1] == np.zeros(frame1.shape)))

        frame2 = np.array([[201,202,203],[211,212,213]]).astype(np.uint8)
        gs.paint(frame2)

        self.assertIsNotNone(gs.frameBuffer)
        self.assertIsNone(gs.poolBuffer)
        self.assertEqual(gs.frameBuffer.shape, (2,) + frame2.shape)
        self.assertEqual(gs.lastIndex, 2)
        self.assertTrue(np.all(gs.frameBuffer[0] == frame1.astype(np.float32) / 255.0))
        self.assertTrue(np.all(gs.frameBuffer[1] == frame2.astype(np.float32) / 255.0))

        frame3 = np.array([[1,2,3],[11,12,13]]).astype(np.uint8)
        gs.paint(frame3)

        self.assertIsNotNone(gs.frameBuffer)
        self.assertIsNone(gs.poolBuffer)
        self.assertEqual(gs.frameBuffer.shape, (2,) + frame3.shape)
        # The lastIndex is equal to bufferSize, therefore it clears to zero
        # before adding a frame and increments it to 1 after adding a frame.
        self.assertEqual(gs.lastIndex, 1)
        self.assertTrue(np.all(gs.frameBuffer[0] == frame3.astype(np.float32) / 255.0))
        self.assertTrue(np.all(gs.frameBuffer[1] == frame2.astype(np.float32) / 255.0))

    def test_02_grab(self):
        
        from alewrap_py.game_screen import GameScreen
        
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2)
        
        gs = GameScreen(args)

        frame1 = np.array([[1,102,103],[111,12,113]]).astype(np.uint8)
        gs.paint(frame1)

        frame2 = np.array([[201,2,203],[211,212,13]]).astype(np.uint8)
        gs.paint(frame2)

        frame3 = np.array([[1,202,3],[11,112,13]]).astype(np.uint8)
        gs.paint(frame3)

        obs = gs.grab()
        self.assertIsNotNone(gs.frameBuffer)
        self.assertIsNotNone(gs.poolBuffer)
        self.assertEqual(gs.poolBuffer.shape, frame3.shape)

        # The gs.frameBuffer contains to frame2 and frame3.
        # The poolBuffer contains maximun of frame2 and frame3.
        expected = np.array([[201,202,203],[211,212,13]]).astype(np.float32)
        expected /= 255
        self.assertTrue(np.all(gs.poolBuffer == expected))
        self.assertTrue(np.all(gs.poolBuffer == obs))

    def test_03_clear(self):
        
        from alewrap_py.game_screen import GameScreen
        
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2)
        
        gs = GameScreen(args)

        frame1 = np.array([[1,102,103],[111,12,113]]).astype(np.uint8)
        gs.paint(frame1)

        frame2 = np.array([[201,2,203],[211,212,13]]).astype(np.uint8)
        gs.paint(frame2)

        frame3 = np.array([[1,202,3],[11,112,13]]).astype(np.uint8)
        gs.paint(frame3)

        obs = gs.grab()

        gs.clear()
        
        self.assertIsNotNone(gs.frameBuffer)
        self.assertIsNotNone(gs.poolBuffer)
        self.assertTrue(np.all(gs.frameBuffer == np.zeros((2,)+frame1.shape)))
        self.assertEqual(gs.lastIndex, 0)
        # The poolBuffer is not clear.
        self.assertTrue(np.all(gs.poolBuffer == obs))


    def test_04_normalization_mode1_for_env(self):
    
        from alewrap_py.game_screen import GameScreen

        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env')
        
        gs = GameScreen(args)
        print(gs.screen_normalize)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), np.float32(s / 255.0))


        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='env')
        
        gs = GameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), np.float32(s / 255.0))


        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='agent')
        
        gs = GameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)

        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='none')
        
        gs = GameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)

    def test_04_normalization_mode2_for_env(self):
    
        from alewrap_py.game_screen import NopGameScreen

        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), np.float32(s / 255.0))


        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='env')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), np.float32(s / 255.0))


        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='agent')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)

        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='none')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)

    def test_05_normalization_mode1_for_agent(self):
    
        from alewrap_py.game_screen import GameScreen

        # GameScreen will normalization when it instance created in env,
        # but when it instance created in agent is not.
        # Determine of instance creation location is done using the argument 'maximization'.

        # will normalization at env and will maximization at agent.
        # this case is, GameScreen get normalized screen image.
        # because of this, GameScreen will not normalization.
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='agent')
        
        gs = GameScreen(args)

        s = np.ones((210,160,3), np.float32) / 255.0
        gs.paint(s)
        
        assert_equal(gs.grab(), s)


        # will normalization at env and will maximization at agent.
        # this case is, GameScreen get normalized screen image.
        # because of this, GameScreen doesn't normalization.
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='agent', screen_normalize='env')
        
        gs = GameScreen(args)

        s = np.ones((210,160,3), np.float32) / 255.0
        gs.paint(s)
        
        assert_equal(gs.grab(), s)


        # will normalization at trans and will maximization at agent.
        # this case is, GameScreen get unnormalized screen image.
        # because of this, GameScreen will normalization.
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='agent', screen_normalize='trans')
        
        gs = GameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)

        # will not normalization anywhere and will maximization at agent.
        # this case is, GameScreen get unnormalized screen image.
        # because of this, GameScreen will not normalization.
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='agent', screen_normalize='none')
        
        gs = GameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)

    def test_05_normalization_mode2_for_agent(self):
    
        from alewrap_py.game_screen import NopGameScreen

        # will normalization at env and will maximization at env.
        # this case is, GameScreen get normalized screen image.
        # because of this, GameScreen will not normalization.
        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='agent')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.float32) / 255.0
        gs.paint(s)
        
        assert_equal(gs.grab(), np.float32(s / 255.0))


        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='env')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.float32) / 255.0
        gs.paint(s)
        
        assert_equal(gs.grab(), np.float32(s / 255.0))


        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='trans')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)


        pool_frms = dict(type = 'max', size = 2)
        args = dict(pool_frms = pool_frms, verbose = 2, maximization='env', screen_normalize='none')
        
        gs = NopGameScreen(args)

        s = np.ones((210,160,3), np.uint8)
        gs.paint(s)
        
        assert_equal(gs.grab(), s)


