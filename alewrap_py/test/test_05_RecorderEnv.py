import sys
import os
import re
import shutil
import unittest
from unittest.mock import *

import numpy as np
from torch.utils.serialization import load_lua

if sys.path.count('../../') == 0:
    sys.path.append('../../')

from utils import get_random, Namespace
from testutils import *

class TestRecorderEnv(unittest.TestCase):

    def setUp(self):
        random = get_random('pytorch', 1)

    def test_00_init(self):
        from alewrap_py import get_env, GameEnvironment, RecorderEnv
        
        args = dict(env="breakout", actrep=4, random_starts=30, verbose=2)
        env = get_env(args)

        self.assertIsInstance(env, RecorderEnv)
        self.assertIsInstance(env.env, GameEnvironment)
        self.assertIsNotNone(env.max_episode_score)
        self.assertEqual(env.max_episode_score, 0)
        self.assertIsNotNone(env.episode_score)
        self.assertEqual(env.episode_score, 0)
        self.assertIsNotNone(env.episode_scores)
        self.assertIsInstance(env.episode_scores, list)
        self.assertEqual(len(env.episode_scores), 0)
        self.assertIsNotNone(env.frames)
        self.assertEqual(env.frame_shape, env.env.frame_shape)
        self.assertEqual(env.frames.shape, (5000,) + env.env.frame_shape)
        assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
        self.assertEqual(env.frame_index, 0)
        self.assertFalse(env.recording_mode)

    def test_01(self):
    
        # When start_recording () has not yet been called,
        # RecorderEnv does not record and write video, and collect each episode score.
        # However, it summarize a episode score from the reward at each step.
        # The episode score is get at any time by episode_score property,
        # but when you call newGame() or nextRandomGame(), the episode score is cleared to 0.
        # Other behaviors have been delegated to child env.

        mock_child_env = MagicMock()
        type(mock_child_env).frame_shape = PropertyMock(return_value=(210,160,3))
        type(mock_child_env).metadata = {'render.modes': ['human', 'rgb_array']}

        
        from alewrap_py import RecorderEnv
        env = RecorderEnv(mock_child_env,{})

        def test(training):
            expected = [np.ones((210,160,3)), 0, False, {}]
            mock_child_env.getState.return_value = expected
            
            ret = env.getState()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertFalse(env.recording_mode)


            expected[0] += 1
            expected[1] = 1
            expected[2] = True
            expected[3] = {'hoge':1}
            mock_child_env.step.return_value = expected
            
            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 1)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertEqual(env.frame_index, 0)
            self.assertFalse(env.recording_mode)


            expected[0] += 1
            expected[1] = 1
            expected[2] = True
            expected[3] = {'hoge':1}
            mock_child_env.step.return_value = expected
            
            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 2)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertEqual(env.frame_index, 0)
            self.assertFalse(env.recording_mode)


            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.newGame.return_value = expected
            
            ret = env.newGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertEqual(env.frame_index, 0)
            self.assertFalse(env.recording_mode)


            expected[0] += 1
            expected[1] = 5
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 5)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertEqual(env.frame_index, 0)
            self.assertFalse(env.recording_mode)


            expected[0] += 1
            expected[1] = 10
            expected[2] = True
            expected[3] = {'zzz':1}
            mock_child_env.step.return_value = expected

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 15)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertEqual(env.frame_index, 0)
            self.assertFalse(env.recording_mode)


            expected[0] += 1
            expected[1] = 0
            expected[2] = True
            expected[3] = {'hage':1, 'aaa':'bbb'}
            mock_child_env.nextRandomGame.return_value = expected
            
            ret = env.nextRandomGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertEqual(env.frame_index, 0)
            self.assertFalse(env.recording_mode)


            expected[0] += 1
            expected[1] = 10
            expected[2] = True
            expected[3] = {'zzz':1}
            mock_child_env.step.return_value = expected

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 10)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape))
            self.assertEqual(env.frame_index, 0)
            self.assertFalse(env.recording_mode)


        test(training=True)
        test(training=False)

    def test_02(self):
    
        # When start_recording () is called,
        # the RecorderEnv records and writes the video,
        # and collect each episode score,
        # with newGame(),
        # until stop_record() is called.

        if os.path.exists('/tmp/step_0000000001'):
            shutil.rmtree('/tmp/step_0000000001')

        mock_child_env = MagicMock()
        type(mock_child_env).frame_shape = PropertyMock(return_value=(210,160,3))
        type(mock_child_env).metadata = {'render.modes': ['human', 'rgb_array']}

        def test(training):
            from alewrap_py import RecorderEnv
            env = RecorderEnv(mock_child_env,dict(monitor_dir='/tmp',
                                                  video_freq=4,
                                                  verbose=11))
            expected = [np.ones((210,160,3)), 0, False, {}]
            mock_child_env.getState.return_value = expected
            
            ret = env.getState()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape, dtype=np.uint8), verbose=1, use_float_equal=True)
            self.assertEqual(env.frame_index, 0)


            expected[0] += 1
            expected[1] = 1
            expected[2] = True
            expected[3] = {'hoge':1}
            mock_child_env.step.return_value = expected
            
            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 1)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape, dtype=np.uint8))
            self.assertEqual(env.frame_index, 0)


            expected[0] += 1
            expected[1] = 1
            expected[2] = True
            expected[3] = {'hoge':1}
            mock_child_env.step.return_value = expected
            
            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 2)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape, dtype=np.uint8))
            self.assertEqual(env.frame_index, 0)


            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.newGame.return_value = expected
            
            ret = env.newGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape, dtype=np.uint8))
            self.assertEqual(env.frame_index, 0)


            expected[0] += 1
            expected[1] = 5
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 5)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape, dtype=np.uint8))
            self.assertEqual(env.frame_index, 0)

            #--------------------------------
            # start recording
            #--------------------------------
            env.start_recording(1)
            self.assertTrue(env.recording_mode)
            self.episode_id = 1
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            assert_equal(env.frames, np.zeros((5000,) + env.frame_shape, dtype=np.uint8))
            self.assertEqual(env.frame_index, 0)

            print(env.frames.shape)
            print(expected[0].shape)

            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}

            mock_child_env.newGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 1

            ret = env.newGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0, env.episode_scores)
            self.assertEqual(env.frame_index, 1)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 1)
            self.assertTrue(env.recording_mode)
            self.episode_id = 1


            expected[0] += 1
            expected[1] = 5
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 2

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 5)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frame_index, 2)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 2)
            self.assertEqual(env.episode_id, 1)
    

            expected[0] += 1
            expected[1] = 10
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 3

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 15)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frame_index, 3)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 3)
            self.assertEqual(env.episode_id, 1)
    

            #--------------------------------
            # episode 1 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.newGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 4

            ret = env.newGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frame_index, 1)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 4)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 2)
            # not write the video file at first episode.
            self.assertTrue(os.path.exists('/tmp/step_0000000001/episode=0000000001_score=0000000015.avi'))


            expected[0] += 1
            expected[1] = 3
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 5

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 3)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frame_index, 2)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 5)
            self.assertEqual(env.episode_id, 2)

            expected[0] += 1
            expected[1] = 4
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 6

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 7)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frame_index, 3)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 6)
            self.assertEqual(env.episode_id, 2)

            expected[0] += 1
            expected[1] = 5
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 7

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 12)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frame_index, 4)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 7)
            self.assertEqual(env.episode_id, 2)

            #--------------------------------
            # episode 2 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.newGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 8
            
            ret = env.newGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 3)
            self.assertEqual(env.frame_index, 1)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 8)
            # This episode is not get max score and not number of video_freq(4).
            self.assertFalse(os.path.exists('/tmp/step_0000000001/episode=0000000002_score=0000000012.avi'))


            expected[0] += 1
            expected[1] = 10
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 9

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 10)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertEqual(env.frame_index, 2)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 9)
            self.assertEqual(env.episode_id, 3)

            expected[0] += 1
            expected[1] = 15
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 10

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 25)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertEqual(env.frame_index, 3)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 10)
            self.assertEqual(env.episode_id, 3)

            expected[0] += 1
            expected[1] = 5
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 11

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 30)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertEqual(env.frame_index, 4)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 11)
            self.assertEqual(env.episode_id, 3)


            #--------------------------------
            # episode 3 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.newGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 12
            
            ret = env.newGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30])
            self.assertEqual(env.frame_index, 1)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 12)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 4)
            # The video about episode 3 will be written because it does got max score
            # in current recording mode even if episode_id is not number of video_freq(4).
            self.assertTrue(os.path.exists('/tmp/step_0000000001/episode=0000000003_score=0000000030.avi'))


            expected[0] += 1
            expected[1] = 1
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 13

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 1)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30])
            self.assertEqual(env.frame_index, 2)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 13)
            self.assertEqual(env.episode_id, 4)

            expected[0] += 1
            expected[1] = 2
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 14

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 3)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30])
            self.assertEqual(env.frame_index, 3)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 14)
            self.assertEqual(env.episode_id, 4)


            #--------------------------------
            # episode 4 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.newGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 15

            ret = env.newGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30, 3])
            self.assertEqual(env.frame_index, 1)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 15)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 5)

            # Since the episode_id is number of video_freq,
            # the episode 4 video will be written
            # even if episode did not get the maximum score in this recording session.
            self.assertTrue(os.path.exists('/tmp/step_0000000001/episode=0000000004_score=0000000003.avi'))


            expected[0] += 1
            expected[1] = 1
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 16

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 1)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30, 3])
            self.assertEqual(env.frame_index, 2)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            assert_equal(env.frames[env.frame_index-1], np.zeros((210,160,3), dtype=np.uint8) + 16)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 5)


            #--------------------------------
            # stop recording
            #--------------------------------
            env.stop_recording()
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            # If you calls stop_recording() when you don't get
            # terminal state of the game, the score of last episode will be not saved.
            assert_equal(env.episode_scores, [15, 12, 30, 3])
            self.assertEqual(env.frame_index, 0)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            # frames will not clear but it Okay because frame_index will be cleared to zero.
            self.assertFalse(env.recording_mode)
            self.assertEqual(env.episode_id, 0)
            self.assertFalse(len([fname for fname in os.listdir('/tmp') if re.match('^episode=0000000005.*$', fname)]))

        test(training=True)
        test(training=False)


    def test_03(self):
    
        # When start_recording () is called,
        # the RecorderEnv records and writes the video,
        # and collect each episode score,
        # with nextRandomGame(),
        # until stop_record() is called.

        if os.path.exists('/tmp/step_0000000002'):
            shutil.rmtree('/tmp/step_0000000002')

        mock_child_env = MagicMock()
        type(mock_child_env).frame_shape = PropertyMock(return_value=(210,160,3))
        type(mock_child_env).metadata = {'render.modes': ['human', 'rgb_array']}

        
        from alewrap_py import RecorderEnv
        env = RecorderEnv(mock_child_env,dict(monitor_dir='/tmp',
                                              video_freq=4,
                                              verbose=11))

        def test(training):
            expected = [np.ones((210,160,3)), 0, False, {}]
            mock_child_env.getState.return_value = expected
            
            ret = env.getState()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)


            expected[0] += 1
            expected[1] = 1
            expected[2] = True
            expected[3] = {'hoge':1}
            mock_child_env.step.return_value = expected
            
            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 1)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)


            expected[0] += 1
            expected[1] = 1
            expected[2] = True
            expected[3] = {'hoge':1}
            mock_child_env.step.return_value = expected
            
            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 2)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)


            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.nextRandomGame.return_value = expected
            
            ret = env.nextRandomGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)


            expected[0] += 1
            expected[1] = 5
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 5)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)

            #--------------------------------
            # start recording
            #--------------------------------
            env.start_recording(2)
            self.assertTrue(env.recording_mode)
            self.episode_id = 1
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)

            print(env.frames.shape)
            print(expected[0].shape)

            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}

            mock_child_env.nextRandomGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 1

            ret = env.nextRandomGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0, env.episode_scores)
            self.assertEqual(env.frames.shape, (1,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 1)
            self.assertTrue(env.recording_mode)
            self.episode_id = 1


            expected[0] += 1
            expected[1] = 5
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 2

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 5)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (2,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 2)
            self.assertEqual(env.episode_id, 1)
    

            expected[0] += 1
            expected[1] = 10
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 3

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 15)
            self.assertEqual(env.max_episode_score, 0)
            self.assertEqual(len(env.episode_scores), 0)
            self.assertEqual(env.frames.shape, (3,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 3)
            self.assertEqual(env.episode_id, 1)
    

            #--------------------------------
            # episode 1 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.nextRandomGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 4

            ret = env.nextRandomGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frames.shape, (1,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 4)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 2)
            # not write the video file at first episode.
            self.assertTrue(os.path.exists('/tmp/step_0000000002/episode=0000000001_score=0000000015.avi'))


            expected[0] += 1
            expected[1] = 3
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 5

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 3)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frames.shape, (2,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 5)
            self.assertEqual(env.episode_id, 2)

            expected[0] += 1
            expected[1] = 4
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 6

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 7)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frames.shape, (3,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 6)
            self.assertEqual(env.episode_id, 2)

            expected[0] += 1
            expected[1] = 5
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 7

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 12)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15])
            self.assertEqual(env.frames.shape, (4,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 7)
            self.assertEqual(env.episode_id, 2)

            #--------------------------------
            # episode 2 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.nextRandomGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 8
            
            ret = env.nextRandomGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertEqual(env.frames.shape, (1,) + env.frame_shape)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 3)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 8)
            # This episode is not get max score and not number of video_freq(4).
            self.assertFalse(os.path.exists('/tmp/step_0000000002/episode=0000000002_score=0000000012.avi'))


            expected[0] += 1
            expected[1] = 10
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 9

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 10)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertEqual(env.frames.shape, (2,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 9)
            self.assertEqual(env.episode_id, 3)

            expected[0] += 1
            expected[1] = 15
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 10

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 25)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertEqual(env.frames.shape, (3,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 10)
            self.assertEqual(env.episode_id, 3)

            expected[0] += 1
            expected[1] = 5
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 11

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 30)
            self.assertEqual(env.max_episode_score, 15)
            assert_equal(env.episode_scores, [15, 12])
            self.assertEqual(env.frames.shape, (4,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 11)
            self.assertEqual(env.episode_id, 3)


            #--------------------------------
            # episode 3 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.nextRandomGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 12
            
            ret = env.nextRandomGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30])
            self.assertEqual(env.frames.shape, (1,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 12)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 4)
            # The video about episode 3 will be written because it does got max score
            # in current recording mode even if episode_id is not number of video_freq(4).
            self.assertTrue(os.path.exists('/tmp/step_0000000002/episode=0000000003_score=0000000030.avi'))


            expected[0] += 1
            expected[1] = 1
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 13

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 1)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30])
            self.assertEqual(env.frames.shape, (2,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 13)
            self.assertEqual(env.episode_id, 4)

            expected[0] += 1
            expected[1] = 2
            expected[2] = True
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 14

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 3)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30])
            self.assertEqual(env.frames.shape, (3,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 14)
            self.assertEqual(env.episode_id, 4)


            #--------------------------------
            # episode 4 end
            #--------------------------------
            expected[0] += 1
            expected[1] = 0
            expected[2] = False
            expected[3] = {'hage':1}
            mock_child_env.nextRandomGame.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 15

            ret = env.nextRandomGame()
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30, 3])
            self.assertEqual(env.frames.shape, (1,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 15)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 5)

            # Since the episode_id is number of video_freq,
            # the episode 4 video will be written
            # even if episode did not get the maximum score in this recording session.
            self.assertTrue(os.path.exists('/tmp/step_0000000002/episode=0000000004_score=0000000003.avi'))


            expected[0] += 1
            expected[1] = 1
            expected[2] = False
            expected[3] = {'bbb':1}
            mock_child_env.step.return_value = expected
            mock_child_env.render.return_value = np.zeros((210,160,3), dtype=np.uint8) + 16

            ret = env.step(1, training=training)
            assert_equal(ret, expected)
            self.assertEqual(env.episode_score, 1)
            self.assertEqual(env.max_episode_score, 30)
            assert_equal(env.episode_scores, [15, 12, 30, 3])
            self.assertEqual(env.frames.shape, (2,) + env.frame_shape)
            assert_equal(env.frames[-1], np.zeros((210,160,3), dtype=np.uint8) + 16)
            self.assertTrue(env.recording_mode)
            self.assertEqual(env.episode_id, 5)


            #--------------------------------
            # stop recording
            #--------------------------------
            env.stop_recording()
            self.assertEqual(env.episode_score, 0)
            self.assertEqual(env.max_episode_score, 0)
            # If you calls stop_recording() when you don't get
            # terminal state of the game, the score of last episode will be not saved.
            assert_equal(env.episode_scores, [15, 12, 30, 3])
            self.assertEqual(env.frames.shape, (5000,) + env.frame_shape)
            self.assertFalse(env.recording_mode)
            self.assertEqual(env.episode_id, 0)
            self.assertFalse(len([fname for fname in os.listdir('/tmp') if re.match('^episode=0000000005.*$', fname)]))

        test(training=True)
        test(training=False)

    def test_04(self):
        args = dict(env="breakout", actrep=4, random_starts=30, verbose=2)
        self._total_test(args)

    def test_05(self):
        args = dict(env="Breakout-v0", actrep=4, random_starts=30, verbose=2)
        self._total_test(args)

    def test_06(self):
        sys.argv += ['--backend', 'pytorch_legacy',
                     '--env', 'Breakout-v0',
                     '--logdir','/tmp/',
                     '--write_frame_to_png',
                     '--file_name', './learned params/Breakout-v0_pytorch_network_step0050000000.dat',
                     '--steps', '1000']
        sys.path.append('../../dqn')
        from config import get_opt
        from initenv import setup
        from time import time
        from testutils.method_wrappers import set_measer, set_tracer, tracers_summary
        
        opt = get_opt()
        game_env, agent, actions, opt = setup(opt)

        #game_env.env.env.ale = set_tracer(set_measer(game_env.env.env.ale))
        #game_env.env.env = set_tracer(set_measer(game_env.env.env))
        #game_env.env = set_tracer(set_measer(game_env.env))

        from dqn.train_agent import test_main
        from time import time
        
        np.random.seed(1)
        st = time()
        test_main(game_env, agent, actions, opt)
        print('{:0.4f}'.format(time() - st))

        #tracers_summary()

    def _total_test(self, args):
        from alewrap_py import get_env
        random = get_random()

        env = get_env(args)
        actions = env.getActions()

        s, r, t, info = env.getState()

        for step in range(3000):
            action = random.random(0,4)
            s, r, t, info = env.step(actions[action], True)
            
            if t:
                s, r, t, info = env.nextRandomGame()

        env.start_recording(3000)

        s, r, t, info = env.newGame()

        for step in range(3000):
            action = random.random(0,4)
            s, r, t, info = env.step(actions[action], False)
            
            if t:
                s, r, t, info = env.newGame()

        env.stop_recording()

    def test_98(self):
        sys.argv += ['--backend', 'pytorch_legacy',
                     '--env', 'breakout',
                     '--logdir','/tmp',
                     '--steps', '1000',
                     '--step_train_mode', '0',
                     '--maximization', 'non',
                     '--actrep', '(2,5)',
                     '--random_type', 'numpy',
                     #'--render'
                     ]
        sys.path.append('../../dqn')
        from config import get_opt
        from initenv import setup
        from time import time
        from testutils.method_wrappers import set_measer, set_tracer, tracers_summary
        from alewrap_py import GameEnvironment, AleEnv, RecorderEnv, ALEInterface
        
        opt = get_opt()
        game_env, agent, actions, opt = setup(opt)

        self.assertIsInstance(game_env, RecorderEnv)
        self.assertIsInstance(game_env.env, GameEnvironment)
        self.assertIsInstance(game_env.env.env, AleEnv)
        self.assertIsInstance(game_env.env.env.ale, ALEInterface)
        game_env.env.env.ale = set_tracer(set_measer(game_env.env.env.ale))
        game_env.env.env = set_tracer(set_measer(game_env.env.env))
        game_env.env = set_tracer(set_measer(game_env.env))

        from dqn.train_agent import train_main
        from time import time
        
        np.random.seed(1)
        st = time()
        train_main(game_env, agent, actions, opt)
        print('{:0.4f}'.format(time() - st))
        
        tracers_summary()


    def test_99(self):
        sys.argv += ['--backend', 'pytorch_legacy',
                     '--env', 'Breakout-v0',
                     '--logdir','/tmp',
                     '--steps', '1000']
        sys.path.append('../../dqn')
        from config import get_opt
        from initenv import setup
        from time import time
        from testutils.method_wrappers import set_measer, set_tracer, tracers_summary
        
        opt = get_opt()
        game_env, agent, actions, opt = setup(opt)

        #game_env.env.env.ale = set_tracer(set_measer(game_env.env.env.ale))
        #game_env.env.env = set_tracer(set_measer(game_env.env.env))
        #game_env.env = set_tracer(set_measer(game_env.env))

        from dqn.train_agent import train_main
        from time import time
        
        np.random.seed(1)
        st = time()
        train_main(game_env, agent, actions, opt)
        print('{:0.4f}'.format(time() - st))

        #tracers_summary()


    def test_XX(self):
        sys.argv += ['--backend', 'pytorch_legacy',
                     '--env', 'breakout',
                     '--logdir','/tmp/',
                     '--test_episodes', '5',
                     '--maximization', 'agent',
                     '--write_frame_to_png',
                     '--file_name', './learned params/breakout_pytorch_legacy_network_step0050000000.dat',
                     '--steps', '1000']
        sys.path.append('../../dqn')
        from config import get_opt
        from initenv import setup
        from time import time
        from testutils.method_wrappers import set_measer, set_tracer, tracers_summary
        import cv2
        
        opt = get_opt()

        game_env, agent, actions, opt = setup(opt)

        org_forward = agent.preproc.forward
        
        out_dir = opt.log_dir + '/png'
        os.makedirs(out_dir)
        frame_no = 1
        def forward(x):
            nonlocal frame_no
            frame = org_forward(x)
            filename = '{}/frame_{:05d}.png'.format(out_dir, frame_no)
            cv2.imwrite(filename, np.uint8(frame.reshape(84,84,1) * 255))
            frame_no += 1
            return frame
        @patch.object(agent.preproc, 'forward', wraps=forward)
        def test(mock):
            from dqn.train_agent import test_main
            from time import time
            
            np.random.seed(1)
            st = time()
            test_main(game_env, agent, actions, opt)
            print('{:0.4f}'.format(time() - st))

        test()
