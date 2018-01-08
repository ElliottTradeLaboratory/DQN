import sys
import unittest
from unittest.mock import *
import numpy as np
import torch
from torch.utils.serialization import load_lua

sys.path.append('../../')
sys.path.append('../')

from utils import get_random, Namespace
from transition_table import TransitionTable
from testutils import *

class TestTransitionTable(unittest.TestCase):

    def setUp(self):
        np.set_printoptions(threshold=np.inf)
        get_random('torch', 1)

    def test_00_init(self):

        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=100000, bufferSize=512, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        self.assertEqual(transitions.insertIndex, 0)
        self.assertEqual(transitions.numEntries, 0)
        self.assertEqual(transitions.buf_ind, -1)

        self.assertTrue(hasattr(transitions, 's'))
        self.assertEqual(len(transitions.s), 100000)
        self.assertEqual(transitions.s.dtype, np.uint8)
        self.assertTrue(np.all(transitions.s.shape == (100000, 84, 84)))
        self.assertTrue(np.all(transitions.s == np.zeros((100000, 84, 84))))

        self.assertTrue(hasattr(transitions, 'a'))
        self.assertEqual(transitions.a.dtype, np.uint8)
        self.assertEqual(len(transitions.a), 100000)
        self.assertTrue(np.all(transitions.a.shape == (100000,)), transitions.a.shape)
        self.assertTrue(np.all(transitions.a == np.zeros((100000,))))

        self.assertTrue(hasattr(transitions, 'r'))
        self.assertEqual(transitions.r.dtype, np.int)
        self.assertEqual(len(transitions.r), 100000)
        self.assertTrue(np.all(transitions.r.shape == (100000,)))
        self.assertTrue(np.all(transitions.r == np.zeros((100000,))))

        self.assertTrue(hasattr(transitions, 't'))
        self.assertEqual(len(transitions.t), 100000)
        self.assertEqual(transitions.t.dtype, np.uint8)
        self.assertTrue(np.all(transitions.t.shape == (100000,)))
        self.assertTrue(np.all(transitions.t == np.zeros((100000,))))

        self.assertTrue(hasattr(transitions, 'buf_s'))
        self.assertEqual(len(transitions.buf_s), 512,)
        self.assertEqual(transitions.buf_s.dtype, np.float32)
        self.assertTrue(np.all(transitions.buf_s.shape == (512, 4, 84, 84)))
        self.assertTrue(np.all(transitions.buf_s == np.zeros((512, 4, 84, 84))))

        self.assertTrue(hasattr(transitions, 'buf_a'))
        self.assertEqual(transitions.buf_a.dtype, np.uint8)
        self.assertEqual(len(transitions.buf_a), 512,)
        self.assertTrue(np.all(transitions.buf_a.shape == (512,)))
        self.assertTrue(np.all(transitions.buf_a == np.zeros((512,))))

        self.assertTrue(hasattr(transitions, 'buf_r'))
        self.assertEqual(transitions.buf_r.dtype, np.int)
        self.assertEqual(len(transitions.buf_r), 512,)
        self.assertTrue(np.all(transitions.buf_r.shape == (512,)))
        self.assertTrue(np.all(transitions.buf_r == np.zeros((512,))))

        self.assertTrue(hasattr(transitions, 'buf_term'))
        self.assertEqual(len(transitions.buf_term), 512,)
        self.assertEqual(transitions.buf_term.dtype, np.uint8)
        self.assertTrue(np.all(transitions.buf_term.shape == (512,)))
        self.assertTrue(np.all(transitions.buf_term == np.zeros((512,))))

        self.assertTrue(hasattr(transitions, 'buf_s2'))
        self.assertEqual(len(transitions.buf_s2), 512,)
        self.assertEqual(transitions.buf_s2.dtype, np.float32)
        self.assertTrue(np.all(transitions.buf_s2.shape == (512, 4, 84, 84)))
        self.assertTrue(np.all(transitions.buf_s2 == np.zeros((512, 4, 84, 84))))

        self.assertTrue(hasattr(transitions, 'recent_s'))
        self.assertEqual(len(transitions.recent_s), 4)
        self.assertEqual(transitions.recent_s.dtype, np.float32)
        self.assertTrue(np.all(transitions.recent_s.shape == (4, 84, 84)))
        self.assertTrue(np.all(transitions.recent_s == np.zeros((4, 84, 84))))

        self.assertTrue(hasattr(transitions, 'recent_t'))
        self.assertEqual(len(transitions.recent_t), 4)
        self.assertEqual(transitions.recent_t.dtype, np.uint8)
        self.assertTrue(np.all(transitions.recent_t.shape == (4,)))
        self.assertTrue(np.all(transitions.recent_t == np.zeros((4,))))


    def test_01_recent_state(self):

        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=100000, bufferSize=512, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        zero = np.zeros((84, 84))
        
        states = np.array([zero+1, zero+2, zero+3, zero+4, zero+5]).astype(np.float32) / 255.0
        terminals = [True, False, True, True, False]
        expected_s = [np.array([zero, zero, zero, zero+1]).astype(np.uint8),
                      np.array([zero, zero, zero+1, zero+2]).astype(np.uint8),
                      np.array([zero, zero+1, zero+2, zero+3]).astype(np.uint8),
                      np.array([zero+1, zero+2, zero+3, zero+4]).astype(np.uint8),
                      np.array([zero+2, zero+3, zero+4, zero+5]).astype(np.uint8)]
        expected_t = [np.array([0, 0, 0, 1]).astype(np.uint8),
                      np.array([0, 0, 1, 0]).astype(np.uint8),
                      np.array([0, 1, 0, 1]).astype(np.uint8),
                      np.array([1, 0, 1, 1]).astype(np.uint8),
                      np.array([0, 1, 1, 0]).astype(np.uint8),
                     ]
        out_s =      [np.array([zero, zero, zero, zero+1]).astype(np.uint8),
                      np.array([zero, zero, zero, zero+2]).astype(np.uint8),
                      np.array([zero, zero, zero+2, zero+3]).astype(np.uint8),
                      np.array([zero, zero, zero, zero+4]).astype(np.uint8),
                      np.array([zero, zero, zero, zero+5]).astype(np.uint8)]

        for i, state in enumerate(zip(states, terminals, expected_s, expected_t, out_s)):
            print('====={}====='.format(i))
            transitions.add_recent_state(state[0], state[1])
            assert_equal(transitions.recent_s, state[2], use_float_equal=True, verbose=0)
            self.assertTrue(np.all(transitions.recent_t == state[3]), transitions.recent_t)

            cur_state = transitions.get_recent()
            assert_equal(cur_state*255.0, state[4], use_float_equal=True, verbose=0)
    
    def test_02_add_1(self):
    
        zero_s = np.zeros((84, 84)).astype(np.uint8)

        tables = [
            [zero_s+100, 0,  200, 0],
            [zero_s+101, 1,  201, 0],
            [zero_s+102, 2,  202, 0],
            [zero_s+103, 3,  203, 0],
            [zero_s+104, 4,  204, 0],
            [zero_s+105, 5,  205, 0],
            [zero_s+106, 6,  206, 1],
            [zero_s+107, 7,  207, 0],
            [zero_s+108, 8,  208, 0],
            [zero_s+109, 9,  209, 0],
            [zero_s+110, 10, 210, 0],
            [zero_s+111, 11, 211, 1],
            [zero_s+112, 12, 212, 0],
            [zero_s+113, 13, 213, 0],
            [zero_s+114, 14, 214, 0],
            [zero_s+115, 15, 215, 0],
            [zero_s+116, 16, 216, 0],
            [zero_s+117, 17, 217, 1],
            [zero_s+118, 18, 218, 0],
            [zero_s+119, 19, 219, 0],
        ]
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=10, bufferSize=5, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        expected_s = np.zeros((10,84,84), dtype=np.uint8)

        expected_a = np.zeros(10).astype(np.uint8)
        expected_r = np.zeros(10).astype(np.int)
        expected_t = np.zeros(10).astype(np.uint8)

        for i, vals in enumerate(tables):
            #with self.subTest(i=i):
                s, a, r, t = vals

                # each experiences add to transitions cyclically.
                idx = i % 10

                expected_s[idx,...] = s
                expected_a[idx] = a
                expected_r[idx] = r
                expected_t[idx] = t

                # the s will normalize before call to add() and it type is float32
                # and in same time, the type of t is boolean. 
                transitions.add(s.astype(np.float32)/255.0, a, r, t==1)

                # insertIndex cycles from 0 to replay_memory size - 1.
                self.assertEqual(transitions.insertIndex, 0 if i%10 == 9 else i%10+1)
                # numEntries will increase to replay_ememory size and will be fixed thereafter.
                self.assertEqual(transitions.numEntries, min(i+1, 10))
                # in this time, buf_ind is not change.
                self.assertEqual(transitions.buf_ind, -1)
                # s is denormalized in add() and it type is casted to uint8.
                assert_equal(transitions.s, expected_s, use_float_equal=True, verbose=0)
                assert_equal(transitions.a, expected_a, use_float_equal=True, verbose=0)
                assert_equal(transitions.r, expected_r, use_float_equal=True, verbose=0)
                # t is converted from boolean to number.
                assert_equal(transitions.t, expected_t, use_float_equal=True, verbose=0)
        

    def test_03_sample_assertion_check(self):
        zero_s = np.zeros((84, 84)).astype(np.uint8)
        tables = [
            [zero_s+100, 0,  200, 0],
            [zero_s+101, 1,  201, 0],
            [zero_s+102, 2,  202, 0],
            [zero_s+103, 3,  203, 0],
        ]
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=10, bufferSize=5, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        for vals in tables:
            s, a, r, t = vals
            transitions.add(s.astype(np.float32)/255.0, a, r, t==1)

        # If number of numEntries is less than batch_size of argument of sample() that 32, raises AssertionError.
        with self.assertRaises(AssertionError):
            s, a, r, s2, t = transitions.sample(32)

        # If number of numEntries is less than bufferSize that 5, raises AssertionError.
        with self.assertRaises(AssertionError):
            s, a, r, s2, t = transitions.sample(2)

    def test_03_sample_random_sampling(self):
        zero_s = np.zeros((84, 84)).astype(np.uint8)
        tables = [
            [zero_s+100, 0,  200, 0],
            [zero_s+101, 1,  201, 0],
            [zero_s+102, 2,  202, 0],
            [zero_s+103, 3,  203, 0],
            [zero_s+104, 4,  204, 0],
            [zero_s+105, 5,  205, 0],
            [zero_s+106, 6,  206, 1],
            [zero_s+107, 7,  207, 0],
            [zero_s+108, 8,  208, 0],
            [zero_s+109, 9,  209, 0],
            [zero_s+110, 10, 210, 0],
            [zero_s+111, 11, 211, 1],
            [zero_s+112, 12, 212, 0],
            [zero_s+113, 13, 213, 0],
            [zero_s+114, 14, 214, 0],
            [zero_s+115, 15, 215, 0],
            [zero_s+116, 16, 216, 0],
            [zero_s+117, 17, 217, 1],
            [zero_s+118, 18, 218, 0],
            [zero_s+119, 19, 219, 0],
        ]
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=20, bufferSize=5, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        for vals in tables:
            s, a, r, t = vals
            transitions.add(s.astype(np.float32)/255.0, a, r, t==1)

        self.assertEqual(transitions.numEntries, 20)

        org_random = transitions.random.random
        samples_indexes = []
        def inject_random(a, b):
            nonlocal samples_indexes
            samples_indexes.append(org_random(a, b))
            return samples_indexes[-1]

        with patch.object(transitions.random, 'random', wraps=inject_random) as mock_random:

            for i in range(1000):
                with self.subTest(i):
                    transitions.sample(2)
        
        # In this case, the maximum sample index is 16 that numEntries - hist_len.
        # because s2 is get from sample index + 1
        self.assertEqual(max(samples_indexes), 16)
        # in the same time, the minimum sample index is 2, because DQN3.0 said 'start at 2 because of previous action'.
        self.assertEqual(min(samples_indexes), 2)

    def test_03_sample_fill_buffer(self):
        zero_s = np.zeros((84, 84)).astype(np.uint8)
        tables = np.array([
            [zero_s+100, 0,  200, 0],
            [zero_s+101, 1,  201, 0],
            [zero_s+102, 2,  202, 0],
            [zero_s+103, 3,  203, 0],
            [zero_s+104, 4,  204, 0],
            [zero_s+105, 5,  205, 0],
            [zero_s+106, 6,  206, 0],
            [zero_s+107, 7,  207, 0],
            [zero_s+108, 8,  208, 0],
            [zero_s+109, 9,  209, 0],
            [zero_s+110, 10, 210, 0],
            [zero_s+111, 11, 211, 0],
            [zero_s+112, 12, 212, 0],
            [zero_s+113, 13, 213, 0],
            [zero_s+114, 14, 214, 0],
            [zero_s+115, 15, 215, 0],
            [zero_s+116, 16, 216, 0],
            [zero_s+117, 17, 217, 0],
            [zero_s+118, 18, 218, 0],
            [zero_s+119, 19, 219, 0],
        ])
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=20, bufferSize=4, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        self.assertEqual(len(transitions.buf_s), 4)
        self.assertEqual(len(transitions.buf_a), 4)
        self.assertEqual(len(transitions.buf_r), 4)
        self.assertEqual(len(transitions.buf_s2), 4)
        self.assertEqual(len(transitions.buf_term), 4)

        for vals in tables:
            s, a, r, t = vals
            transitions.add(s.astype(np.float32)/255.0, a, r, t==1)

        org_fill_buffer = transitions._fill_buffer
        def inject_fill_buffer():
            org_fill_buffer()

        sample_indexes = [i for i in range(1,16)] * 2

        @patch.object(transitions, "_fill_buffer", wraps=inject_fill_buffer)
        @patch.object(transitions.random, 'random', side_effect=sample_indexes+sample_indexes)
        def test_func(mock_random, mock_fill_buffer):
            sampled = transitions.sample(2)

            self.assertEqual(len(transitions.buf_s), 4)
            self.assertEqual(len(transitions.buf_a), 4)
            self.assertEqual(len(transitions.buf_r), 4)
            self.assertEqual(len(transitions.buf_s2), 4)
            self.assertEqual(len(transitions.buf_term), 4)

            # If sample() called first time, _fill_buffer called once.
            mock_fill_buffer.assert_called_once_with()

            # In this time, buf_s,... are filled sampled experience data from tables[1] to tables[4].
            for i, bufs in zip(sample_indexes,
                                zip(transitions.buf_s,
                                    transitions.buf_a,
                                    transitions.buf_r,
                                    transitions.buf_s2,
                                    transitions.buf_term)):
                # the sampled s has 4 frames from sampled index to sampled index + hist_len - 1
                self.assertTrue(float_equal(bufs[0]*255.0, make_state(tables, i-1)))

                # the sampled a is get from sample index + 3.
                self.assertEqual(bufs[1], tables[(i-1)+3,1])

                # the sampled r is get from sample index + 3.
                self.assertEqual(bufs[2], tables[(i-1)+3,2])

                # the sampled s2 has 4 frames from sampled index+1 to sampled index+1 + hist_len - 1
                self.assertTrue(float_equal(bufs[3]*255.0, make_state(tables, (i-1)+1)))

                # the sampled t is get from sample index + 4
                self.assertEqual(bufs[4], tables[(i-1)+4,3])

            # The return value of sample () on the first call is the same as idx = 1 and 2 above.
            i = 1
            # the sampled s has 4 frames from sampled index to sampled index + hist_len - 1
            self.assertTrue(float_equal(sampled[0]*255.0, make_state(tables, i-1, batch_size=2)))

            # the sampled a is get from sample index + 3.
            self.assertTrue(np.all(sampled[1] == tables[(i-1)+3:(i-1)+5,1]))

            # the sampled r is get from sample index + 3.
            self.assertTrue(np.all(sampled[2] == tables[(i-1)+3:(i-1)+5,2]))

            # the sampled s2 has 4 frames from sampled index+1 to sampled index+1 + hist_len - 1
            self.assertTrue(float_equal(sampled[3]*255.0, make_state(tables, (i-1)+1, batch_size=2)))

            # the sampled t is get from sample index + 4
            self.assertTrue(np.all(sampled[4] == tables[(i-1)+4:(i-1)+6,3]))


            self.assertEqual(transitions.buf_ind, 2)

            mock_fill_buffer.reset_mock()

            sampled = transitions.sample(2)

            self.assertEqual(len(transitions.buf_s), 4)
            self.assertEqual(len(transitions.buf_a), 4)
            self.assertEqual(len(transitions.buf_r), 4)
            self.assertEqual(len(transitions.buf_s2), 4)
            self.assertEqual(len(transitions.buf_term), 4)

            # If sample() called second time when buf_ind+hist_len-1 is still less than  bufferSize,
            # _fill_buffer() will not called.
            self.assertEqual(transitions.buf_ind, 4)
            self.assertLess(transitions.buf_ind, transitions.buf_ind+transitions.hist_len-1)
            mock_fill_buffer.assert_not_called()

            # The return value of sample () on the second call is the same as idx = 3 and 4 above.
            i = 3
            # the sampled s has 4 frames from sampled index to sampled index + hist_len - 1
            self.assertTrue(float_equal(sampled[0]*255.0, make_state(tables, i-1, batch_size=2)))

            # the sampled a is get from sample index + 3.
            self.assertTrue(np.all(sampled[1] == tables[(i-1)+3:(i-1)+5,1]))

            # the sampled r is get from sample index + 3.
            self.assertTrue(np.all(sampled[2] == tables[(i-1)+3:(i-1)+5,2]))

            # the sampled s2 has 4 frames from sampled index+1 to sampled index+1 + hist_len - 1
            self.assertTrue(float_equal(sampled[3]*255.0, make_state(tables, (i-1)+1, batch_size=2)))

            # the sampled t is get from sample index + 4
            self.assertTrue(np.all(sampled[4] == tables[(i-1)+4:(i-1)+6,3]))


            mock_fill_buffer.reset_mock()
            sample_index_start = mock_random.call_count
            start_index = sample_indexes[sample_index_start]

            sampled = transitions.sample(2)

            # When sampled () is called once more, buf_ind+hist_len-1 is over than bufferSize,
            # therefore fill_buffer will run.
            self.assertEqual(transitions.buf_ind, 2)
            self.assertLess(transitions.buf_ind, transitions.buf_ind+transitions.hist_len-1)
            mock_fill_buffer.assert_called_once_with()

            # the buf_s,... are filled the next experiences data
            # that got from between 5 to 9 indexes in tables.
            for i, bufs in zip(sample_indexes[sample_index_start:],
                                zip(transitions.buf_s,
                                    transitions.buf_a,
                                    transitions.buf_r,
                                    transitions.buf_s2,
                                    transitions.buf_term)):
                # the sampled s has 4 frames from sampled index to sampled index + hist_len - 1
                self.assertTrue(float_equal(bufs[0]*255.0, make_state(tables, i-1)))

                # the sampled a is get from sample index + 3.
                self.assertEqual(bufs[1], tables[(i-1)+3,1])

                # the sampled r is get from sample index + 3.
                self.assertEqual(bufs[2], tables[(i-1)+3,2])

                # the sampled s2 has 4 frames from sampled index+1 to sampled index+1 + hist_len - 1
                self.assertTrue(float_equal(bufs[3]*255.0, make_state(tables, (i-1)+1)))

                # the sampled t is get from sample index + 4
                self.assertEqual(bufs[4], tables[(i-1)+4,3])

        test_func()

    def test_03_sample_4(self):
        zero_s = np.zeros((84, 84)).astype(np.uint8)
        tables = np.array([
            [zero_s+100, 0,  200, 0],
            [zero_s+101, 1,  201, 0], # <- If sample index is 1 will sampled.
            [zero_s+102, 2,  202, 1],
            [zero_s+103, 3,  203, 0], # <- If sample index is 3 will resample. 
            [zero_s+104, 4,  204, 0], # <- If sample index is 4 will sampled.
            [zero_s+105, 5,  205, 0],
            [zero_s+106, 6,  206, 1], # <- If sample index is 6 will sampled.
            [zero_s+107, 7,  207, 0],
            [zero_s+108, 8,  208, 0], # <- If sample index is 8 will resample. 
            [zero_s+109, 9,  209, 0],
            [zero_s+110, 10, 210, 0],
            [zero_s+111, 11, 211, 1],
            [zero_s+112, 12, 212, 0],
            [zero_s+113, 13, 213, 0], # <- If sample index is 13 will sampled.
            [zero_s+114, 14, 214, 0], # <- If sample index is 14 will resample.
            [zero_s+115, 15, 215, 0],
            [zero_s+116, 16, 216, 0],
            [zero_s+117, 17, 217, 1],
            [zero_s+118, 18, 218, 0],
            [zero_s+119, 19, 219, 0],
        ])
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=20, bufferSize=4, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        # Setup transition table.
        for vals in tables:
            s, a, r, t = vals
            transitions.add(s.astype(np.float32)/255.0, a, r, t==1)

        org_fill_buffer = transitions._fill_buffer
        def inject_fill_buffer():
            org_fill_buffer()

        @patch.object(transitions, "_fill_buffer", wraps=inject_fill_buffer)
        @patch.object(transitions.random, 'random', side_effect=(np.array([1,3,4,6,8,14,13])+1).tolist())
        def test_func(mock_random, mock_fill_buffer):
            s, a, r, s2, term = transitions.sample(2)

            self.assertEqual(len(transitions.buf_s), 4)
            self.assertEqual(len(transitions.buf_a), 4)
            self.assertEqual(len(transitions.buf_r), 4)
            self.assertEqual(len(transitions.buf_s2), 4)
            self.assertEqual(len(transitions.buf_term), 4)

            # If sample() called first time, _fill_buffer called once.
            mock_fill_buffer.assert_called_once_with()

            # _fill_buffer includes experiences data with indexes 1,4,6 and 13.
            for i, bufs in zip([1,4,6,13],
                                zip(transitions.buf_s,
                                    transitions.buf_a,
                                    transitions.buf_r,
                                    transitions.buf_s2,
                                    transitions.buf_term)):
                #with self.subTest('call sample() first time: idx[{}]:buf_s'.format(i)):
                #    # the sampled s has 4 frames from sampled index to sampled index + hist_len - 1
                #    self.assertTrue(float_equal(bufs[0]*255.0, make_state(tables, i)))
                with self.subTest('call sample() first time: idx[{}]:buf_a'.format(i)):
                    # the sampled a is get from sample index + 3.
                    self.assertEqual(bufs[1], tables[i+3,1])
                with self.subTest('call sample() first time: idx[{}]:buf_r'.format(i)):
                    # the sampled r is get from sample index + 3.
                    self.assertEqual(bufs[2], tables[i+3,2])
                #with self.subTest('call sample() first time: idx[{}]:buf_s2'.format(i)):
                #    # the sampled s2 has 4 frames from sampled index+1 to sampled index+1 + hist_len - 1
                #    self.assertTrue(float_equal(bufs[3]*255.0, make_state(tables, i+1)))
                with self.subTest('call sample() first time: idx[{}]:buf_t'.format(i)):
                    # the sampled t is get from sample index + 4
                    self.assertEqual(bufs[4], tables[i+4,3])
                    self.assertEqual(tables[i+3,3], 0)
                    if i == 13:
                        self.assertEqual(bufs[4], 1)

            #--------------------
            # setup expected_s
            #--------------------
            expected_s = np.empty((4,4,84,84), dtype=np.float32)
            expected_s[0,...] = make_state(tables, 1)
            expected_s[1,...] = make_state(tables, 4)
            expected_s[2,...] = make_state(tables, 6)
            expected_s[3,...] = make_state(tables, 13)

            # The first sample is include the terminal is True in index 1.
            # In this case, indexes 0 and 1 of the s are all filled with zeros.
            """
                [zero_s+101, 1,  201, 0], <= filled with zero
                [zero_s+102, 2,  202, 1], <= filled with zero
                [zero_s+103, 3,  203, 0],
                [zero_s+104, 4,  204, 0], 
            """
            expected_s[0,0:2,...] = np.zeros((2, 84,84)).astype(np.float32)

            # The second sample also in index 2.
            # In this case, sampled state is only index 3 in the s.
            # Other state are all filled with zeros.
            """
                [zero_s+104, 4,  204, 0], <= filled with zero
                [zero_s+105, 5,  205, 0], <= filled with zero
                [zero_s+106, 6,  206, 1], <= filled with zero
                [zero_s+107, 7,  207, 0],
            """
            expected_s[1,0:3,...] = np.zeros((3, 84,84)).astype(np.float32)

            # The 3rd sample also in index 0.
            """
                [zero_s+106, 6,  206, 1], <= filled with zero
                [zero_s+107, 7,  207, 0],
                [zero_s+108, 8,  208, 0],
                [zero_s+109, 9,  209, 0],
            """
            expected_s[2,0,...] = np.zeros((1, 84,84)).astype(np.float32)

            # The 4th sample doesn't filled with zero.
            """
                [zero_s+113, 13, 213, 0],
                [zero_s+114, 14, 214, 0],
                [zero_s+115, 15, 215, 0],
                [zero_s+116, 16, 216, 0],
            """

            #--------------------
            # setup expected_s2
            #--------------------
            expected_s2 = np.empty((4,4,84,84), dtype=np.float32)
            expected_s2[0,...] = make_state(tables, 2)
            expected_s2[1,...] = make_state(tables, 5)
            expected_s2[2,...] = make_state(tables, 7)
            expected_s2[3,...] = make_state(tables, 14)

            # the 1st s2.
            """
                [zero_s+102, 2,  202, 1], <= filled with zero
                [zero_s+103, 3,  203, 0],
                [zero_s+104, 4,  204, 0],
                [zero_s+105, 5,  205, 0],
            """
            expected_s2[0,0:1,...] = np.zeros((1, 84,84)).astype(np.float32)

            # the 2nd s2.
            """
                [zero_s+105, 5,  205, 0], <= filled with zero
                [zero_s+106, 6,  206, 1], <= filled with zero
                [zero_s+107, 7,  207, 0],
                [zero_s+108, 8,  208, 0],
            """
            expected_s2[1,0:2,...] = np.zeros((2, 84,84)).astype(np.float32)

            # the 3rd s2 has not all zeros state.
            """
                [zero_s+107, 7,  207, 0],
                [zero_s+108, 8,  208, 0],
                [zero_s+109, 9,  209, 0],
                [zero_s+110, 10, 210, 0],
            """

            # the 4th s2 has not all zeros state.
            # Because it checks the terminal state between indices 0 and 3.
            """
                [zero_s+114, 14, 214, 0],
                [zero_s+115, 15, 215, 0],
                [zero_s+116, 16, 216, 0],
                [zero_s+117, 17, 217, 1],
            """

            #--------------------------------------
            # check s and s2 in return of sample()
            #--------------------------------------

            # first sample() is already called.
            self.assertTrue(float_equal(s*255.0, expected_s[0:2,...]))
            self.assertTrue(float_equal(s2*255.0, expected_s2[0:2,...]))

            # next sample() call.
            s, a, r, s2, term = transitions.sample(2)
            self.assertTrue(float_equal(s*255.0, expected_s[2:,...]))
            self.assertTrue(float_equal(s2*255.0, expected_s2[2:,...], verbose=0))

        test_func()

    def test_4_normalize_env(self):
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=20, bufferSize=4, debug=False, screen_normalize='env')
        transitions = TransitionTable(args, shapes)

        s = np.full((84,84), 123.0, dtype=np.float32)

        transitions.add_recent_state(s / 255.0, True)
        assert_equal(transitions.recent_s[3], s.astype(np.uint8))
        get_s = transitions.get_recent()
        assert_equal(get_s[3], s / 255.0)

        for _ in range(20):
            transitions.add(s / 255.0, 0, 0, False)

        assert np.all(transitions.s == 123)

        get_s = transitions.sample(1)
        assert_equal(get_s[3], s / 255.0)

    def test_4_normalize_trans(self):
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=20, bufferSize=4, debug=False, screen_normalize='trans')
        transitions = TransitionTable(args, shapes)

        s = np.full((84,84), 123.0, dtype=np.float32)

        print(np.mean(s))

        transitions.add_recent_state(s, True)
        assert_equal(transitions.recent_s[3], s.astype(np.uint8))
        get_s = transitions.get_recent()
        assert_equal(get_s[3], s / 255.0)

        for _ in range(20):
            transitions.add(s, 0, 0, False)

        assert np.all(transitions.s == 123)

        get_s = transitions.sample(1)
        assert_equal(get_s[3], s / 255.0)

    def test_4_normalize_none(self):
        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        args = Namespace(hist_len=4, replay_memory=20, bufferSize=4, debug=False, screen_normalize='none')
        transitions = TransitionTable(args, shapes)

        s = np.full((84,84), 123.0, dtype=np.float32)

        print(np.mean(s))

        transitions.add_recent_state(s, True)
        assert_equal(transitions.recent_s[3], s.astype(np.uint8))
        get_s = transitions.get_recent()
        assert_equal(get_s[3], s)

        for _ in range(20):
            transitions.add(s, 0, 0, False)

        assert np.all(transitions.s == 123)

        get_s = transitions.sample(1)
        assert_equal(get_s[3], s)

    def test_99_diff_to_dqn3(self):

        verbose = 0

        print('\n---- loading transition s ----')
        sys.stdout.flush()
        dqn_transitions_s = load_lua('./dqn3.0_dump/transitions_10000_s.dat')
        print('---- loading transition a ----')
        sys.stdout.flush()
        dqn_transitions_a = load_lua('./dqn3.0_dump/transitions_10000_a.dat')
        print('---- loading transition r ----')
        sys.stdout.flush()
        dqn_transitions_r = load_lua('./dqn3.0_dump/transitions_10000_r.dat')
        print('---- loading transition t ----')
        sys.stdout.flush()
        dqn_transitions_t = load_lua('./dqn3.0_dump/transitions_10000_t.dat')

        print('---- preproc dqn_transitions ----')
        dqn_transitions_s = dqn_transitions_s.numpy()
        dqn_transitions_a = dqn_transitions_a.numpy()
        dqn_transitions_r = dqn_transitions_r.numpy()
        dqn_transitions_t = dqn_transitions_t.numpy()
        dqn_transitions_s = dqn_transitions_s[:10000]
        dqn_transitions_a = dqn_transitions_a[:10000]
        dqn_transitions_r = dqn_transitions_r[:10000]
        dqn_transitions_t = dqn_transitions_t[:10000]
        dqn_transitions_s = dqn_transitions_s.reshape(10000,84,84).astype(np.float32) / 255.0
        dqn_transitions_a = dqn_transitions_a - 1

        print('---- loading sample s ----')
        sys.stdout.flush()
        dqn_sample_s = load_lua('./dqn3.0_dump/transitions_sample_1000_s.dat')
        print('---- loading sample a ----')
        sys.stdout.flush()
        dqn_sample_a = load_lua('./dqn3.0_dump/transitions_sample_1000_a.dat')
        print('---- loading sample r ----')
        sys.stdout.flush()
        dqn_sample_r = load_lua('./dqn3.0_dump/transitions_sample_1000_r.dat')
        print('---- loading sample s2 ----')
        sys.stdout.flush()
        dqn_sample_s2 = load_lua('./dqn3.0_dump/transitions_sample_1000_s2.dat')
        print('---- loading sample t ----')
        sys.stdout.flush()
        dqn_sample_t = load_lua('./dqn3.0_dump/transitions_sample_1000_t.dat')
        print('---- loading sample_index ----')
        sys.stdout.flush()
        dqn_sample_index = load_lua('./dqn3.0_dump/transitions_sample_index.dat')
        print('dqn_sample_index len', len(dqn_sample_index))
        sys.stdout.flush()

        print('---- preproc dqn_sample ----')
        dqn_sample_s = [v.numpy().reshape(8, 4, 84, 84) for v in dqn_sample_s]
        dqn_sample_a = [v.numpy() - 1 for v in dqn_sample_a]
        dqn_sample_r = [v.numpy() for v in dqn_sample_r]
        dqn_sample_s2 = [v.numpy().reshape(8, 4, 84, 84) for v in dqn_sample_s2]
        dqn_sample_t = [v.numpy() for v in dqn_sample_t]

        print('---- loading buffer s ----')
        sys.stdout.flush()
        dqn_buffer_s = []
        dqn_buffer_a = []
        dqn_buffer_r = []
        dqn_buffer_s2 = []
        dqn_buffer_t = []
        for step in range(1, 1000, 64):
            dqn_buffer_s.append(load_lua('./dqn3.0_dump/buffer_s_{:03d}.dat'.format(step)))
            dqn_buffer_a.append(load_lua('./dqn3.0_dump/buffer_a_{:03d}.dat'.format(step)))
            dqn_buffer_r.append(load_lua('./dqn3.0_dump/buffer_r_{:03d}.dat'.format(step)))
            dqn_buffer_s2.append(load_lua('./dqn3.0_dump/buffer_s2_{:03d}.dat'.format(step)))
            dqn_buffer_t.append(load_lua('./dqn3.0_dump/buffer_t_{:03d}.dat'.format(step)))

        dqn_buffer_s = [buf.numpy().reshape(512,4,84,84) for buf in dqn_buffer_s]
        dqn_buffer_a = [buf.numpy() - 1 for buf in dqn_buffer_a]
        dqn_buffer_r = [buf.numpy() for buf in dqn_buffer_r]
        dqn_buffer_s2 =[buf.numpy().reshape(512,4,84,84) for buf in dqn_buffer_s2]
        dqn_buffer_t = [buf.numpy() for buf in dqn_buffer_t]

        shapes = {'s': (84, 84),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}
        sys.argv += ['--backend','pytorch',
                     '--env', 'breakout',
                     '--random_type', 'torch',
                     '--logdir','/tmp',
                     '--screen_normalize', 'env']
        from config import get_opt
        from initenv import setup
        opt =get_opt()
        game_env, agent, game_actions, args = setup(opt)
        
        transitions = agent.transitions

        print('---- add transitions ----')
        sys.stdout.flush()
        for s, a, r, t in zip(dqn_transitions_s,
                              dqn_transitions_a,
                              dqn_transitions_r,
                              dqn_transitions_t):
            transitions.add(s, a, r, t)

        self.assertEqual(transitions.numEntries, 10000)

        for i in range(transitions.numEntries):
            assert_equal(transitions.s[i], np.uint8(dqn_transitions_s[i]*255.0))
            assert_equal(transitions.a[i], dqn_transitions_a[i])
            assert_equal(transitions.r[i], dqn_transitions_r[i])
            assert_equal(transitions.t[i], dqn_transitions_t[i])

        transitions.random.manualSeed(1)

        org_random = transitions.random.random

        from utils.random import TorchRandom
        self.assertIsInstance(transitions.random, TorchRandom)

        sample_indexes = []
        def inject_random(a, b):
            nonlocal sample_indexes
            r = org_random(a,b)
            sample_indexes.append(r-1)
            return r

        org_fillBuffer = transitions._fill_buffer
        fill_buffer_count = 0
        def inject_fill_buffer():
            nonlocal fill_buffer_count
            fill_buffer_count+=1
            org_fillBuffer()

        def get_dqn_fillBuffer():
            for buffer in zip(dqn_buffer_s,
                              dqn_buffer_a,
                              dqn_buffer_r,
                              dqn_buffer_s2,
                              dqn_buffer_t):

                yield buffer

        @patch.object(transitions, '_fill_buffer', wraps=inject_fill_buffer)
        @patch.object(transitions.random, 'random', wraps=inject_random)
        def test_func(mock_random, mock_fillBuffer):

            print('---- test start ----')

            dqn_buffer_gen = get_dqn_fillBuffer()
            sys.stdout.flush()
            for i, dqn_sample in enumerate(zip(dqn_sample_s,
                                               dqn_sample_a,
                                               dqn_sample_r,
                                               dqn_sample_s2,
                                               dqn_sample_t)):
                dqn_s, dqn_a, dqn_r, dqn_s2, dqn_t = dqn_sample

                s, a, r, s2, t = transitions.sample(8)

                if mock_fillBuffer.call_count == 1:

                    #print('===sample_indexes===')
                    #print(sample_indexes)
                    #print('===dqn_sample_index===')
                    #print(dqn_sample_index[:len(sample_indexes)])

                    sample_index_idx = (fill_buffer_count-1) * 512
                    sample_index_end = fill_buffer_count * 512
                    assert_equal(sample_indexes[sample_index_idx:sample_index_end], dqn_sample_index[sample_index_idx:sample_index_end])

                    dqn_buff_s,dqn_buff_a,dqn_buff_r,dqn_buff_s2,dqn_buff_t = next(dqn_buffer_gen)
                    #print('===buf_a===')
                    #print(transitions.buf_a)
                    #print('===dqn buf_a===')
                    #print(dqn_buff_a)
                    assert_equal(transitions.buf_a , dqn_buff_a, verbose=0)
                    assert_equal(transitions.buf_r , dqn_buff_r)
                    assert_equal(transitions.buf_term , dqn_buff_t)
                    for i, index in enumerate(sample_indexes[sample_index_idx:sample_index_end]):
                        for j in range(4):
                            self.assertTrue(float_equal(np.uint8(transitions.buf_s[i][j]*255.0), transitions.s[index+j], verbose=0))
                            self.assertTrue(float_equal(transitions.buf_s[i][j], dqn_buff_s[i][j], verbose=0))
                    self.assertTrue(float_equal(transitions.buf_s, dqn_buff_s, verbose=verbose))
                    self.assertTrue(float_equal(transitions.buf_s2, dqn_buff_s2, verbose=verbose))
                    mock_fillBuffer.reset_mock()

                with self.subTest('{} check s'.format(i)):
                    self.assertTrue(float_equal(s, dqn_s, name=',={} check s,='.format(i), verbose=verbose))
                with self.subTest('{} check a'.format(i)):
                    assert_equal(a , dqn_a)
                with self.subTest('{} check r'.format(i)):
                    assert_equal(r , dqn_r)
                with self.subTest('{} check s2'.format(i)):
                    self.assertTrue(float_equal(s2, dqn_s2, name=',={} check s2,='.format(i), verbose=verbose))
                with self.subTest('{} check t'.format(i)):
                    assert_equal(t , dqn_t)

        test_func()

