import sys
import unittest
from unittest.mock import *
import numpy as np

from torch.utils.serialization import load_lua

sys.path.append('../../')
import utils
from utils.random import *
from utils import get_random

class Test01PytorchRandom(unittest.TestCase):

    class MockRandom(object):
        def __init__(self):
            self.manualSeed = Mock()

    @patch.dict('utils.random._random_factory', dict(hoge=Mock), clear=True)
    def test_00_get_random(self):

        """
        """
        with self.assertRaisesRegex(ValueError, 'When using this function for the first time, random_type is required\.'):
            get_random()

        """
             If it called arguments with random_type and seed, 
             it creates new random instance according to argument random_type
             with setting seed according to argument seed.
        """
        random = get_random('hoge', 1)
        random.manualSeed.assert_called_once_with(1)

        """
            If it called with no arguments, it returns the random instance
            that created at previous call.
        """
        random2 = get_random()
        random.manualSeed.assert_called_once_with(1)
        self.assertEqual(random, random2)

        """
            If it called with arguments that random_type and seed again
            (even if random_type is the same to argument of first call), 
            it also creates new random instance according to argument random_type
            with setting seed according to argument seed.
        """
        random3 = get_random('hoge', 1)
        random3.manualSeed.assert_called_once_with(1)
        self.assertNotEqual(random, random3)

        """
             If it called arguments with only random_type, 
             it creates new random instance according to argument random_type
             and does not setting seed.
        """
        random4 = get_random('hoge')
        random4.manualSeed.assert_not_called()
        self.assertNotEqual(random, random4)

        """
             Raise the ValueError with error message if it called arguments with 
             random_type that not contain to the _random_factory.
        """
        with self.assertRaisesRegex(ValueError, "random_type expected \['hoge'\] but got 'hoge2'"):
            random5 = get_random('hoge2')


    @patch('torch.manual_seed')
    def test_01_init(self, mock):
        """
          If get_random called with random_type that 'pytorch',
          return the instance of PytorchRandom and set random seed
          using torch.manual_seed function.
        """
        random = get_random('pytorch', 1)
        self.assertIsNotNone(random)
        self.assertIsInstance(random, PytorchRandom)
        mock.assert_called_once_with(1)

    def test_02_random(self):
        """
          If random called with arguments that a and b,
          return the random values that exactly same to 
          result of torch.random(a, b) in torch7.
        """
        dqn3_result = np.array(load_lua('./dqn3.0_random_dump/torch_random.dat'))
        dqn3_result = dqn3_result.flatten()
        # torch.random() is random sampled over [0, long max - 1].

        random = get_random('pytorch', 1)
        for i in range(len(dqn3_result)):
            with self.subTest(i=i):
                self.assertEqual(dqn3_result[i], random.random())

    def test_03_random_with_a(self):
        """
          If random called with arguments that a and b,
          return the random values that exactly same to 
          result of torch.random(a, b) in torch7.
        """
        dqn3_result = np.array(load_lua('./dqn3.0_random_dump/torch_random_with_a=4.dat'))
        dqn3_result = dqn3_result.flatten()
        # Because the start index value is 1 at torch7 but python is 0.
        dqn3_result -= 1

        random = get_random('pytorch', 1)
        for i in range(len(dqn3_result)):
            with self.subTest(i=i):
                self.assertEqual(dqn3_result[i], random.random(4))

    def test_04_random_with_a_and_b(self):
        """
          If random called with arguments that a and b,
          return the random values that exactly same to 
          result of torch.random(a, b) in torch7.
        """
        dqn3_result = np.array(load_lua('./dqn3.0_random_dump/torch_random_with_a=1_and_b=4.dat'))
        dqn3_result = dqn3_result.flatten()

        # Because the start index value is 1 at torch7 but python is 0.
        dqn3_result -= 1

        random = get_random('pytorch', 1)
        for i in range(len(dqn3_result)):
            with self.subTest(i=i):
                # a=0 is also due to the difference in the starting index.
                self.assertEqual(dqn3_result[i], random.random(0,4))


    def test_05_uniform(self):
        """
          If random called with arguments that a and b,
          return the random values that exactly same to 
          result of torch.random(a, b) in torch7.
        """
        dqn3_result = np.array(load_lua('./dqn3.0_random_dump/torch_uniform.dat'))
        dqn3_result = dqn3_result.flatten()

        random = get_random('pytorch', 1)
        for i in range(len(dqn3_result)):
            with self.subTest(i=i):
                self.assertEqual(np.around(dqn3_result[i],5), np.around(random.uniform(), 5))

    def test_06_uniform_with_a(self):
        """
          If random called with arguments that a and b,
          return the random values that exactly same to 
          result of torch.random(a, b) in torch7.
        """
        dqn3_result = np.array(load_lua('./dqn3.0_random_dump/torch_uniform_with_a=2.dat'))
        dqn3_result = dqn3_result.flatten()

        random = get_random('pytorch', 1)
        for i in range(len(dqn3_result)):
            with self.subTest(i=i):
                self.assertEqual(np.around(dqn3_result[i],5), np.around(random.uniform(2), 5))

    def test_07_uniform_with_a_and_b(self):
        """
          If random called with arguments that a and b,
          return the random values that exactly same to 
          result of torch.random(a, b) in torch7.
        """
        dqn3_result = np.array(load_lua('./dqn3.0_random_dump/torch_uniform_with_a=-0.9_and_b=0.9.dat'))
        dqn3_result = dqn3_result.flatten()

        random = get_random('pytorch', 1)
        for i in range(len(dqn3_result)):
            with self.subTest(i=i):
                # Less than 5 below the decimal point is not match as 
                # it may different the spec of floating point operation both python and lua.
                self.assertEqual(np.around(dqn3_result[i],5), np.around(random.uniform(-0.9,0.9), 5))

    def test_08_uniform_batch_a_and_b_len32(self):
        """
          If random called with arguments that a and b,
          return the random values that exactly same to 
          result of torch.random(a, b) in torch7.
        """
        dqn3_result = load_lua('./dqn3.0_random_dump/torch_uniform_with_a=-0.9_and_b=0.9_len=32.dat')
        print([type(v) for v in dqn3_result])
        dqn3_result = np.array(dqn3_result).flatten()
        dqn3_result = [v.numpy() for v in dqn3_result]

        random = get_random('pytorch', 1)
        for i in range(len(dqn3_result)):
            with self.subTest(i=i):
                # Less than 3 below the decimal point is not match as 
                # it may different the spec of floating point operation both python and lua.
                self.assertTrue(np.all(np.around(dqn3_result[i],3) == np.around(random.uniform_batch(-0.9,0.9,32), 3)))

