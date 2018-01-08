import sys
import unittest
from unittest.mock import *
import numpy as np

sys.path.append('../')

class TestAssertions(unittest.TestCase):

    def setUp(self):

        np.set_printoptions(threshold=np.inf)

    def test_float_equal1(self):
    
        from assertions import float_equal
        
        a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        assert float_equal(a,a)
        
        b = np.array([0.2, 0.2, 0.3], dtype=np.float32)

        assert not float_equal(a,b,verbose=1)
        
        a = np.array([0.000001, 0.2, 0.3], dtype=np.float32)
        b = np.array([0.000002, 0.2, 0.3], dtype=np.float32)

        assert float_equal(a,a,verbose=1)
