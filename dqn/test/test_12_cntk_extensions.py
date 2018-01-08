import sys
import unittest
from unittest.mock import *
import numpy as np

import cntk as C

sys.path.append('../')
sys.path.append('../../')

class TestCNTKExtensions(unittest.TestCase):

    def test_01_start_gradient(self):
        from cntk_extensions import start_gradient
        
        C.device.try_set_default_device(C.device.cpu())
        
        q_all = C.sequence.input_variable((1,4), needs_gradient=True, name='q_all')
        targets = C.sequence.input_variable((1,4), needs_gradient=False, name='targets')
        op = start_gradient(q_all, targets)
        
        q_all_data = np.array([[0.1,0.5,0.3,0.9]], dtype=np.float32)
        targets_data = np.array([[0.0,0.0,0.6,0.0]], dtype=np.float32)

        
        grads, result = op.grad({q_all:q_all_data, targets:targets_data}, wrt=[q_all], outputs=[op.output])
        
        print('grads',grads)
        print('result',result)
   