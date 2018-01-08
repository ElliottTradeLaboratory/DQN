import sys
import unittest
from unittest.mock import *

import numpy as np

sys.path.append('../')
sys.path.append('../../')

class TestPytorchExtensions(unittest.TestCase):

    def test_01_RectifierFunction(self):
        import torch
        from torch.autograd import gradcheck, Variable
        from pytorch_extensions import RectifierFunction
        
        input = (Variable(torch.randn(20,20).double(), requires_grad=True),)
        test = gradcheck(RectifierFunction.apply, input, eps=1e-6, atol=1e-4)
        print(test) 
