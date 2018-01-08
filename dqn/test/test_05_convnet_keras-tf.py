import unittest
from abstract_convnet_keras import *

class Test01KerasTensorflowConvnet(AbstractTestKerasConvnet, unittest.TestCase):

    def backend(self):
        return "tensorflow"
