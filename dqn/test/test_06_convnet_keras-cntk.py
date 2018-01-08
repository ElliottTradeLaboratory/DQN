import unittest
from abstract_convnet_keras import *


class Test01KerasCNTKConvnet(AbstractTestKerasConvnet, unittest.TestCase):

    def backend(self):
        return "cntk"
