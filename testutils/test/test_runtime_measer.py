import sys
from time import time, sleep
import unittest
from unittest.mock import *

sys.path.append('../')
class TestRuntimeMeaser(unittest.TestCase):

    def test1(self):
    
        class _TestClass(object):
            def __init__(self):
                self.prop1 = None
                self.prop2 = 0
            def method1(self, a, b):
                print('method1 called', a, b)
                sleep(1)
            def method2(self, c,d,e):
                print('method2 called', c, d, e)
                sleep(2)
            def method3(self):
                print('method3 called')
                self.prop3 = 123
                sleep(3)
                self._inner_method(9,8,7)
            def _inner_method(self, x,y,z):
                print('_inner_method called', x,y,z)
                sleep(0.5)
                return x,y,z

        from method_wrappers import set_measer, measers_summary, RuntimeMeaser
        from method_wrappers import set_tracer, tracers_summary, Tracer
        a = _TestClass()
        a = set_measer(a)
        a = set_tracer(a)

        print(a.method1)
        for _ in range(3):
            a.method3()
            a.method1([1,3,4],2)
            a.method2(dict(a=1,b=2,c=[3,4,5]),3,e=4)
            a.prop1 = 'a'

        measers_summary()
        tracers_summary()
