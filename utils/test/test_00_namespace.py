import sys
import unittest

sys.path.append('../')

class TestNamespace(unittest.TestCase):

    def test_01(self):
        import namespace
        ns = namespace.Namespace()
        
        self.assertEqual(len(ns), 0)
        
        ns = namespace.Namespace(a=1, b=2)
        
        self.assertEqual(len(ns), 2)
        self.assertEqual(ns.a, 1)
        self.assertEqual(ns.b, 2)
        self.assertEqual(ns['a'], 1)
        self.assertEqual(ns['b'], 2)
        self.assertEqual(ns.get('a'), 1)
        self.assertEqual(ns.get('b'), 2)
        
        ns.c = 3
        self.assertEqual(len(ns), 3)
        self.assertEqual(ns.a, 1)
        self.assertEqual(ns.b, 2)
        self.assertEqual(ns.c, 3)
        self.assertEqual(ns['a'], 1)
        self.assertEqual(ns['b'], 2)
        self.assertEqual(ns['c'], 3)
        self.assertEqual(ns.get('a'), 1)
        self.assertEqual(ns.get('b'), 2)
        self.assertEqual(ns.get('c'), 3)
        
        for i, kv in enumerate(ns.items()):
            with self.subTest(i=i):
                self.assertIn(kv[0], ['a', 'b', 'c'])
                self.assertIn(kv[1], [1, 2, 3])
        self.assertEqual(i, 2)

    def test_02(self):
        import namespace
        ns = namespace.Namespace(**dict(a=1, b=2, c=3))
        self.assertEqual(len(ns), 3)
        self.assertEqual(ns.a, 1)
        self.assertEqual(ns.b, 2)
        self.assertEqual(ns.c, 3)
        self.assertEqual(ns['a'], 1)
        self.assertEqual(ns['b'], 2)
        self.assertEqual(ns['c'], 3)
        self.assertEqual(ns.get('a'), 1)
        self.assertEqual(ns.get('b'), 2)
        self.assertEqual(ns.get('c'), 3)

    def test_03(self):
        import namespace
        ns = namespace.Namespace([('a',1), ('b',2), ('c',3)])
        self.assertEqual(len(ns), 3)
        self.assertEqual(ns.a, 1)
        self.assertEqual(ns.b, 2)
        self.assertEqual(ns.c, 3)
        self.assertEqual(ns['a'], 1)
        self.assertEqual(ns['b'], 2)
        self.assertEqual(ns['c'], 3)

    def test_04(self):
        import namespace
        ns = namespace.Namespace(zip(['a','b','c'],[1,2,3]))
        self.assertEqual(len(ns), 3)
        self.assertEqual(ns.a, 1)
        self.assertEqual(ns.b, 2)
        self.assertEqual(ns.c, 3)
        self.assertEqual(ns['a'], 1)
        self.assertEqual(ns['b'], 2)
        self.assertEqual(ns['c'], 3)
        self.assertEqual(ns.get('a'), 1)
        self.assertEqual(ns.get('b'), 2)
        self.assertEqual(ns.get('c'), 3)

        self.assertIn('a', ns)
        
        self.assertIsNone(ns['x'])
        self.assertIsNone(ns.get('x'))
        with self.assertRaises(AttributeError):
            a = ns.hogehoge
        
    def test_05(self):
        import namespace
        ns = namespace.Namespace(zip(['a','b','c'],[1,2,3]))
        self.assertTrue(ns.keys(), ['a','b','c'])
        self.assertTrue(ns.values(), [1,2,3])

    def test_06(self):
        import namespace
        ns1 = namespace.Namespace(zip(['a','b','c'],[1,2,3]))
        ns2 = namespace.Namespace([('a',1), ('b',2), ('c',3)])
        ns3 = namespace.Namespace(**dict(a=1, b=2, c=3))
        ns4 = namespace.Namespace()
        ns4.a = 1
        ns4.b = 2
        ns4.c = 3
        self.assertTrue(ns1 == ns2 == ns3 == ns4)

    def test_07_summary(self):
        import namespace
        ns = namespace.Namespace(zip(['a','b','c'],[1,2,3]))
        ns.d = dict(aa=4, bb=6, cc=7)
        ns.f = [100,200,300,400]
        ns.e = namespace.Namespace({1:10, 2:20, 3:30})
        
        import io
        f = io.StringIO()
        ns.summary(f)

        expected = 'a 1\n' \
                   'b 2\n' \
                   'c 3\n' \
                   'd aa 4\n' \
                   'd bb 6\n' \
                   'd cc 7\n' \
                   'e 1 10\n' \
                   'e 2 20\n' \
                   'e 3 30\n' \
                   'f [100, 200, 300, 400]\n'
        self.assertEqual(f.getvalue(), expected)

