import sys
from collections import OrderedDict, deque
import unittest
import numpy as np
import tensorflow as tf
import cv2

sys.path.append('../')
sys.path.append('../../')
from utils import *
from utils.tensorboard_logger import Logger
from dqn.visualizer import *

_count = 0
def count(init=None, next=True):
    global _count
    if init is not None:
        _count = init
    elif next:
        _count += 1
    return _count
    

_writer = None
class Test00EvaluationVisualizer(unittest.TestCase):

    visualizer = None

    @classmethod
    def setUpClass(cls):
        opt = Namespace(log_dir = '/tmp/test/EvaluationVisualizer')
        
        cls.visualizer = EvaluationVisualizer(opt)
        global _writer
        if _writer is None:
            print('new writer')
            _writer = cls.visualizer.logger.writer

    @classmethod
    def tearDownClass(cls):
        cls.visualizer.flush(1)
        
    def test_00__init__(self):
        cls = Test00EvaluationVisualizer

        self.assertIsNotNone(cls.visualizer.logger)
        self.assertIsInstance(cls.visualizer.logger, Logger)
        self.assertEqual(len(cls.visualizer.logger.values), 0)

    def test_01_addEvaluation(self):
        cls = Test00EvaluationVisualizer

        values = {'episode_score': np.array([123,45,596]),
                  'xxx': np.array([200,150,100]),}
                  
        msg_regx =  "^given invalid value name\, expect \['episode_score'\, 'episode_count'\] \
got \[('episode_score'[,]{0,1}[ ]{0,1}|'xxx'[,]{0,1}[ ]{0,1}){2}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addEvaluation(values)
            
        values = {'episode_score': np.array([123,45,596]),
                  'hogehoge': np.array([123,45,596]),
                  'episode_count': np.array([200,150,100]),}
                  
        msg_regx =  "^given invalid value name\, expect \['episode_score'\, 'episode_count'\] \
got \[('episode_score'[,]{0,1}[ ]{0,1}|'episode_count'[,]{0,1}[ ]{0,1}|'hogehoge'[,]{0,1}[ ]{0,1}){3}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addEvaluation(values)
            
        values = OrderedDict(episode_score=np.array([123,45,596]),
                            episode_count=29)
                  
        cls.visualizer.addEvaluation(values)
        self.assertEqual(len(cls.visualizer.logger.values), 4)
        init = 0
        for name, val in values.items():
            if name == 'episode_score':
                self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '6.evaluation/{}/max'.format(name))
                self.assertEqual(np.around(cls.visualizer.logger.values[count(next=False)].simple_value,1), val.max())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '6.evaluation/{}/min'.format(name))
                self.assertEqual(np.around(cls.visualizer.logger.values[count(next=False)].simple_value,1), val.min())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '6.evaluation/{}/avg'.format(name))
                self.assertEqual(np.around(cls.visualizer.logger.values[count(next=False)].simple_value,1), np.around(np.mean(val), 1))
            else:
                self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '6.evaluation/{}'.format(name))
                self.assertEqual(np.around(cls.visualizer.logger.values[count(next=False)].simple_value,1), val)
            init = None

    def test_02_addValidation(self):
        cls = Test00EvaluationVisualizer

        values = {'TDerror': np.array([123,45,596]),
                  'xxx': np.array([200,150,100]),}
                  
        msg_regx =  "^given invalid value name\, expect \['TDerror', 'V'\] \
got \[('TDerror'[,]{0,1}[ ]{0,1}|'xxx'[,]{0,1}[ ]{0,1}){2}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addValidation(values)
            
        values = {'TDerror': np.array([123,45,596]),
                  'hogehoge': np.array([123,45,596]),
                  'episode_count': np.array([200,150,100]),}
                  
        msg_regx =  "^given invalid value name\, expect \['TDerror', 'V'\] \
got \[('TDerror'[,]{0,1}[ ]{0,1}|'episode_count'[,]{0,1}[ ]{0,1}|'hogehoge'[,]{0,1}[ ]{0,1}){3}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addValidation(values)
            
        values = OrderedDict(TDerror=1.23, V=2.42)
                  
        cls.visualizer.addValidation(values)
        self.assertEqual(len(cls.visualizer.logger.values), 6)
        init = 4
        for name, val in values.items():
            self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '7.validation/{}'.format(name))
            self.assertEqual(np.around(cls.visualizer.logger.values[count(next=False)].simple_value,2), val)
            init = None

    def test_03_flush(self):
        cls = Test00EvaluationVisualizer
        cls.visualizer.flush(getattr(self, 'step', 1))
        self.assertEqual(len(cls.visualizer.logger.values), 0)

    def test_04_loop(self):
        for i in range(10):
            with self.subTest(i=i):
                self.step = i+1
                self.test_01_addEvaluation()
                self.test_02_addValidation()
                self.test_03_flush()


class TestCurrentStateVisualizer(unittest.TestCase):

    visualizer = None

    @classmethod
    def setUpClass(cls):
        opt = Namespace(log_dir = '/tmp/test/TestCurrentStateVisualizer', backend='tensorflow')
        cls.visualizer = CurrentStateVisualizer(opt)

    @classmethod
    def tearDownClass(cls):
        cls.visualizer.flush(1)
        
    def test_00__init__(self):
        cls = TestCurrentStateVisualizer

        self.assertIsNotNone(cls.visualizer.logger)
        self.assertIsInstance(cls.visualizer.logger, Logger)
        self.assertEqual(len(cls.visualizer.logger.values), 0)
        self.assertEqual(_writer, cls.visualizer.logger.writer)

    def test_01_addCurrentState(self):
        cls = TestCurrentStateVisualizer

        values = {'average_episode_scores': 234.44,
                  'episode_count': 456,
                  'XX': 0.9}
                  
        msg_regx =  "^given invalid value name\, expect \['average_episode_scores'\, 'episode_count', 'epsilon'\] \
got \[('average_episode_scores'[,]{0,1}[ ]{0,1}|'episode_count'[,]{0,1}[ ]{0,1}|'XX'[,]{0,1}[ ]{0,1}){3}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addCurrentState(values)
            
        values = {'average_episode_scores': np.array([123,345,672]),
                  'episode_count': 456}
                  
        msg_regx =  "^given invalid value name\, expect \['average_episode_scores'\, 'episode_count', 'epsilon'\] \
got \[('average_episode_scores'[,]{0,1}[ ]{0,1}|'episode_count'[,]{0,1}[ ]{0,1}){2}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addCurrentState(values)
            
        values = OrderedDict(average_episode_scores=4.3,
                  episode_count=456.9,
                  epsilon= 0.9)
        cls.visualizer.addCurrentState(values)
        self.assertEqual(len(cls.visualizer.logger.values), 3)

        init = 0
        for name, val in values.items():
            self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '1.current_state/{}'.format(name))
            self.assertEqual(np.around(cls.visualizer.logger.values[count(next=False)].simple_value,1), val)
            init = None

    def test_02_flush(self):
        cls = TestCurrentStateVisualizer
        cls.visualizer.flush(getattr(self, 'step', 1))
        self.assertEqual(len(cls.visualizer.logger.values), 0)


    def test_03_loop(self):
        for i in range(10):
            with self.subTest(i=i):
                self.step = i+1
                self.test_01_addCurrentState()
                self.test_02_flush()

class TestLearningVisualizer(unittest.TestCase):

    visualizer = None

    @classmethod
    def setUpClass(cls):
        layer_names = ['L1', 'L2', 'q_all']
        opt = Namespace(log_dir = '/tmp/test/TestLearningVisualizer', backend='tensorflow')
        cls.visualizer = LearningVisualizer(opt, layer_names)

    def test_00__init__(self):
        cls = TestLearningVisualizer

        self.assertIsNotNone(cls.visualizer.logger)
        self.assertIsInstance(cls.visualizer.logger, Logger)
        self.assertEqual(len(cls.visualizer.logger.values), 0)
        self.assertEqual(_writer, cls.visualizer.logger.writer)

    def test_01_addInputImages(self):
        cls = TestLearningVisualizer
        
        img = cv2.imread('./test_img/screen.png')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images = np.array([img, img , img, img])
        cls.visualizer.addInputImages(2, images)
        self.assertEqual(len(cls.visualizer.logger.values), 4)
        self.assertEqual(cls.visualizer.logger.values[0].tag, '2.network/input/image/0')
        self.assertEqual(cls.visualizer.logger.values[1].tag, '2.network/input/image/1')
        self.assertEqual(cls.visualizer.logger.values[2].tag, '2.network/input/image/2')
        self.assertEqual(cls.visualizer.logger.values[3].tag, '2.network/input/image/3')
        
        img2 = img[...,1:2]
        images = np.array([img2, img2 , img2, img2])
        cls.visualizer.addInputImages(3, images)
        self.assertEqual(len(cls.visualizer.logger.values), 8)
        self.assertEqual(cls.visualizer.logger.values[4].tag, '3.target_network/input/image/0')
        self.assertEqual(cls.visualizer.logger.values[5].tag, '3.target_network/input/image/1')
        self.assertEqual(cls.visualizer.logger.values[6].tag, '3.target_network/input/image/2')
        self.assertEqual(cls.visualizer.logger.values[7].tag, '3.target_network/input/image/3')

    def test_02_addTargetNetworkParameters(self):
        cls = TestLearningVisualizer

        parameters = {'X1': [np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12]),np.array([13,14,15])],
                      'X2': [np.array([101,102,103]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])],
                      'X3': [np.array([201,202,203]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])]}
        msg_regx =  "^given invalid value name\, expect \['L1'\, 'L2', 'q_all'\] got \[('X1'[,]{0,1}[ ]{0,1}|'X2'[,]{0,1}[ ]{0,1}|'X3'[,]{0,1}[ ]{0,1}){3}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addTargetNetworkParameters(parameters)

        parameters = OrderedDict([('L1',[np.array([[1,2,3],[11,12,13],[21,22,23]]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12]),np.array([13,14,15])]),
                                  ('L2',[np.array([[101,102,103],[111,112,113],[121,122,123]]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])]),
                                  ('q_all',[np.array([[201,202,203],[211,212,213],[221,222,223]]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])])])
        cls.visualizer.addTargetNetworkParameters(parameters)

        self.assertEqual(len(cls.visualizer.logger.values), 71)

        init = 8
        for layer_name, vals in parameters.items():
            for val, p_name in zip(vals, ['output', 'weight', 'bias', 'weight/gradients', 'bias/gradients']):
                self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '3.target_network/{}/{}/max'.format(layer_name, p_name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.max())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '3.target_network/{}/{}/min'.format(layer_name, p_name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.min())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '3.target_network/{}/{}/avg'.format(layer_name, p_name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, np.mean(val))
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '3.target_network/{}/{}'.format(layer_name, p_name))
                self.assertIsNotNone(cls.visualizer.logger.values[count(next=False)].histo)
                init = None
            if layer_name == 'q_all':
                tag = '3.target_network/{}/action'.format(layer_name)
                val = [211,212,213]
                for i in range(len(val)):
                    self.assertEqual(cls.visualizer.logger.values[count()].tag, '{}/{}'.format(tag, i))
                    self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val[i])

        
    def test_03_addNetworkParameters(self):
        cls = TestLearningVisualizer

        parameters = {'X1': [np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12]),np.array([13,14,15])],
                      'X2': [np.array([101,102,103]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])],
                      'X3': [np.array([201,202,203]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])]}
        msg_regx =  "^given invalid value name\, expect \['L1'\, 'L2', 'q_all'\] got \[('X1'[,]{0,1}[ ]{0,1}|'X2'[,]{0,1}[ ]{0,1}|'X3'[,]{0,1}[ ]{0,1}){3}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addNetworkParameters(parameters)

        parameters = OrderedDict([('L1',[np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12]),np.array([13,14,15])]),
                                ('L2',[np.array([101,102,103]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])]),
                                ('q_all',[np.array([[201,202,203],[211,212,213],[221,222,223]]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])])])
        cls.visualizer.addNetworkParameters(parameters)
        #print(cls.visualizer.logger.values[11])
        self.assertEqual(len(cls.visualizer.logger.values), 134)
        init = 71
        for layer_name, vals in parameters.items():
            for val, p_name in zip(vals, ['output', 'weight', 'bias', 'weight/gradients', 'bias/gradients']):
                self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '2.network/{}/{}/max'.format(layer_name, p_name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.max())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '2.network/{}/{}/min'.format(layer_name, p_name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.min())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '2.network/{}/{}/avg'.format(layer_name, p_name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, np.mean(val))
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '2.network/{}/{}'.format(layer_name, p_name))
                self.assertIsNotNone(cls.visualizer.logger.values[count(next=False)].histo)
                init = None
            if layer_name == 'q_all':
                tag = '2.network/{}/action'.format(layer_name)
                val = [211,212,213]
                for i in range(len(val)):
                    self.assertEqual(cls.visualizer.logger.values[count()].tag, '{}/{}'.format(tag, i))
                    self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val[i])


    def test_04_addGetQUpdateValues(self):
        cls = TestLearningVisualizer

        parameters = {'X1': [np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12]),np.array([13,14,15])],
                      'X2': [np.array([101,102,103]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])],
                      'X3': [np.array([201,202,203]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])]}
        msg_regx =  "^given invalid value name\, expect \['q2_all'\, 'q2_max'\, 'q2'\, 'r'\, 'q_all'\, 'q'\, 'delta'\, 'targets'\] \
got \[('X1'[,]{0,1}[ ]{0,1}|'X2'[,]{0,1}[ ]{0,1}|'X3'[,]{0,1}[ ]{0,1}){3}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addGetQUpdateValues(parameters)


        values = {'q2_all': np.array([[1,2,3]]),
                  'q2_max': np.array([101,102,103]),
                  'q2': np.array([201,202,203]),
                  'r': np.array([201,202,203]),
                  'q_all': np.array([201,202,203]),
                  'q': np.array([201,202,203]),
                  'delta': np.array([201,202,203]),
                  'targets': np.array([201,202,203]),
                  }
        cls.visualizer.addGetQUpdateValues(values)
        self.assertEqual(len(cls.visualizer.logger.values), 166)
        init = 134
        for name, val in values.items():
            self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '4.getQUpdate/{}/max'.format(name))
            self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.max())
            self.assertEqual(cls.visualizer.logger.values[count()].tag, '4.getQUpdate/{}/min'.format(name))
            self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.min())
            self.assertEqual(cls.visualizer.logger.values[count()].tag, '4.getQUpdate/{}/avg'.format(name))
            self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, np.mean(val))
            self.assertEqual(cls.visualizer.logger.values[count()].tag, '4.getQUpdate/{}'.format(name))
            self.assertIsNotNone(cls.visualizer.logger.values[count(next=False)].histo)
            init=None

        
    def test_05_addRMSpropValues(self):
        cls = TestLearningVisualizer


        parameters = {'g': [np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12]),np.array([13,14,15])],
                      'g2': [np.array([101,102,103]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])],
                      'X3': [np.array([201,202,203]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])],
                      'deltas': [np.array([201,202,203]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])],
                      'learning_rate': [np.array([201,202,203]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])]}
        msg_regx =  "^given invalid value name\, expect \['g'\, 'g2'\, 'tmp'\, 'deltas'\, 'learning_rate'\] \
got \[('g'[,]{0,1}[ ]{0,1}|'g2'[,]{0,1}[ ]{0,1}|'X3'[,]{0,1}[ ]{0,1}|'deltas'[,]{0,1}[ ]{0,1}|'learning_rate'[,]{0,1}[ ]{0,1}){5}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addRMSpropValues(parameters)

        parameters = {'g': [np.array([1,2,3]),np.array([4,5,6]),np.array([7,8,9]),np.array([10,11,12]),np.array([13,14,15])],
                      'g2': [np.array([101,102,103]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])],
                      'tmp': [np.array([101,102,103]),np.array([104,105,106]),np.array([107,108,109]),np.array([110,111,112]),np.array([113,114,115])],
                      'learning_rate': [np.array([201,202,203]),np.array([204,205,206]),np.array([207,208,209]),np.array([210,211,212]),np.array([213,214,215])]}
        msg_regx =  "^given invalid value name\, expect \['g'\, 'g2'\, 'tmp'\, 'deltas'\, 'learning_rate'\] \
got \[('g'[,]{0,1}[ ]{0,1}|'g2'[,]{0,1}[ ]{0,1}|'tmp'[,]{0,1}[ ]{0,1}|'learning_rate'[,]{0,1}[ ]{0,1}){4}\]$"
        with self.assertRaisesRegex(ValueError,msg_regx) as cm:
            cls.visualizer.addRMSpropValues(parameters)

        values = {'g': np.array([[1,2,3]]),
                  'g2': np.array([101,102,103]),
                  'tmp': np.array([201,202,203]),
                  'deltas': np.array([201,202,203]),
                  'learning_rate': 0.5,
                  }
        cls.visualizer.addRMSpropValues(values)
        self.assertEqual(len(cls.visualizer.logger.values), 183)
        init = 166
        for name, val in values.items():
            if name == 'learning_rate':
                self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '5.RMSprop/learning_rate')
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, 0.5)
            else:
                self.assertEqual(cls.visualizer.logger.values[count(init)].tag, '5.RMSprop/{}/max'.format(name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.max())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '5.RMSprop/{}/min'.format(name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, val.min())
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '5.RMSprop/{}/avg'.format(name))
                self.assertEqual(cls.visualizer.logger.values[count(next=False)].simple_value, np.mean(val))
                self.assertEqual(cls.visualizer.logger.values[count()].tag, '5.RMSprop/{}'.format(name))
                self.assertIsNotNone(cls.visualizer.logger.values[count(next=False)].histo)
            init=None

    def test_06_flush(self):
        cls = TestLearningVisualizer
        cls.visualizer.flush(getattr(self, 'step', 1))
        self.assertEqual(len(cls.visualizer.logger.values), 0)
        

    def test_07_loop(self):
        for i in range(10):
            with self.subTest(i=i):
                self.step = i+1
                self.test_01_addInputImages()
                self.test_02_addTargetNetworkParameters()
                self.test_03_addNetworkParameters()
                self.test_04_addGetQUpdateValues()
                self.test_05_addRMSpropValues()
                self.test_06_flush()
