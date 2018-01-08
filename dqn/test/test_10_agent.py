import sys
import unittest
from unittest.mock import *
import numpy as np

sys.path.append('../')
sys.path.append('../../')
from agents import DQNAgent
from utils import Namespace, get_random
from scale import ScaleImage
from testutils import *

class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        import sys
        from time import sleep
        
        sleep(1)
        
        from config import get_opt
        sys.argv += ['--backend', 'pytorch',
                     '--env', 'breakout',
                     '--logdir', '/tmp',
                     '--gpu', '0'
                     ]
        self.args = get_opt()
        self.args.actions = [0,1,3,4]
        self.args.n_actions = 4
        get_random('numpy', 1)

    @patch('common.get_preprocess', return_value=ScaleImage)
    @patch('common.create_networks', return_value=(MagicMock(),MagicMock()))
    def test_00_init(self, mock_create_networks, mock_get_preprocess):
        
        agent = DQNAgent(self.args)
        
        mock_create_networks.assert_called_once_with(self.args)
        
        agent.trainer.update_target_network.assert_called_once_with()

        self.assertTrue(hasattr(agent, 'network'))
        self.assertIsNotNone(agent.network)
        self.assertEqual(agent.network, mock_create_networks.return_value[0])

        self.assertTrue(hasattr(agent, 'trainer'))
        self.assertIsNotNone(agent.trainer)
        self.assertEqual(agent.trainer, mock_create_networks.return_value[1])
        

        mock_get_preprocess.assert_called_once_with(self.args.preproc)

        self.assertTrue(hasattr(agent, 'preproc'))
        self.assertIsNotNone(agent.preproc)
        self.assertIsInstance(agent.preproc, ScaleImage)

        from transition_table import TransitionTable

        self.assertTrue(hasattr(agent, 'transitions'))
        self.assertIsNotNone(agent.transitions)
        self.assertIsInstance(agent.transitions, TransitionTable)


    def test_01_greedy(self):

        agent = DQNAgent(self.args)

        with patch.object(agent.network, 'forward') as mock_forward:
            
            state = np.zeros((4, 84,84))
            state[0] += 1
            state[1] += 2
            state[2] += 3
            state[3] += 4

            # The _greedy() calls network.forward() method once for predict w.r.t. state.
            # If forward() returns [[0,1,3,2]], _greedy() returns 2
            # the index value of the element with the maximum return value of forward()

            mock_forward.return_value=[[0,1,3,2]]
            action = agent._greedy(state)
            
            mock_forward.assert_called_once_with(state)
            self.assertEqual(action, 2)

            # When maximum value exists in multiple elements of return of forward()
            # _greedy() choces a index of max value element randomly.
            # So next test, forward() returns value that max value exists
            # in the three elements with indexes 0, 2 and 3 follows:
            #    [[3,1,3,3]]
            # After that, _greedy() may collects indexes with max value into a vector.
            #    [0,2,3]
            # And it choices an index of vector using random().
            # In this test, random() will returns 1.
            # Therefore, I expects that _greedy() get 2 from a vector and return it.

            mock_forward.reset_mock()
            with patch.object(agent.random, 'random', return_value=1) as mock_random:
                mock_forward.return_value=[[3,1,3,3]]
                action = agent._greedy(state)

                mock_forward.assert_called_once_with(state)
                mock_random.assert_called_once_with(3)
                self.assertEqual(action, 2)


    def test_02_eGreedy(self):
    
        state = np.zeros((4, 84,84))
        state[0] += 1
        state[1] += 2
        state[2] += 3
        state[3] += 4

        agent = DQNAgent(self.args)

        @patch.object(agent, '_greedy')
        @patch.object(agent.random, 'uniform')
        @patch.object(agent.random, 'random')
        def test_func(mock_random, mock_uniform, mock_greedy):

            # The _eGreedy() implements ε-greedy.
            # When the argument testing_ep is None and
            # numStep still less or equal than 50000,
            # ε is 1 fixedly.
            # In this time, _eGreedy() never call _greedy()
            # and return the action index
            # will get randomly between 0 and number of actions
            # that depending on the game.

            ret_ramdom = [1,3,0,4,3]
            ret_uniform = [1.0,0.5,0.2,1.0,0.6]

            for numStep in [1, 100, 30000, 50000]:

                agent.numStep = numStep
                mock_random.side_effect = ret_ramdom
                mock_uniform.side_effect = ret_uniform

                for ret_rand, ret_uni in zip(mock_random, ret_uniform):
                    with self.subTest('numStep:{} ε:{}'.format(agent.numStep, agent.ep)):
                        mock_random.reset_mock()
                        mock_uniform.reset_mock()

                        action = agent._eGreedy(state, None)

                        self.assertEqual(agent.ep, 1)
                        mock_greedy.assert_not_called()
                        mock_uniform.assert_not_called()
                        mock_random.assert_called_once_with(0, 4)
                        self.assertEqual(action, ret_rand)

        
            # When the argument testing_ep is None and numSteps is 50001,
            # ε will start to be decreace lineary.
            # In this step, ε is 0.999999099999999918608750704152 on lua,
            # If ε is calculated the same as DQN 3.0,
            # it will be the same 0.99999909999999999918608750704152 on Python as well.
            # when ε is computed the same as DQN3.0. 
            # In this time, when uniform() returns random value that grearter or equal than
            # ε, the _eGreedy() will call _greedy().
            # but when uniform() returns value that less than ε,
            # the _eGreedy() will not call _greedy()
            # in actuality, uniform() never return it because
            # what means this test is contains boundary test.

            mock_random.side_effect = None
            mock_uniform.side_effect = None
            agent.numSteps = 50001
            for i, uniform in enumerate([0.99999909999909990000000,
                                         1,
                                         0.999999099999999918608750704152]):
                with self.subTest('numStep:{} uniform:{}'.format(agent.numSteps, uniform)):
                    mock_random.reset_mock()
                    mock_random.return_value = 2
                    mock_uniform.reset_mock()
                    mock_uniform.return_value = uniform
                    mock_greedy.reset_mock()
                    mock_greedy.return_value = 1

                    action = agent._eGreedy(state, None)

                    self.assertEqual('{:.30f}'.format(0.999999099999999918608750704152), 
                                     '{:.30f}'.format(agent.ep))
                    if i == 0:
                        mock_greedy.assert_not_called()
                        mock_uniform.assert_called_with()
                        mock_random.assert_called_once_with(0, 4)
                        self.assertEqual(action, 2)
                    else:
                        mock_greedy.assert_called_once_with(state)
                        mock_uniform.assert_called_with()
                        mock_random.assert_not_called()
                        self.assertEqual(action, 1)

            # Well, the numStep is increasing to 1049999,
            # ε is 0.100000900000000003675637572087 on lua.
            # If ε is calculated the same as DQN 3.0,
            # it will be the same 0.100000900000000003675637572087 on Python as well.
            # In this time, when uniform() returns random value that grearter or equal
            # than ε, the _eGreedy() will call _greedy().
            # but when uniform() returns value that less than ε,
            # the _eGreedy() will not call _greedy()

            agent.numSteps = 1049999
            for i, uniform in enumerate([0.10000000000000000000,
                                         0.100000900000000003675637572087,
                                         0.1000010]):
                with self.subTest('numStep:{} uniform:{}'.format(agent.numSteps, uniform)):
                    mock_random.reset_mock()
                    mock_random.return_value = 2
                    mock_uniform.reset_mock()
                    mock_uniform.return_value = uniform
                    mock_greedy.reset_mock()
                    mock_greedy.return_value = 1

                    action = agent._eGreedy(state, None)

                    self.assertEqual('{:.30f}'.format(0.100000900000000003675637572087), 
                                     '{:.30f}'.format(agent.ep))

                    if i == 0:
                        mock_greedy.assert_not_called()
                        mock_uniform.assert_called_with()
                        mock_random.assert_called_once_with(0, 4)
                        self.assertEqual(action, 2)
                    else:
                        mock_greedy.assert_called_once_with(state)
                        mock_uniform.assert_called_with()
                        mock_random.assert_not_called()
                        self.assertEqual(action, 1)


            # If argument testing_ep is not None that means during test step.
            # In this time, ε does not decrease and use testing_ep(0.05) fixedly
            # even if numSteps increasing 900000, 1000000 and 1040000.
            for i, test_val in enumerate(zip([0.04,
                                              0.05,
                                              0.06],
                                             [ 900000,
                                              1000000,
                                              1040000])):
                uniform, numSteps = test_val
                agent.numSteps = numSteps
                with self.subTest('numStep:{} uniform:{}'.format(agent.numSteps, uniform)):

                    mock_random.reset_mock()
                    mock_random.return_value = 2
                    mock_uniform.reset_mock()
                    mock_uniform.return_value = uniform
                    mock_greedy.reset_mock()
                    mock_greedy.return_value = 1

                    action = agent._eGreedy(state, 0.05)

                    self.assertEqual(0.05, agent.ep)

                    if i == 0:
                        mock_greedy.assert_not_called()
                        mock_uniform.assert_called_with()
                        mock_random.assert_called_once_with(0, 4)
                        self.assertEqual(action, 2)
                    else:
                        mock_greedy.assert_called_once_with(state)
                        mock_uniform.assert_called_with()
                        mock_random.assert_not_called()
                        self.assertEqual(action, 1)
            
        test_func()
        
    def test_03_qLearnMinibatch(self):
        
        self.args.learn_start = 3
        self.args.update_freq = 2
        self.args.clipe_reward = 1
        self.args.rescale_r = True
        self.args.prog_freq = 10
        self.args.target_q = 1000
        self.args.steps = 5000
        self.args.add_summaries_all = False

        agent = DQNAgent(self.args)

        s = np.ones((32,4,84,84), dtype=np.float32)
        a = np.ones((32,), dtype=np.uint8)
        r = np.ones((32,), dtype=np.int) + 100
        s2 = s + 200

        @patch.object(agent, 'transitions')
        @patch.object(agent, 'trainer')
        def test(mock_trainer,
                 mock_transitions):

            # In before the prog_freq, the agent calls only qLearnMinibatch()
            for numSteps in range(1,10):
                print(numSteps)
                expected=[
                    s+numSteps,
                    a+numSteps,
                    r+numSteps,
                    s2+numSteps,
                    np.zeros((32,),dtype=np.uint8)
                ]
                mock_transitions.reset_mock()
                mock_transitions.sample.return_value = expected
                mock_trainer.reset_mock()
                agent.numSteps = numSteps
                agent._qLearnMinibatch()

                self.assertEqual(mock_transitions.sample.call_count, 1)
                self.assertEqual(len(mock_trainer.method_calls), 1)
                calls = mock_trainer.method_calls
                print([call[0] for call in calls])
                assert_call_equal(calls[0], ('qLearnMinibatch', (expected, False), {}))

            # At the prog_freq, the agent calls qLearnMinibatch() and
            # add_learning_summaries(), and add_learning_summaries()
            # will called after qLearnMinibatch().
            for numSteps in range(10,11):
                print(numSteps)
                expected=[
                    s+numSteps,
                    a+numSteps,
                    r+numSteps,
                    s2+numSteps,
                    np.zeros((32,),dtype=np.uint8)
                ]
                mock_transitions.reset_mock()
                mock_transitions.sample.return_value = expected
                mock_trainer.reset_mock()
                agent.numSteps = numSteps
                agent._qLearnMinibatch()

                self.assertEqual(mock_transitions.sample.call_count, 1)
                self.assertEqual(len(mock_trainer.method_calls), 2)
                calls = mock_trainer.method_calls
                print([call[0] for call in calls])
                assert_call_equal(calls[0], ('qLearnMinibatch', (expected, True), {}))
                assert_call_equal(calls[1], ('add_learning_summaries', (numSteps,), {}))

        test()

    def test_04_perceive_training_mode1(self):
        
        # In training mode, agent call transitions.add(), trainer.qLearnMinibatch(),
        # and agent.sample_validation_data() at the specified intervals.
        # In addition, numSteps is increased.
        
        from scale import ScaleImage
        
        scale = ScaleImage(84, 84, self.args)
        
        self.args.learn_start = 1
        self.args.update_freq = 2
        self.args.clipe_reward = 1
        self.args.rescale_r = True
        self.args.prog_freq = 100
        self.args.target_q = 1000
        self.args.steps = 5000

        agent = DQNAgent(self.args)

        state = np.zeros((210,160,3), dtype=np.float32)

        org_sample_validation_data = agent._sample_validation_data
        @patch.object(agent, '_eGreedy', side_effect=[1,2,3,4,5,6])
        @patch.object(agent, 'transitions')
        @patch.object(agent, 'trainer')
        @patch.object(agent, '_sample_validation_data',
                             wraps=lambda :org_sample_validation_data())
        def test_training_mode(mock_sample_validation_data,
                          mock_trainer,
                          mock_transitions,
                          mock_eGreedy):
            
            transitions_sample_return = [np.zeros((32, 4,84,84)),
                                                    np.zeros((32,)),
                                                    np.zeros((32,)),
                                                    np.zeros((32, 4,84,84)),
                                                    np.zeros((32,))]
            mock_transitions.sample.return_value = transitions_sample_return

            # Step 1 : numStep=0
            # Nothing happens at this step
            agent.perceive(state+1, 0, False, testing=False, testing_ep=None)
            
            mock_transitions.add.assert_not_called()
            mock_trainer.qLearnMinibatch.assert_not_called()
            mock_sample_validation_data.assert_not_called()
  
            self.assertEqual(agent.r_max, 1)
            self.assertIsNotNone(agent.lastState)
            self.assertTrue(np.all(agent.lastState == scale.forward(state+1)))
            self.assertIsNotNone(agent.lastAction)
            self.assertEqual(agent.lastAction, 1)
            self.assertIsNotNone(agent.lastTerminal)
            self.assertFalse(agent.lastTerminal)
            self.assertEqual(agent.numSteps, 1)

            # Step 2 : numStep=1
            # In this step, agent calls only transitions.add().
            agent.perceive(state+2, 10, False, testing=False, testing_ep=None)

            # reward is clipped before call to add()
            self.assertEqual(mock_transitions.add.call_count, 1)
            arg, _ = mock_transitions.add.call_args
            s, a, r, t =arg
            self.assertTrue(np.all(s == scale.forward(state+1)))
            self.assertEqual(a, 1)
            self.assertEqual(r, 1)
            self.assertFalse(t)
            mock_trainer.qLearnMinibatch.assert_not_called()
            mock_sample_validation_data.assert_not_called()

            # The r_max uses reward rescaling but it is a maximum 'clipped' rewards [-1, 1].
            # Therefor, r_max is up to 1 that means rescaled rewards are up to 1 at all time.
            # In the breakout, the rewards that using at learning are 0 or 1 at all time,
            # because rewards of breakout are only positive integer value.
            self.assertEqual(agent.r_max, 1)
            self.assertIsNotNone(agent.lastState)
            self.assertTrue(np.all(agent.lastState == scale.forward(state+2)))
            self.assertIsNotNone(agent.lastAction)
            self.assertEqual(agent.lastAction, 2)
            self.assertIsNotNone(agent.lastTerminal)
            self.assertFalse(agent.lastTerminal)
            self.assertEqual(agent.numSteps, 2)


            # Step 3 : numStep=2
            # In this step, agent calls  transitions.add() and
            # sample_validation_data()
            agent.perceive(state+3, -10, False, testing=False, testing_ep=None)

            self.assertEqual(mock_transitions.add.call_count, 2)
            arg, _ = mock_transitions.add.call_args
            s, a, r, t =arg
            self.assertTrue(np.all(s == scale.forward(state+2)))
            self.assertEqual(a, 2)
            self.assertEqual(r, -1)
            self.assertFalse(t)
            mock_sample_validation_data.assert_called_once_with()
            self.assertEqual(mock_trainer.qLearnMinibatch.call_count, 1)
            args, _ = mock_trainer.qLearnMinibatch.call_args
            for name, expect_v, v in zip(['s', 'a', 'r', 's2', 't'],
                                          transitions_sample_return,
                                          args[0]):
                with self.subTest(name):
                    self.assertTrue(np.all(expect_v == v))

            old_valids = {}
            for v_name in ['valid_s',
                           'valid_a',
                           'valid_r',
                           'valid_s2',
                           'valid_term']:
                with self.subTest(v_name):
                    self.assertTrue(hasattr(agent, v_name))
                    self.assertIsNotNone(getattr(agent, v_name))
                    old_valids[v_name] = getattr(agent, v_name)

            self.assertEqual(agent.r_max, 1)
            self.assertIsNotNone(agent.lastState)
            self.assertTrue(np.all(agent.lastState == scale.forward(state+3)))
            self.assertIsNotNone(agent.lastAction)
            self.assertEqual(agent.lastAction, 3)
            self.assertIsNotNone(agent.lastTerminal)
            self.assertFalse(agent.lastTerminal)
            self.assertEqual(agent.numSteps, 3)


            # Step 4 : numStep=3
            # In this step, agent calls only transitions.add() and
            # trainer.qLearnMinibatch().
            agent.perceive(state+4, 20, False, testing=False, testing_ep=None)

            self.assertEqual(mock_transitions.add.call_count, 3)
            arg, _ = mock_transitions.add.call_args
            s, a, r, t =arg
            self.assertTrue(np.all(s == scale.forward(state+3)))
            self.assertEqual(a, 3)
            self.assertEqual(r, 1)
            self.assertFalse(t)
            self.assertEqual(mock_sample_validation_data.call_count, 1)
            self.assertEqual(mock_trainer.qLearnMinibatch.call_count, 1)

            self.assertEqual(agent.r_max, 1)
            self.assertIsNotNone(agent.lastState)
            self.assertTrue(np.all(agent.lastState == scale.forward(state+4)))
            self.assertIsNotNone(agent.lastAction)
            self.assertEqual(agent.lastAction, 4)
            self.assertIsNotNone(agent.lastTerminal)
            self.assertFalse(agent.lastTerminal)
            self.assertEqual(agent.numSteps, 4)

        test_training_mode()
        
    def test_04_perceive_training_mode2(self):
        
        # Let's test the training mode until the number of max training steps
        # that specified to self.args.steps.
        # sample_validation_data(), qLearnMinibatch() and
        # update_target_network() are called at the specified frequence.

        from scale import ScaleImage
        
        scale = ScaleImage(84, 84, self.args)
        
        self.args.learn_start = 1
        self.args.update_freq = 2
        self.args.clipe_reward = 1
        self.args.rescale_r = True
        self.args.prog_freq = 100
        self.args.target_q = 1000
        self.args.steps = 5000

        agent = DQNAgent(self.args)

        state = np.zeros((210,160,3), dtype=np.float32)

        org_sample_validation_data = agent._sample_validation_data
        @patch.object(agent, '_eGreedy', return_value=0)
        @patch.object(agent, 'transitions')
        @patch.object(agent, 'trainer')
        @patch.object(agent, '_sample_validation_data',
                             wraps=lambda :org_sample_validation_data())
        def test_training_mode(mock_sample_validation_data,
                          mock_trainer,
                          mock_transitions,
                          mock_eGreedy):
            
            
            transitions_sample_return = [np.zeros((32, 4,84,84)),
                                                    np.zeros((32,)),
                                                    np.zeros((32,)),
                                                    np.zeros((32, 4,84,84)),
                                                    np.zeros((32,))]
            mock_transitions.sample.return_value = transitions_sample_return

            sample_validation_data_count = mock_sample_validation_data.call_count
            qLearnMinibatch_call_count = mock_trainer.qLearnMinibatch.call_count
            _add_learning_summaries_count = mock_trainer.add_learning_summaries.call_count
            update_target_network_count = mock_trainer.update_target_network.call_count
            agent.numSteps = 0

            old_valids = {}
            for step in range(self.args.steps):
                if step % 1000 == 0:
                    print('training mode loop test step {}/{}'.format(step, self.args.steps))

                agent.perceive(state+(step/255.0), 20, False, testing=False, testing_ep=None)

                # _sample_validation_data() is called only once.
                if sample_validation_data_count != mock_sample_validation_data.call_count:
                    self.assertEqual(mock_sample_validation_data.call_count, 1)
                    self.assertEqual(agent.numSteps-1, self.args.learn_start+1)
                    sample_validation_data_count = mock_sample_validation_data.call_count

                    for v_name in ['valid_s',
                                  'valid_a',
                                  'valid_r',
                                  'valid_s2',
                                  'valid_term']:
                            self.assertTrue(hasattr(agent, v_name))
                            self.assertIsNotNone(getattr(agent, v_name))
                            old_valids[v_name] = getattr(agent, v_name)
                elif agent.numSteps-1 > self.args.learn_start+1:
                    # the valid samples are never changed.
                    for v_name in ['valid_s',
                                  'valid_a',
                                  'valid_r',
                                  'valid_s2',
                                  'valid_term']:
                            self.assertTrue(hasattr(agent, v_name))
                            self.assertIsNotNone(getattr(agent, v_name))
                            self.assertTrue(np.all(getattr(agent, v_name) == old_valids[v_name]))

                # qLearnMinibatch() is called in numSteps is a multiple of update_freq before numSteps increases.
                if mock_trainer.qLearnMinibatch.call_count != qLearnMinibatch_call_count:
                    self.assertTrue((agent.numSteps-1) % self.args.update_freq == 0)
                    qLearnMinibatch_call_count = mock_trainer.qLearnMinibatch.call_count

                # add_learning_summaries_count() is called in numSteps is a multiple of prog_freq before numSteps increases.
                if mock_trainer.add_learning_summaries.call_count != _add_learning_summaries_count:
                    self.assertTrue((agent.numSteps-1) % self.args.prog_freq == 0)
                    _add_learning_summaries_count = mock_trainer.add_learning_summaries.call_count

                # update_target_network() is called in the next step where numSteps is a multiple of target_q after numSteps increases.
                if mock_trainer.update_target_network.call_count != update_target_network_count:
                    self.assertTrue(agent.numSteps % self.args.target_q == 1)
                    update_target_network_count = mock_trainer.update_target_network.call_count 

            self.assertEqual(mock_trainer.qLearnMinibatch.call_count, 2499)
            self.assertEqual(mock_trainer.add_learning_summaries.call_count, 49)
            self.assertEqual(mock_trainer.update_target_network.call_count, 5)

        test_training_mode()
    
    def test_04_perceive_testing_mode1(self):
        
        # In testing mode, agent never call transitions.add(), trainer.qLearnMinibatch(),
        # trainer.update_target_network(), and agent.sample_validation_data().
        # In addition, numSteps is not increased.
        
        from scale import ScaleImage
        
        scale = ScaleImage(84, 84, self.args)
        
        self.args.learn_start = 1
        self.args.update_freq = 2
        self.args.clipe_reward = 1
        self.args.rescale_r = True
        self.args.prog_freq = 100
        self.args.target_q = 1000
        self.args.steps = 5000

        agent = DQNAgent(self.args)

        state = np.zeros((210,160,3), dtype=np.float32)

        org_sample_validation_data = agent._sample_validation_data
        @patch.object(agent, '_eGreedy', return_value=0)
        @patch.object(agent.transitions, 'add')
        @patch.object(agent, 'trainer')
        @patch.object(agent, '_sample_validation_data',
                             wraps=lambda :org_sample_validation_data())
        def test_testing_mode(mock_sample_validation_data,
                          mock_trainer,
                          mock_transitions_add,
                          mock_eGreedy):

            agent.numSteps = 0

            for step in range(self.args.steps):
                if step % 1000 == 0:
                    print('training mode loop test step {}/{}'.format(step, self.args.steps))
                s = state+(step/255.0)/255.0
                agent.perceive(s, 20, False, testing=True, testing_ep=0.5)

                mock_sample_validation_data.assert_not_called()
                mock_transitions_add.assert_not_called()
                mock_trainer.qLearnMinibatch.assert_not_called()
                mock_trainer.update_target_network.assert_not_called()
                
                args, args2 = mock_eGreedy.call_args
                self.assertTrue(np.all(args[0] == agent.transitions.get_recent()))
                self.assertEqual(args[1], 0.5)


                self.assertEqual(agent.numSteps, 0)
 
        test_testing_mode()
        
        
    def test_05_maximization_option(self):
    
        from alewrap_py.game_screen import GameScreen, NopGameScreen

        zero = np.zeros((210, 160, 3), dtype=np.float32)
        expected = [(zero + i) / 255.0 for i in range(10, -1, -1)]

        def inject_preprocess(rawstate):
            nonlocal return_s
            s = org_preprocess(rawstate)
            return_s.append(s)
            return s


        self.args.maximization = "agent"
        agent = DQNAgent(self.args)
        self.assertIsInstance(agent._screen, GameScreen)
        org_preprocess = agent._preprocess
        return_s = []
        def inject_preprocess(rawstate):
            nonlocal return_s
            s = org_preprocess(rawstate)
            return_s.append(s)
            return s
        @patch.object(agent, '_preprocess', wraps=inject_preprocess)
        def test(mock_preprocess):
            for testing, testing_ep in [(False, None), (True, 0.05)]:
                for s in expected:
                    agent.perceive(s, 0, False, testing=testing, testing_ep=testing_ep)
                                         # 1st element, max is equal the expected[0]
                                         # because fillBuffer has only it.
                                         # 2nd element, fillBuffer has expected[0]
                                         # and expected[1] and expected[0] is greater than
                                         # expected[1], because of this, 2nd～ are expected[0]～[8]
                                         # because of this, 2nd～ are expected[0]～[8]
                for s, ex_s in zip(return_s, [expected[0], *expected[:9]]):
                    print('s',s[0][:3])
                    print('ex_s', ex_s[0][0])
                    assert_equal(s[0][:3], ex_s[0][0], use_float_equal=True, verbose=1)
        test()

        self.args.maximization = "env"
        agent = DQNAgent(self.args)
        self.assertIsInstance(agent._screen, NopGameScreen)
        org_preprocess = agent._preprocess
        return_s = []
        @patch.object(agent, '_preprocess', wraps=inject_preprocess)
        def test(mock_preprocess):
            for testing, testing_ep in [(False, None), (True, 0.05)]:
                for s in expected:
                    agent.perceive(s, 0, False, testing=testing, testing_ep=testing_ep)
                                         # 1st element, max is equal the expected[0]
                                         # because fillBuffer has only it.
                                         # 2nd element, fillBuffer has expected[0]
                                         # and expected[1] and expected[0] is greater than
                                         # expected[1], because of this, 2nd～ are expected[0]～[8]
                                         # because of this, 2nd～ are expected[0]～[8]
                for s, ex_s in zip(return_s, expected):
                    print('s',s[0][:3]*255.0)
                    print('ex_s', ex_s[0][0]*255.0)
                    assert_equal(s[0][:3], ex_s[0][0], use_float_equal=True, verbose=1)
        test()


    def test_06_regreedy(self):


        state = np.zeros((210,160,3), dtype=np.float32)
        self.args.replay_memory = 8
        self.args.bufferSize = 2
        self.args.valid_size = 1
        self.args.hist_len = 4
        self.args.input_dims     = (self.args.hist_len * self.args.ncols, self.args.input_width, self.args.input_height)
        self.args.minibatch_size = 1
        self.args.ep_endt = self.args.replay_memory
        self.args.ep_endt_restarted
        self.args.ep_restart = (self.args.ep_start-self.args.ep_end) / 2.0
        self.args.ep_endt_restarted = self.args.replay_memory / 2.0
        self.args.use_regreedy = 0
        self.args.regreedy_threshold = 3
        self.args.regreedy_rate = 0.9
        self.args.regreedy_ema_momemtum = 0.9
        self.args.gpu = 0
        self.args.update_freq = 1000

        # NOT USE REGREEDY
        from agents import DefaultGreedyMethod, ReGreedyMethod
        agent = DQNAgent(self.args)

        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_start)

        agent.perceive(state, 1, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 1)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 11)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, True, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 0)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 10)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 20)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 30)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 40)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 50)
        self.assertEqual(agent.ep, self.args.ep_start)

        print('---not use regreedy ----',agent.learn_start)
        agent.numSteps = 50001
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.ep, self.args.ep_end)

        agent.decide_regreedy(100)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)

        agent.decide_regreedy(90)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)

        agent.decide_regreedy(70)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)

        agent.decide_regreedy(90)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)

        agent.decide_regreedy(90)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)

        agent.decide_regreedy(90)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)


        # USE REGREEDY
        self.args.use_regreedy = 1
        self.args.regreedy_threshold = 3
        self.args.regreedy_rate = 0.9
        self.args.regreedy_ema_momemtum = 0.5
        self.args.ep_restart = (self.args.ep_start-self.args.ep_end) / 2.0
        self.args.ep_endt_restarted = int(self.args.replay_memory / 2.0)
        agent = DQNAgent(self.args)

        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_start)

        agent.perceive(state, 1, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 1)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 11)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, True, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 0)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 10)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 20)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 30)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 40)
        self.assertEqual(agent.ep, self.args.ep_start)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.score, 50)
        self.assertEqual(agent.ep, self.args.ep_start)

        print('---use regreedy ----',agent.learn_start)
        agent.numSteps = 50001
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        print('perceive')
        print('perceive')
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertEqual(agent.ep, self.args.ep_end)

        agent.decide_regreedy(100)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)
        self.assertEqual(agent.max_eval_average_score, 50)
        assert_equal(list(agent.moving_average), [50])

        agent.decide_regreedy(90)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)
        self.assertEqual(agent.max_eval_average_score, 70)
        assert_equal(list(agent.moving_average), [50,70])

        agent.decide_regreedy(70)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)
        self.assertEqual(agent.max_eval_average_score, 70)
        assert_equal(list(agent.moving_average), [50,70, 70])

        agent.decide_regreedy(50)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)
        self.assertEqual(agent.max_eval_average_score, 70)
        assert_equal(list(agent.moving_average), [50,70,70,60])

        agent.decide_regreedy(30)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)
        self.assertEqual(agent.max_eval_average_score, 70)
        assert_equal(list(agent.moving_average), [70,70,60,45])

        agent.decide_regreedy(10)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, DefaultGreedyMethod)
        self.assertEqual(agent.ep, self.args.ep_end)
        self.assertEqual(agent.max_eval_average_score, 70)
        assert_equal(list(agent.moving_average), [70,60,45,27.5])


        agent.decide_regreedy(10)
        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, ReGreedyMethod)
        self.assertEqual(agent.max_eval_average_score, 70)
        self.assertEqual(agent.regreedy_score, 63)
        self.assertEqual(agent.ep, self.args.ep_restart)
        self.assertEqual(agent.score, 200)
        assert_equal(list(agent.moving_average), [60, 45, 27.5, 18.75])

        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, ReGreedyMethod)
        self.assertEqual(agent.max_eval_average_score, 70)
        self.assertEqual(agent.regreedy_score, 63)
        self.assertEqual(agent.score, 210)
        assert_equal(agent.ep, 0.45-(0.45-0.1)/3.0)
        assert_equal(list(agent.moving_average), [60, 45, 27.5, 18.75])

        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, ReGreedyMethod)
        self.assertEqual(agent.max_eval_average_score, 70)
        self.assertEqual(agent.regreedy_score, 63)
        self.assertEqual(agent.score, 220)
        assert_equal(agent.ep, 0.45-(0.45-0.1)/3.0*2.0)
        assert_equal(list(agent.moving_average), [60, 45, 27.5, 18.75])

        agent.perceive(state, 10, False, testing=False, testing_ep=None)
        self.assertIsInstance(agent._do_greedy, ReGreedyMethod)
        self.assertEqual(agent.max_eval_average_score, 70)
        self.assertEqual(agent.regreedy_score, 63)
        self.assertEqual(agent.score, 230)
        assert_equal(agent.ep, agent.ep_end)
        assert_equal(list(agent.moving_average), [60, 45, 27.5, 18.75])
