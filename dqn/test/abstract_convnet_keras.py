import sys
import os
import re
from time import sleep
from collections import deque
import importlib
import unittest
from unittest.mock import *
import numpy as np

from torch.utils.serialization import load_lua

if sys.path.count('../../') == 0:
    sys.path.append('../../')
if sys.path.count('../') == 0:
    sys.path.append('../')

from utils import get_random, Namespace
from testutils import *

class AbstractTestKerasConvnet(object):


    def setUp(self):

        np.set_printoptions(threshold=np.inf)
        get_random('pytorch', 1)
        
        self.backend = self.backend()
        os.environ['KERAS_BACKEND'] =self.backend
        
        """
        if self.backend == 'cntk':
            from cntk.debugging import set_computation_network_trace_level
            set_computation_network_trace_level(1)
        """
    def test_01_create_model(self):
        from convnet_keras import KerasConvnet
        from config import get_opt
        sys.argv += ['--backend', self.backend, '--env', 'breakout', '--debug', '--gpu', '-1', '--random_type', 'pytorch']
        args = get_opt()
        args.actions   = [0,1,3,4]
        args.n_actions   = 4

        get_random().manualSeed(1)
        with get_stdout() as stdout:
            network = KerasConvnet(args, 'network')

        expected = [
            "initializer : torch nn layers default (w:like torch nn layers, b:like torch nn layers)",
            "_________________________________________________________________",
            "Layer (type)                 Output Shape              Param #   ",
            "=================================================================",
            "s (InputLayer)               (None, 4, 84, 84)         0         ",
            "_________________________________________________________________",
            "premute (Permute)            (None, 84, 84, 4)         0         ",
            "_________________________________________________________________",
            "Conv1 (Conv2D)               (None, 20, 20, 32)        8224      ",
            "_________________________________________________________________",
            "relu1 (Activation)           (None, 20, 20, 32)        0         ",
            "_________________________________________________________________",
            "Conv2 (Conv2D)               (None, 9, 9, 64)          32832     ",
            "_________________________________________________________________",
            "relu2 (Activation)           (None, 9, 9, 64)          0         ",
            "_________________________________________________________________",
            "Conv3 (Conv2D)               (None, 7, 7, 64)          36928     ",
            "_________________________________________________________________",
            "relu3 (Activation)           (None, 7, 7, 64)          0         ",
            "_________________________________________________________________",
            "flatten (Flatten)            (None, 3136)              0         ",
            "_________________________________________________________________",
            "Linear (Dense)               (None, 512)               1606144   ",
            "_________________________________________________________________",
            "relu4 (Activation)           (None, 512)               0         ",
            "_________________________________________________________________",
            "q_all (Dense)                (None, 4)                 2052      ",
            "=================================================================",
            "Total params: 1,686,180",
            "Trainable params: 1,686,180",
            "Non-trainable params: 0",
            "_________________________________________________________________",
        ]

        #assert_equal(stdout.outputs, expected)
        
        
        # The parameters with initialized are exactly the same as those of Pythorch network.
        params = network.model.get_weights()
        
        get_random().manualSeed(1)

        from pytorch_extensions import setup
        from convnet_pytorch import PyTorchConvnet
        args.backend='pytorch_legacy'
        setup(args)
        pytorch_network = PyTorchConvnet(args, "network")

        py_params, _ = pytorch_network.model.parameters()
        for p, py in zip(params, [p.numpy() for p in py_params]):
            print('******* cntk {}******')
            print(p.T)
            print('******* pytorch {}******')
            print(py)
            assert_equal(p.T, py, use_float_equal=True, verbose=10)


    def test_02_forward(self):
        sleep(1)
        from convnet_keras import KerasConvnet

        from config import get_opt
        sys.argv += ['--backend', self.backend, '--env', 'breakout', '--debug', '--gpu', '-1', '--random_type', 'pytorch']
        args = get_opt()
        args.actions   = [0,1,3,4]
        args.n_actions   = 4

        network = KerasConvnet(args, 'network')

        s = np.ones((1,4,84,84), dtype=np.float32) / 255.0
        
        predict = network.forward(s)
        
        assert isinstance(predict, np.ndarray), print(type(predict))
        assert_equal(predict.shape, (1,4))

    def test_03_save_load(self):
        sleep(1)
        from convnet_keras import KerasConvnet

        from config import get_opt
        sys.argv += ['--backend', self.backend, '--env', 'breakout', '--debug', '--gpu', '-1', '--random_type', 'pytorch']
        args = get_opt()
        args.actions   = [0,1,3,4]
        args.n_actions   = 4

        network = KerasConvnet(args, 'network')

        params = network.model.get_weights()
        for p in params:
            p.fill(1)
        network.model.set_weights(params)

        network.save('/tmp/saved_network.dat')

        params = network.model.get_weights()
        for p in params:
            p.fill(2)
        network.model.set_weights(params)

        network.load('/tmp/saved_network.dat')

        params = network.model.get_weights()
        for p in params:
            assert_equal(p, np.ones_like(p))

    def test_04_trainer_init1(self):
        sleep(1)
        self._test_trainer_init('DQN3.0')

    #def test_05_trainer_init2(self):
    #    self._test_trainer_init('built-in')

    def _test_trainer_init(self, optimizer):
        from keras import backend as K
        import common

        from config import get_opt
        sys.argv += ['--backend', self.backend, '--env', 'breakout', '--debug', '--gpu', '-1', '--random_type', 'pytorch']
        args = get_opt()
        args.actions   = [0,1,3,4]
        args.n_actions   = 4

        network, trainer = common.create_networks(args)

        assert trainer.func_getQUpdate is not None
        assert isinstance(trainer.func_getQUpdate, K.Function)
        assert trainer.func_qLearnMinibatch is not None
        assert isinstance(trainer.func_qLearnMinibatch, K.Function)
        assert trainer.func_qLearnMinibatch_for_summary is not None
        assert isinstance(trainer.func_qLearnMinibatch_for_summary, K.Function)
        assert trainer.func_update_target_network is not None
        assert isinstance(trainer.func_update_target_network, K.Function)

        assert trainer.inputs is not None
        assert len(trainer.inputs) == 5

        assert trainer.getQUpdate_ops is not None
        assert trainer.RMSprop_ops is not None



    def test_06_update_target_network1(self):
        sleep(1)
        import common
        from config import get_opt
        sys.argv += ['--backend', self.backend, '--env', 'breakout', '--debug', '--gpu', '-1', '--random_type', 'pytorch']
        args = get_opt()
        args.actions   = [0,1,3,4]
        args.n_actions   = 4

        network, trainer = common.create_networks(args)

        params = trainer.network.model.get_weights()
        for p in params:
            p.fill(1)
        trainer.network.model.set_weights(params)
        
        params = trainer.target_network.model.get_weights()
        for p in params:
            p.fill(1)
        trainer.target_network.model.set_weights(params)

        assert_equal(trainer.target_network.model.get_weights(),
                     trainer.network.model.get_weights())

        params = trainer.network.model.get_weights()
        for p in params:
            p.fill(2)
        trainer.network.model.set_weights(params)

        assert_not_equal(trainer.target_network.model.get_weights(),
                         trainer.network.model.get_weights())

        trainer.update_target_network()

        assert_equal(trainer.target_network.model.get_weights(),
                     trainer.network.model.get_weights())

    def test_07_qLearnMinibatch_01(self):
        sleep(1)
        if self.backend == 'tensorflow':
            import tensorflow as tf
            with tf.device('/cpu:0'):
                self._test_07_qLearnMinibatch_01()
        else:
            self._test_07_qLearnMinibatch_01()

    def _test_07_qLearnMinibatch_01(self):

        from common import create_networks, get_extensions
        from convnet_keras import KerasConvnet
        from config import get_opt
        sys.argv += ['--backend', self.backend,
                     '--env', 'breakout',
                     '--logdir', '/tmp',
                     '--gpu', '0',
                     '--random_type', 'torch',
                     '--loss_function', 'DQN3.0',
                     '--optimizer', 'DQN3.0',
                     #'--tf_debug'
                     ]
        args = get_opt()
        from initenv import setup
        
        game_env, agent, game_actions, args = setup(args)
        trainer = agent.trainer

        Ex = get_extensions(args)

        print('OK-2')
        network, trainer = create_networks(args)
        print('OK-1')
        Ex.variable_initialization()

        print('OK')
        trainer.update_target_network()
        print('OK2')

        get_random().manualSeed(1)

        for step in range(50004, 50016, 4):

            fname = './dqn3.0_dump/cpu_batch_{:010d}.dat'.format(step)
            batch = load_lua(fname)
            
            s, a, r, s2, term = [v.numpy() for v in batch]
            s = s.reshape(32, 4, 84, 84)
            s2 = s2.reshape(32, 4, 84, 84)
            a -= 1
            a = a.astype(np.uint8)
            r = r.astype(np.int8)
            term = term.astype(np.uint8)

            print('q_all bias before learning')
            print(trainer.network.model.get_layer('q_all').get_weights()[-1])

            trainer.qLearnMinibatch([s, a, r, s2, term], True)

            network_outputs,\
            network_vars,\
            network_grads,\
            target_network_outputs,\
            target_network_vars,\
            getQUpdate_vals,\
            getRMSprop_vals = trainer.outputs

            assert network_outputs is not None
            assert network_vars is not None
            assert network_grads is not None
            assert target_network_outputs is not None
            assert target_network_vars is not None
            assert getQUpdate_vals is not None
            assert (getRMSprop_vals is not None and args.optimizer == 'DQN3.0') \
                    or (getRMSprop_vals is None and args.optimizer != 'DQN3.0')

            print("targets")
            print(getQUpdate_vals[-1])
            #print(np.array(getQUpdate_vals[-1]).sum(0))
            print("q_all bias grad")
            print(network_grads[-1])
            print("q_all bias")
            print(network_vars[-1])

            # getQUpdate vars
            fname = './dqn3.0_dump/getQUpdate_{:010d}.dat'.format(step)
            dqn_getQUpdate = [v.numpy() for v in load_lua(fname)]
            print('DQN3.0 targets')
            print(dqn_getQUpdate[-1])
            print(dqn_getQUpdate[-1].sum(0))
            break

            for name, vars, dvars in zip(['q2_all','q2_max', 'q2', 'r', 'q_all', 'q', 'delta','targets'],
                                         getQUpdate_vals,
                                         dqn_getQUpdate):
                assert_equal(vars, dvars, use_float_equal=True)

            for name, n_params, t_params in zip(trainer.network.trainable_layer_names,
                                                trainer.network.get_trainable_parameters(numpy=True),
                                                trainer.target_network.get_trainable_parameters(numpy=True)
                                                ):
                fname = './dqn3.0_dump/cpu_network_vars_{:010d}_{}.dat'.format(step, name)
                dqn_vars = load_lua(fname)
                dqn_vars = [v.numpy()  for v in dqn_vars]
                """
                for var_name, dv, v in zip(['w', 'b', 'dw', 'db'],
                                           dqn_vars,
                                           n_params[1:]):
                    assert_equal(v, dv, use_float_equal=True)
                    print('===== {}-{} ====='.format(name, var_name))
                    print(v)
                    print('===== dqn {}-{} ====='.format(name, var_name))
                    print(dv)
                """

            # RMSprop vars
            fname = './dqn3.0_dump/cpu_rmsprop_vars_{:010d}.dat'.format(step)
            temp = [v.numpy() for v in load_lua(fname)]
            rmsprop = trainer._rmsprop_values
            for name, vars, dvars in zip(['g', 'g2', 'tmp', 'deltas'],
                                   [rmsprop[:10],rmsprop[10:20],rmsprop[20:30],rmsprop[30:40]],
                                   temp):
                start_idx = 0
                end_idx = 0
                dqn_vars = []
                for v in vars:
                    end_idx += v.size
                    dqn_vars.append(dvars[start_idx:end_idx])
                    start_idx = end_idx

                vars = np.array(vars).reshape(-1, 2)
                dqn_vars = np.array(dqn_vars).reshape(-1, 2)

                """
                for layer_name, dvs, vs in zip(trainer.network.trainable_layer_names,
                                               dqn_vars,
                                               vars):

                    assert_equal(v, dv, use_float_equal=True)
                    print('===== {}-{}-w ====='.format(name, layer_name))
                    print(vs[0].shape)
                    print('===== dqn {}-{}-w ====='.format(name, layer_name))
                    print(dvs[0].shape)
                    print('===== {}-{}-b ====='.format(name, layer_name))
                    print(vs[1].shape)
                    print('===== dqn {}-{}-b ====='.format(name, layer_name))
                    print(dvs[1].shape)
                    sys.stdout.flush()
                """

    def test_08_add_learning_summaries(self):
        sleep(1)
        from common import create_networks
        from convnet_keras import KerasConvnet
        from config import get_opt
        sys.argv += ['--backend', self.backend, '--env', 'breakout', '--debug', '--gpu', '-1', '--random_type', 'pytorch']
        args = get_opt()
        args.actions   = [0,1,3,4]
        args.n_actions   = 4

        network, trainer = create_networks(args)

        @patch.object(trainer, "learning_visualizer")
        @patch.object(trainer, "func_qLearnMinibatch_for_summary", return_value=return_value)
        def test(mock_func_qLearnMinibatch_for_summary,
                 mock_learning_visualizer):

            s = s2 = np.ones((32,4,84,84),dtype=np.float32)
            a = np.ones((32,),dtype=np.uint8)
            r = np.zeros((32,),dtype=np.int8)
            term = np.zeros((32,),dtype=np.uint8)
            trainer.qLearnMinibatch([s, a, r, s2, term], True)
            trainer.add_learning_summaries(1000)

            # addNetworkParameters
            expected = {}
            expected_grads = deque(grads)
            for layer_name in network.summarizable_layer_names:
                layer = trainer.network.model.get_layer(layer_name)
                if re.match(r"^relu\d$", layer_name):
                    vars = [layer.output, None, None, None, None]
                else:
                    vars = [layer.output, *layer.get_weights()] + [expected_grads.popleft()] * 2
                expected[layer_name] = vars

            assert_equal(mock_learning_visualizer.addNetworkParameters.call_args[0][0],
                         (expected))


        test()
