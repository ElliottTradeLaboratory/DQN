import sys
import re
import unittest
from unittest.mock import *
import numpy as np
import torch
from torch.utils.serialization import load_lua

sys.path.append('../')
sys.path.append('../../')

from utils import Namespace, get_random
from testutils import *

#class TestPytorchTrainer(unittest.TestCase):

class AbstractTestTrainer(object):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        from time import sleep
        
        sleep(1)
        np.set_printoptions(threshold=np.inf)
        get_random('torch', 1)
        
        backend, self.backendClass, self.trainerClass =\
                self._get_trainer_info()

    def _get_trainer_info(self):
        raise NotImplementedError()

    def test_00_init(self):
        for backend, loss_function, optimizer in [('pytorch_legacy', 'DQN3.0', 'DQN3.0')
#                                  ,('pytorch', 'DQN3.0', 'DQN3.0')
#                                  ,('pytorch', 'RMSprop_w_DQN3loss')
#                                  ,('pytorch', 'RMSprop_w_MSEloss')
#                                  ,('pytorch', 'RMSpropCentered_w_DQN3loss')
#                                  ,('pytorch', 'RMSpropCentered_w_MSEloss')
#                                  ,('pytorch', 'CustomRMSpropCentered_w_DQN3loss')
#                                  ,('pytorch', 'CustomRMSpropCentered_w_MSEloss')
                                  ]:
            with self.subTest('test_00_init: backend:{} optimizer:{}'.format(backend, optimizer)):
                self.backend = backend
                self.loss_function = loss_function
                self.optimizer = optimizer
                self._test_00_init(backend, loss_function, optimizer)
                break

    def _test_00_init(self, backend, loss_function, optimizer):
        from common import create_networks
        from convnet_common import Convnet, Trainer
        from config import get_opt

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--logdir', '/tmp']
        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4
        self.network, self.trainer = create_networks(args)

        self.assertIsNotNone(self.network)
        self.assertIsInstance(self.network, self.backendClass)
        self.assertIsInstance(self.network, Convnet)
        self.assertTrue(hasattr(self.network, 'forward'))
        self.assertTrue(hasattr(self.network, '_save'))
        self.assertTrue(hasattr(self.network, '_load'))

        self.assertIsNotNone(self.trainer)
        self.assertIsInstance(self.trainer, self.trainerClass)
        self.assertIsInstance(self.trainer, Trainer)
        self.assertTrue(hasattr(self.trainer, '_getQUpdate'))
        self.assertTrue(hasattr(self.trainer, '_qLearnMinibatch'))
        self.assertTrue(hasattr(self.trainer, '_update_target_network'))
        self._test_00_init_subclass_check()
        for name, params in zip(self.network.trainable_layer_names,
                                        self.trainer.network.get_trainable_parameters(numpy=True)):
            output, w, b, dw, db = params
            if backend == 'pytorch_legacy':
                assert output is not None
            if not re.match(r'^relu\d$', name):
                assert w is not None
                assert np.all(w != 0), w
                assert b is not None
                assert np.all(b != 0), b
                if backend == 'pytorch_legacy':
                    assert dw is not None
                    assert db is not None
                else:
                    assert dw is None
                    assert db is None
                """
                print('+++++ {} +++++'.format(name))
                print(w)
                print(b)
                """

    def _test_00_init_subclass_check(self):
        from convnet_common import BASE_LAYER_NAMES, BASE_EXCLUDE_LAYER_NAMES_FOR_SUMMARY
        self.assertEqual(self.network.layer_names, BASE_LAYER_NAMES)
        self.assertEqual(self.network.summarizable_layer_names, 
                         [lname for lname in BASE_LAYER_NAMES if not lname in BASE_EXCLUDE_LAYER_NAMES_FOR_SUMMARY])

    def test_00_init2(self):
        self._test_00_init2('pytorch')
        self._test_00_init2('pytorch_legacy')

    def _test_00_init2(self, backend):
        from common import create_networks, get_extensions
        from convnet_common import Convnet, Trainer
        from config import get_opt
        from initializer import get_initializer

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--gpu', '-1',
                     '--logdir', '/tmp']
        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4

        Ex = get_extensions(args)


        from keras import backend as K
        import tensorflow as tf

        sess = K.get_session()

        for initializer in ['torch_nn_default', 'uniform', 'deep_q_rl']:
            args.initializer = initializer
            Ex.setup(args)
            get_random('pytorch', 1)
            if backend == 'pytorch':
                from torch.nn import Conv2d, Linear
                _modules = [Conv2d(4, 32, kernel_size=8, stride=4, padding=1),
                            Conv2d(32, 64, kernel_size=4, stride=2),
                            Conv2d(64, 64, kernel_size=3, stride=1),
                            Linear(3136, 512),
                            Linear(512, args.n_actions)]
                def get_param(v):
                    return v.data
            else:
                from torch.legacy.nn import SpatialConvolution, Linear
                _modules = [SpatialConvolution(4, 32, 8, 8, 4, 4, 1),
                            SpatialConvolution(32, 64, 4, 4, 2, 2),
                            SpatialConvolution(64, 64, 3, 3, 1, 1),
                            Linear(3136, 512),
                            Linear(512, args.n_actions)]
                def get_param(v):
                    return v

            get_random('pytorch', 1)
            for mod in _modules:
                before_w = get_param(mod.weight).clone()
                before_b = get_param(mod.bias).clone()

                print(Ex.weight_init)

                Ex.weight_init(mod)

                after_w = get_param(mod.weight).clone()
                after_b = get_param(mod.bias).clone()

                print(after_b.shape)
                if initializer == 'torch_nn_default':
                    assert before_w.equal(after_w)
                    assert_equal(before_b.numpy(), after_b.numpy(), use_float_equal=True, verbose=10)
                else:
                    assert not before_w.equal(after_w)
                    assert not before_b.equal(after_b)
                    assert_equal(Ex.weight_init.init_w, after_w.numpy(), use_float_equal=True, verbose=10)
                    print(Ex.weight_init.init_b.shape, after_b.numpy().shape)
                    assert_equal(Ex.weight_init.init_b, after_b.numpy(), use_float_equal=True, verbose=10)


    def _test_01_get_summarizable_parameters(self):
        raise NotImplementedError()

    def test_01_get_summarizable_parameters(self):
        for backend, loss_function, optimizer in [('pytorch_legacy', 'DQN3.0', 'DQN3.0')
#                               ,('pytorch', 'DQN3.0', 'DQN3.0')
#                              ,('pytorch', 'RMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'RMSpropCentered_w_MSEloss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_MSEloss')
                              ]:
            with self.subTest('test_01_get_summarizable_parameters: backend:{} optimizer:{}'.format(backend, optimizer)):
                self.backend = backend
                self.optimizer = optimizer
                self.loss_function = loss_function
                self._test_01_get_summarizable_parameters_(backend, loss_function, optimizer)

    def _test_01_get_summarizable_parameters_(self, backend, loss_function, optimizer):
        from common import create_networks
        from config import get_opt

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--logdir', '/tmp',
                     '--gpu', '-1']
        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4
        self.network, _ = create_networks(args)

        self._test_01_get_summarizable_parameters()

    def _test_02_update_target_network_fill_params(self, fill_net_vars, fill_target_vars):
        raise NotImplementedError()

    def test_02_update_target_network(self):
        for backend, loss_func, optimizer in [('pytorch_legacy', 'DQN3.0', 'DQN3.0')
#                                  ,('pytorch', 'DQN3.0', 'DQN3.0')
#                              ,('pytorch_legacy', 'LegacyRMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSprop_w_DQN3loss')
#                              ,('pytorch', 'RMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'RMSpropCentered_w_MSEloss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_MSEloss')
                               ]:
            with self.subTest('test_02_update_target_network: backend:{} optimizer:{}'.format(backend, optimizer)):
                self.backend = backend
                self.optimizer = optimizer
                self._test_02_update_target_network(backend, loss_func, optimizer)

    def _test_02_update_target_network(self, backend, loss_function, optimizer):
        from common import create_networks
        from convnet_pytorch import Utils
        from config import get_opt

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--logdir', '/tmp']

        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4

        utils = Utils(args)
        network, self.trainer = create_networks(args)

        assert_not_equal([utils._get_numpy(p) for p in self.trainer.network.model.parameters()[0]],
                         [utils._get_numpy(p) for p in self.trainer.target_network.model.parameters()[0]])
        self.trainer.update_target_network()

        assert_equal([utils._get_numpy(p) for p in self.trainer.network.model.parameters()[0]],
                     [utils._get_numpy(p) for p in self.trainer.target_network.model.parameters()[0]])

    def test_03_0_qLearnMinibatch(self):

        from common import create_networks, get_extensions
        from convnet_pytorch import Utils
        from config import get_opt
        from pytorch_extensions import setup_before_package_loading
        from initenv import setup
        
        sys.argv += ['--backend', 'pytorch_legacy',
                     '--env', 'breakout',
                     '--loss_function', 'DQN3.0',
                     '--optimizer', 'DQN3.0',
                     '--gpu', '-1',
                     '--logdir', '/tmp']
        args = get_opt()
        game_env, agent, game_actions, args = setup(args)

        self.trainer = agent.trainer
        
        self.trainer.update_target_network()
        
        params, grads = self.trainer.target_network.model.parameters()
        
        for step in range(50004, 50016, 4):
            fname = './dqn3.0_dump/cpu_batch_{:010d}.dat'.format(step)
            batch = load_lua(fname)
            
            s, a, r, s2, term  = [v.numpy() for v in batch]
            s = s.reshape(32, 4, 84, 84)
            s2 = s2.reshape(32, 4, 84, 84)
            a = a - 1
            print(a.shape)

            self.trainer.qLearnMinibatch([s, a, r, s2, term], True)

            params = self.trainer.network.get_summarizable_parameters(True)[-1]
            print('q_all', self.trainer.optimizer.q_all.numpy())
            print('delta', self.trainer.optimizer.delta.numpy())
            print('q_all bias grad', params[-1])
            break
            

            for name, params in zip(self.network.summarizable_layer_names,
                                    self.network.get_summarizable_parameters(True)):
                o, w, b, dw, db = params
                print('~~~~~~{}:{}~~~~~~'.format(step, name))
                print(o)
            break


    def test_03_qLearnMinibatch(self):
    
        for backend, loss_function, optimizer in [('pytorch_legacy', 'DQN3.0', 'DQN3.0'),
                                                  ('pytorch', 'DQN3.0', 'DQN3.0')
#                                  ,('pytorch_legacy', 'LegacyRMSprop_w_DQN3loss')
#                                  ,('pytorch_legacy', 'LegacyRMSprop_w_MSEloss')
#                                  ,('pytorch', 'RMSprop_w_DQN3loss')
#                                  ,('pytorch', 'RMSprop_w_MSEloss')
#                                  ,('pytorch', 'RMSpropCentered_w_DQN3loss')
#                                  ,('pytorch', 'RMSpropCentered_w_MSEloss')
#                                  ,('pytorch', 'CustomRMSpropCentered_w_DQN3loss')
#                                  ,('pytorch', 'CustomRMSpropCentered_w_MSEloss')
                                ]:
            with self.subTest('test_03_qLearnMinibatch: backend:{} optimizer:{}'.format(backend, optimizer)):
                self.backend = backend
                self.optimizer = optimizer
                self.loss_function = loss_function
                self._test_03_qLearnMinibatch(backend, loss_function, optimizer)

    def _test_03_qLearnMinibatch(self, backend, loss_function, optimizer):

        from common import create_networks, get_extensions
        from convnet_pytorch import Utils
        from config import get_opt

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--random_type', 'torch',
                     '--preproc', 'image',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--logdir', '/tmp',
                     '--gpu', '-1']

        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4
        Ex = get_extensions(args)
        Ex.setup(args)
        utils = Utils(args)
        get_random('torch', 1)
        self.network, self.trainer = create_networks(args)

        self.trainer.update_target_network()

        for step in range(50004, 50016, 4):
            with self.subTest('{}-{} step({})'.format(backend, optimizer, step)):
                fname = './dqn3.0_dump/cpu_batch_{:010d}.dat'.format(step)
                batch = load_lua(fname)
                
                s, a, r, s2, term = [v.numpy() for v in batch]
                s = s.reshape(32, 4, 84, 84)
                s2 = s2.reshape(32, 4, 84, 84)
                a -= 1

                self.trainer.qLearnMinibatch([s, a, r, s2, term], True)

                # check getQUpdate vars
                fname = './dqn3.0_dump/getQUpdate_{:010d}.dat'.format(step)
                dqn_getQUpdate = [v.numpy() for v in load_lua(fname)]
                getQUpdate = [self.trainer.optimizer.q2_all,
                              self.trainer.optimizer.q2_max,
                              self.trainer.optimizer.q2,
                              self.trainer.optimizer.r,
                              self.trainer.optimizer.q_all,
                              self.trainer.optimizer.q,
                              self.trainer.optimizer.delta,
                              self.trainer.optimizer.targets]
                getQUpdate = [utils._get_numpy(v) for v in getQUpdate]
                for i, vars in enumerate(zip(['q2_all', 'q2_max', 'q2', 'r', 'q_all','q','delta','targets'],
                                             dqn_getQUpdate,
                                             getQUpdate)):
                    if vars[0] == 'targets' and re.match(r'^.*MSEloss$', optimizer):
                        continue
                    #if vars[0] == 'q2_all':
                    #    q2_all =  self.trainer.target_network.forward(s2)
                    #    assert_equal(vars[1], q2_all.numpy(), vars[0], use_float_equal=True, verbose=1)
                    with self.subTest('{}-{} {} getQUpdate vars {}'.format(backend, optimizer, i, vars[0])):
                        print(vars[0])
                        assert_equal(vars[1], vars[2], vars[0], use_float_equal=True, verbose=1)
                        sys.stdout.flush()

                # check RMSprop vars
                if not re.match(r"^.*RMSprop.*$", optimizer):
                    fname = './dqn3.0_dump/cpu_rmsprop_vars_{:010d}.dat'.format(step)
                    dqn_rmsprop = [v.numpy() for v in load_lua(fname)]
                    rmsprop = [utils._get_numpy(v) for v in [self.trainer.optimizer.g, self.trainer.optimizer.g2, self.trainer.optimizer.tmp, self.trainer.optimizer.deltas]]
                    for i, vars in enumerate(zip(['g', 'g2', 'tmp', 'deltas'],
                                                 dqn_rmsprop,
                                                 rmsprop)):
                        with self.subTest('{}-{} {} rmsprop vars {}'.format(backend, optimizer, i, vars[0])):
                            assert_equal(vars[1], vars[2], vars[0], use_float_equal=True, verbose=1)
                            sys.stdout.flush()

                # check trainable params
                [self.assertTrue(np.all(w != 0) and np.all(b != 0), [w,b]) for o, w, b, dw, db in self.trainer.network.get_trainable_parameters(numpy=True)]
                for layer_name, n_params, t_params in zip(self.trainer.network.trainable_layer_names,
                                                          self.trainer.network.get_trainable_parameters(numpy=True),
                                                          self.trainer.target_network.get_trainable_parameters(numpy=True)
                                                          ):
                    o, w, b, dw, db = n_params
                    assert np.all(w != 0)
                    assert np.all(b != 0)
                    with self.subTest('{}-{} after Learning params'.format(backend, optimizer)):
                        fname = './dqn3.0_dump/cpu_network_vars_{:010d}_{}.dat'.format(step, layer_name)
                        dqn_vars = load_lua(fname)
                        dqn_vars = [v.numpy()  for v in dqn_vars]
                        assert_equal(dqn_vars, n_params[1:], use_float_equal=True)

                        fname = './dqn3.0_dump/cpu_target_network_vars_{:010d}_{}.dat'.format(step, layer_name)
                        dqn_vars = load_lua(fname)
                        dqn_vars = [v.numpy()  for v in dqn_vars]
                        assert_equal(dqn_vars, t_params[1:], use_float_equal=True)

    def test_04_save_load(self):
        for backend, loss_func, optimizer in [('pytorch_legacy', 'DQN3.0', 'DQN3.0')
#                                  ,('pytorch', 'DQN3.0', 'DQN3.0')
#                              ,('pytorch_legacy', 'LegacyRMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSprop_w_DQN3loss')
#                              ,('pytorch', 'RMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'RMSpropCentered_w_MSEloss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_MSEloss')
                               ]:
            with self.subTest('test_04_save_load: backend:{} optimizer:{}'.format(backend, optimizer)):
                self.backend = backend
                self.optimizer = optimizer
                self.__test_04_save_load(backend, loss_func, optimizer)

    def __test_04_save_load(self, backend, loss_function, optimizer):
        from common import create_networks
        from config import get_opt
        
        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--gpu', '-1']
        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4
        self.network, _ = create_networks(args)
        
        self._test_04_save_load()

    def _test_04_save_load(self):
        raise NotImplementedError()

    def test_05_add_learning_sumarries(self):
        for backend, loss_function, optimizer in [('pytorch_legacy', 'DQN3.0', 'DQN3.0')
#                              ,('pytorch_legacy', 'LegacyRMSprop_w_DQN3loss')
#                              ,('pytorch_legacy', 'LegacyRMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSprop_w_DQN3loss')
#                              ,('pytorch', 'RMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'RMSpropCentered_w_MSEloss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_MSEloss')
                              ]:
            with self.subTest('test_05_add_learning_sumarries: backend:{} optimizer:{}'.format(backend, optimizer)):
                self.backend = backend
                self.optimizer = optimizer
                self.loss_function = loss_function
                self._test_05_add_learning_sumarries(backend, loss_function, optimizer)

    def _test_05_add_learning_sumarries(self,backend, loss_function, optimizer):

        from common import create_networks
        
        from config import get_opt

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--logdir', '/tmp']
        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4
        
        _, self.trainer = create_networks(args)

        self._test_05_add_learning_before()

        with patch.object(self.trainer, 'learning_visualizer') as mock_visualizer:

            x = load_lua('./dqn3.0_dump/batch_0000050004.dat')
            
            x = [v.numpy() for v in x]
            x[0] = x[0].reshape(32, 4,84,84)
            x[1] = x[1] - 1
            x[3] = x[3].reshape(32, 4,84,84)
            
            self.trainer.qLearnMinibatch(x, True)
            self.trainer.add_learning_summaries(50004)

            expected_params = self._test_05_add_learning_sumaries_get_expected()

            self.assertIn('addInputImages_network', expected_params)
            self.assertIn('addInputImages_target_network', expected_params)
            for call_arg, family_id, s in zip(mock_visualizer.addInputImages.call_args_list,
                                              [2,3],
                                              [expected_params['addInputImages_network'],
                                               expected_params['addInputImages_target_network']]):
                args, _ = call_arg
                self.assertEqual(args[0], family_id)
                self.assertTrue(float_equal(np.array(args[1]), s))
            
            subTest = self.subTest
            def _chekc_params(name):
                with subTest('{}-{} {}'.format(backend, optimizer, name)):
                    call_args_list = getattr(mock_visualizer,name).call_args_list
                    self.assertEqual(len(call_args_list), 1)
                    call_arg = call_args_list[0]
                    args, _ = call_arg
                    args = args[0]
                    self.assertIsInstance(args, dict)
                    ex_params = expected_params[name]
                    for key in ex_params.keys():
                        with subTest('{}-{} {}.{}'.format(backend, optimizer, name, key)):
                            parm = ex_params[key]
                            if np.isscalar(parm):
                                self.assertTrue(np.isscalar(args[key]))
                                self.assertEqual(ex_params[key], args[key])
                            else:
                                for v1, v2 in zip(ex_params[key], args[key]):
                                    if v1 is None:
                                        self.assertIsNone(v2)
                                    else:
                                        if isinstance(v1, np.ndarray):
                                            self.assertTrue(float_equal(v1, v2))
                                        else:
                                            self.assertEqual(v1, v2)

            _chekc_params('addNetworkParameters')
            _chekc_params('addTargetNetworkParameters')
            _chekc_params('addGetQUpdateValues')
            _chekc_params('addRMSpropValues')

        self._test_05_add_learning_after()

    def _test_05_add_learning_before(self):
        raise NotImplementedError()
    
    def _test_05_add_learning_sumaries_get_expected(self):
        raise NotImplementedError()

    def _test_05_add_learning_after(self):
        raise NotImplementedError()
    

    def test_06_compute_validation_statistics(self):
        from time import sleep
        for backend, loss_function, optimizer in [('pytorch_legacy', 'DQN3.0', 'DQN3.0')
#                             ,('pytorch_legacy', 'LegacyRMSprop_w_DQN3loss')
#                             ,('pytorch_legacy', 'LegacyRMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSprop_w_DQN3loss')
#                              ,('pytorch', 'RMSprop_w_MSEloss')
#                              ,('pytorch', 'RMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'RMSpropCentered_w_MSEloss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_DQN3loss')
#                              ,('pytorch', 'CustomRMSpropCentered_w_MSEloss')
                              ]:
            with self.subTest('test_06_compute_validation_statistics: backend:{} optimizer:{}'.format(backend, optimizer)):
                sleep(1)
                self.backend = backend
                self.optimizer = optimizer
                self.loss_function = loss_function
                self._test_06_compute_validation_statistics(backend, loss_function, optimizer)

    def _test_06_compute_validation_statistics(self, backend, loss_function, optimizer):

        from common import create_networks
        
        from config import get_opt

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--logdir', '/tmp']
        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4

        from config import get_opt

        sys.argv += ['--backend', backend,
                     '--env', 'breakout',
                     '--loss_function', loss_function,
                     '--optimizer', optimizer,
                     '--logdir', '/tmp']
        args = get_opt()
        args.actions = [0,1,3,4]
        args.n_actions = 4
        
        _, self.trainer = create_networks(args)

        s = np.ones((5,4,84,84), dtype=np.float32)
        a = np.ones((5,), dtype=np.uint8)
        r = np.ones((5,), dtype=np.int)
        s2 = np.ones((5,4,84,84), dtype=np.float32)
        t = np.ones((5,), dtype=np.uint8)

        delta, q_max = self.trainer.compute_validation_statistics([s,a,r,s2,t])
        
        self.assertIsInstance(delta, np.ndarray)
        self.assertIsInstance(q_max, np.ndarray)
        

class TestPytorchTrainer(AbstractTestTrainer, unittest.TestCase):
        
    def _get_trainer_info(self):
        from convnet_pytorch import PyTorchConvnet, PyTorchTrainer
        return 'pytorch_legacy', PyTorchConvnet, PyTorchTrainer


    def _test_01_get_summarizable_parameters(self):
        from torch.legacy.nn import Linear, SpatialConvolution
        from convnet_pytorch import Utils
        from pytorch_extensions import Rectifier
        utils = Utils(Namespace(gpu=-1, backend=self.backend, optimizer=self.optimizer))

        # Create expected params.
        expecteds_1 = []
        for i, mod in enumerate(self.network.model.modules):
            if isinstance(mod, SpatialConvolution) or isinstance(mod, Linear):
                mod.weight.copy_(torch.FloatTensor(mod.weight).fill_(i))
                mod.bias.copy_(torch.FloatTensor(mod.bias).fill_(i))
                mod.gradWeight.copy_(torch.FloatTensor(mod.gradWeight).fill_(i))
                mod.gradBias.copy_(torch.FloatTensor(mod.gradBias).fill_(i))
                expected = [
                    mod.output,
                    mod.weight,
                    mod.bias,
                    mod.gradWeight,
                    mod.gradBias
                ]
            
            elif isinstance(mod, Rectifier):
                expected = [mod.output, None, None, None, None]
                
            else:
                continue

            expecteds_1.append(expected)

        # test get summarizable parameters with numpy
        for net_vars, expecteds in zip(self.network.get_summarizable_parameters(numpy=True), expecteds_1):
            for name, net_var, expect_var in zip(['name', 'output', 'w', 'b', 'dw', 'db'],
                                                    net_vars,
                                                    expecteds):
                sys.stdout.flush()
                with self.subTest('{}-{} get_summarizable_parameters numpy=True: {}'.format(self.backend, self.optimizer, name)):
                    if name == 'name':
                        assert_equal(net_var, utils._get_numpy(expect_var))
                    else:
                        if expect_var is None:
                            assert net_var is None, name
                        else:
                            expect_var = utils._get_numpy(expect_var)
                            assert isinstance(net_var, np.ndarray), '{}, {}'.format(name, type(net_var))
                            self.assertTrue(np.all(net_var.shape == expect_var.shape))
                            self.assertTrue(np.all(net_var == expect_var))


        # test get summarizable parameters with torch Tensor
        for net_vars, expecteds in zip(self.network.get_summarizable_parameters(numpy=False), expecteds_1):
            for name, net_var, expect_var in zip(['output', 'w', 'b', 'dw', 'db'],
                                                    net_vars,
                                                    expecteds):
                sys.stdout.flush()
                with self.subTest('{}-{} get_summarizable_parameters numpy=False {}'.format(self.backend, self.optimizer, name)):
                    if name == 'name':
                        self.assertEqual(net_var, expect_var)
                    else:
                        if expect_var is None:
                            assert net_var is None, name
                        else:
                            assert isinstance(net_var, torch.FloatTensor), '{}, {}'.format(name, type(net_var))
                            self.assertEqual(net_var.data_ptr, expect_var.data_ptr)

    def _test_04_save_load(self):
        import os
        from datetime import datetime

        if self.backend=='pytorch_legacy':
            w, dw = self.network.model.flattenParameters()

            one = torch.FloatTensor(w.size()).fill_(1.0)
            w.copy_(one)
            dw.copy_(one)
            
            # the flatten parameters need to be acquired immediately before use.
            w, dw = self.network.model.flattenParameters()
            self.assertTrue(w.equal(one))
            self.assertTrue(dw.equal(one))

            org_w = w.clone()
            org_dw = dw.clone()

            self.assertTrue(org_w.equal(one))
            self.assertTrue(org_w.equal(one))

            w, dw = self.network.model.flattenParameters()
            self.assertTrue(org_w.equal(w))
            self.assertTrue(org_dw.equal(dw))

            filepath = '/tmp/saved_network_{}.dat'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

            self.network.save(filepath)
            
            self.assertTrue(os.path.exists(filepath))
            
            two = torch.FloatTensor(w.size()).fill_(2.0)

            w, dw = self.network.model.flattenParameters()
            w.copy_(two)
            dw.copy_(two)

            w, dw = self.network.model.flattenParameters()
            self.assertTrue(w.equal(two))
            self.assertTrue(dw.equal(two))
            self.assertFalse(org_w.equal(w))
            self.assertFalse(org_dw.equal(dw))
            self.assertTrue(org_w.equal(one))
            self.assertTrue(org_w.equal(one))

            self.network.load(filepath)
            
            w, dw = self.network.model.flattenParameters()
            self.assertTrue(org_w.equal(w))
            self.assertTrue(org_dw.equal(dw))
            self.assertTrue(org_w.equal(one))
            self.assertTrue(org_w.equal(one))
            self.assertTrue(w.equal(one))
            self.assertTrue(dw.equal(one))

        else:
            ws, dws = self.network.model.parameters()
            org_ws = [w.data.clone().numpy() for w in ws]
            w = org_ws[-1]
            print('org_ws', w)
            np_ws = [w.data.clone().numpy() for w in ws]
            assert_equal(org_ws, np_ws)
            
            filepath = '/tmp/saved_network_{}.dat'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.network.save(filepath)

            changed_ws = [w.data.add_(1) for w in ws]
            changed_ws = [w.clone().numpy() for w in changed_ws]
            w = changed_ws[-1]
            print('changed_ws', w)
            ws, dws = self.network.model.parameters()
            np_ws = [w.data.clone().numpy() for w in ws]
            w = np_ws[-1]
            print('re-get np_ws', w)
            assert_equal(changed_ws, np_ws)
            assert_not_equal(org_ws, np_ws)

            self.network.load(filepath)
            ws, dws = self.network.model.parameters()
            np_ws = [w.data.clone().numpy() for w in ws]
            w = np_ws[-1]
            print('after load np_ws', w)
            w = org_ws[-1]
            print('org_ws', w)
            assert_equal(org_ws, np_ws)


    def _test_05_add_learning_before(self):

        org_qLearnMinibatch = self.trainer.qLearnMinibatch
        
        def inject_qLearnMinibatch(x, do_summary):
            org_qLearnMinibatch(x, do_summary)

        self.patcher = patch.object(self.trainer, 'qLearnMinibatch' , wraps=inject_qLearnMinibatch)
        self.mock_qLearnMinibatch = self.patcher.start()

    def _test_05_add_learning_sumaries_get_expected(self):
        
        from visualizer import LearningVisualizer
        from convnet_pytorch import Utils
        utils = Utils(Namespace(gpu=-1, backend=self.backend))
        
        params = {}
        
        x, _ = self.mock_qLearnMinibatch.call_args
        s, a, r, s2, t = x[0]
        params['addInputImages_network'] = s
        params['addInputImages_target_network'] = s2
        
        addNetworkParameters = dict(zip(self.trainer.network.summarizable_layer_names,
                                        self.trainer.network.get_summarizable_parameters()))
        params['addNetworkParameters'] = addNetworkParameters
        
        addTargetNetworkParameters = dict(zip(self.trainer.target_network.summarizable_layer_names,
                                              self.trainer.target_network.get_summarizable_parameters()))
        params['addTargetNetworkParameters'] = addTargetNetworkParameters

        addGetQUpdateValues = {}
        for name in LearningVisualizer.GET_Q_UPDATE_VALUE_NAMES:
            print(name)
            addGetQUpdateValues[name] = utils._get_numpy(self.trainer._get_trainer_attrs(name))

        params['addGetQUpdateValues'] = addGetQUpdateValues

        addRMSpropValues = {}
        for name in LearningVisualizer.RMS_PROP_VALUE_NAMES[:4]:
            addRMSpropValues[name] = getattr(self.trainer, name).numpy()
        addRMSpropValues[LearningVisualizer.RMS_PROP_VALUE_NAMES[-1]] = self.trainer.args.lr

        params['addRMSpropValues'] = addRMSpropValues

        return params

    def _test_05_add_learning_after(self):
        self.patcher.stop()
        
