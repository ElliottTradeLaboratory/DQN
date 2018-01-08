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
from testutils import float_equal

#class TestPytorchTrainer(unittest.TestCase):

class AbstractTestTrainer(object):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def _setinfo(self):
        self.convnet = convnet
        self.convnetClass = convnetClass
        self.trainerClass = self.trainerClass

    def setUp(self):

        np.set_printoptions(threshold=np.inf)
        get_random('pytorch', 1)
        
        self.convnet, self.convnetClass, self.trainerClass =\
            self._get_trainer_info()

    def _get_trainer_info(self):
        raise NotImplementedError()

    def test_00_init(self):
    
        from dqn import create_networks
        from convnet_common import Convnet, Trainer
        
        args = Namespace(discount=0.99,
                         lr=0.00025,
                         gpu=-1,
                         log_dir='/tmp',
                         n_actions=4,
                         verbose=2,
                         clip_delta=1,
                         minibatch_size=32,
                         convnet=self.convnet)
        
        network, self.trainer = create_networks(args)

        self.assertIsNotNone(network)
        self.assertIsInstance(network, self.convnetClass)
        self.assertIsInstance(network, Convnet)
        self.assertTrue(hasattr(network, 'forward'))
        self.assertTrue(hasattr(network, '_get_layer_params'))
        self.assertTrue(hasattr(network, '_save'))
        self.assertTrue(hasattr(network, '_load'))

        self.assertIsNotNone(self.trainer)
        self.assertIsInstance(self.trainer, self.trainerClass)
        self.assertIsInstance(self.trainer, Trainer)
        self.assertTrue(hasattr(self.trainer, '_getQUpdate'))
        self.assertTrue(hasattr(self.trainer, '_qLearnMinibatch'))
        self.assertTrue(hasattr(self.trainer, '_add_learning_summaries'))
        self.assertTrue(hasattr(self.trainer, '_update_target_network'))
        self.assertTrue(hasattr(self.trainer, '_getQUpdate_values'))
        self.assertTrue(hasattr(self.trainer, '_rmsprop_values'))
    

    def _test_01_get_parameters(self):
        raise NotImplementedError()

    def test_01_get_parameters(self):
        from dqn import create_networks
        
        args = Namespace(discount=0.99,
                         lr=0.00025,
                         gpu=-1,
                         log_dir='/tmp',
                         n_actions=4,
                         verbose=2,
                         clip_delta=1,
                         minibatch_size=32,
                         convnet=self.convnet)
        
        self.network, _ = create_networks(args)

        self._test_01_get_parameters()

    def _test_02_update_target_network_fill_params(self, fill_net_vars, fill_target_vars):
        raise NotImplementedError()

    def test_02_update_target_network(self):
        from dqn import create_networks
        
        args = Namespace(discount=0.99,
                         lr=0.00025,
                         gpu=-1,
                         log_dir='/tmp',
                         n_actions=4,
                         verbose=2,
                         clip_delta=1,
                         minibatch_size=32,
                         convnet=self.convnet)
        
        network, self.trainer = create_networks(args)

        for i, net_vars in enumerate(zip(self.trainer.network.get_parameters(),
                                         self.trainer.target_network.get_parameters())):
            network_vars, target_network_vars = net_vars
            self.assertEqual(network_vars[0], target_network_vars[0])
            if not re.match('^relu.$', network_vars[0]):
                for net_nm, net_var, target_var in zip(['w', 'b', 'dw', 'db'],
                                            network_vars[2:],
                                            target_network_vars[2:]):
                    print(i, 'before update_target_network setup : shape is must exactry same', net_nm)
                    with self.subTest('{} before update_target_network setup : shape is must exactry same {} {}'.format(i, net_nm, network_vars[0])):
                        self.assertTrue(np.all(net_var.shape == target_var.shape))

        self._test_02_update_target_network_fill_params(1, 2)

        for i, net_vars in enumerate(zip(self.trainer.network.get_parameters(),
                                         self.trainer.target_network.get_parameters())):
            network_vars, target_network_vars = net_vars
            self.assertEqual(network_vars[0], target_network_vars[0])
            if not re.match('^relu.$', network_vars[0]):
                for net_nm, net_var, target_var in zip(['w', 'b', 'dw', 'db'],
                                            network_vars[2:],
                                            target_network_vars[2:]):
                    print(i, 'before update_target_network setup : fill 1 or 2', net_nm)
                    with self.subTest('{} before update_target_network setup : fill 1 or 2 {} {}'.format(i, net_nm, network_vars[0])):
                        self.assertTrue(np.all(net_var == np.zeros(net_var.shape).astype(np.float32)+1))
                        self.assertTrue(np.all(target_var == np.zeros(target_var.shape).astype(np.float32)+2))

        self.trainer.update_target_network()

        for i, net_vars in enumerate(zip(self.trainer.network.get_parameters(),
                                         self.trainer.target_network.get_parameters())):
            network_vars, target_network_vars = net_vars
            self.assertEqual(network_vars[0], target_network_vars[0])
            if not re.match('^relu.$', network_vars[0]):
                for net_nm, net_var, target_var in zip(['w', 'b', 'dw', 'db'],
                                            network_vars[2:],
                                            target_network_vars[2:]):
                    print(i, 'after update_target_network must equal', net_nm)
                    with self.subTest('{} after update_target_network must equal'.format(i, net_nm, network_vars[0])):
                         self.assertTrue(np.all(net_var == target_var))

    def test_03_qLearnMinibatch_01(self):
    
        from dqn import create_networks
        
        args = Namespace(discount=0.99,
                         lr=0.00025,
                         gpu=-1,
                         log_dir='/tmp',
                         n_actions=4,
                         verbose=2,
                         clip_delta=1,
                         minibatch_size=32,
                         convnet=self.convnet)
        
        network, self.trainer = create_networks(args)
        
        self.trainer.update_target_network()
        
        for i, net_vars in enumerate(zip(self.trainer.network.get_parameters(), self.trainer.target_network.get_parameters())):
            network_vars, target_network_vars = net_vars
            self.assertEqual(network_vars[0], target_network_vars[0])
            if not re.match('^relu.$', network_vars[0]):
                for net_nm, gradvars in zip(['network', 'target_network'], net_vars):
                    with self.subTest('{} before learning grad params are must zero {} {}'.format(i, net_nm, network_vars[0])):
                        dw, db = gradvars[4:]
                        self.assertTrue(np.all(dw == np.zeros(dw.shape)))
                        self.assertTrue(np.all(db == np.zeros(db.shape)))

        for step in range(50004, 50016, 4):
            with self.subTest(i=step):
                fname = './dqn3.0_dump/cpu_batch_{:010d}.dat'.format(step)
                batch = load_lua(fname)
                
                s, a, r, s2, term = [v.numpy() for v in batch]
                s = s.reshape(32, 4, 84, 84)
                s2 = s2.reshape(32, 4, 84, 84)
                a -= 1
                self.trainer.qLearnMinibatch([s, a, r, s2, term])


                fname = './dqn3.0_dump/getQUpdate_{:010d}.dat'.format(step)
                dqn_getQUpdate = [v.numpy() for v in load_lua(fname)]
                getQUpdate = [self.trainer.q2_all,
                              self.trainer.q2_max,
                              self.trainer.q2,
                              self.trainer.r,
                              self.trainer.q_all,
                              self.trainer.q,
                              self.trainer.delta,
                              self.trainer.targets]
                getQUpdate = [v.numpy() for v in getQUpdate]
                for i, vars in enumerate(zip(['q2_all', 'q2_max', 'q2', 'r', 'q_all','q','delta','targets'],
                                             dqn_getQUpdate,
                                             getQUpdate)):
                    with self.subTest('{} getQUpdate vars {}'.format(i, vars[0])):
                        self.assertTrue(float_equal(vars[1], vars[2], vars[0]))
                        sys.stdout.flush()

                fname = './dqn3.0_dump/cpu_rmsprop_vars_{:010d}.dat'.format(step)
                dqn_rmsprop = [v.numpy() for v in load_lua(fname)]
                rmsprop = [v.numpy() for v in [self.trainer.g, self.trainer.g2, self.trainer.tmp, self.trainer.deltas]]
                for i, vars in enumerate(zip(['g', 'g2', 'tmp', 'deltas'],
                                             dqn_rmsprop,
                                             rmsprop)):
                    with self.subTest('{} rmsprop vars {}'.format(i, vars[0])):
                        self.assertTrue(float_equal(vars[1], vars[2], vars[0]))
                        sys.stdout.flush()

                for i, net in enumerate(zip(self.trainer.network.get_parameters(), self.trainer.target_network.get_parameters())):
                    network, target_network = net
                    with self.subTest('{} after Learning params {}'.format(i, network[0])):
                        self.assertEqual(network[0], target_network[0])
                        if not re.match('^relu.$', network[0]):
                            sys.stdout.flush()
                            fname = './dqn3.0_dump/cpu_network_vars_{:010d}_{}.dat'.format(step, network[0])
                            dqn_vars = load_lua(fname)
                            dqn_vars = [v.numpy()  for v in dqn_vars]
                            vars = network[2:]
                            for i, vars in enumerate(zip(['w', 'b', 'dw', 'db'],
                                                         dqn_vars,
                                                         vars)):
                                with self.subTest('{} network.{} vars {}'.format(i, network[0], vars[0])):
                                    self.assertTrue(np.all(vars[1].shape == vars[2].shape))
                                    self.assertTrue(float_equal(vars[1], vars[2], 'network {}_{}'.format(network[0], vars[0])))
                                    sys.stdout.flush()

                            fname = './dqn3.0_dump/cpu_target_network_vars_{:010d}_{}.dat'.format(step, network[0])
                            dqn_vars = load_lua(fname)
                            dqn_vars = [v.numpy()  for v in dqn_vars]
                            vars = target_network[2:]
                            for i, vars in enumerate(zip(['w', 'b', 'dw', 'db'],
                                                         dqn_vars,
                                                         vars)):
                                with self.subTest('{} target_network.{} vars {}'.format(i, target_network[0], vars[0])):
                                    self.assertTrue(np.all(vars[1].shape == vars[2].shape))
                                    self.assertTrue(float_equal(vars[1], vars[2], 'target_network {}_{}'.format(target_network[0], vars[0])))
                                    sys.stdout.flush()


    def test_04_save_load(self):

        from dqn import create_networks
        
        args = Namespace(discount=0.99,
                         lr=0.00025,
                         gpu=-1,
                         log_dir='/tmp',
                         n_actions=4,
                         verbose=2,
                         clip_delta=1,
                         minibatch_size=32,
                         convnet=self.convnet)
        
        self.network, _ = create_networks(args)
        
        self._test_04_save_load()

    def _test_04_save_load(self):
        raise NotImplementedError()

    def test_05_add_learning_sumarries(self):


        from dqn import create_networks
        
        args = Namespace(discount=0.99,
                         lr=0.00025,
                         gpu=-1,
                         log_dir='/tmp',
                         n_actions=4,
                         verbose=2,
                         clip_delta=1,
                         minibatch_size=32,
                         convnet=self.convnet)
        
        _, self.trainer = create_networks(args)

        self._test_05_add_learning_before()

        with patch.object(self.trainer, 'learning_visualizer') as mock_visualizer:

            x = load_lua('./dqn3.0_dump/batch_0000050004.dat')
            
            x = [v.numpy() for v in x]
            x[0] = x[0].reshape(32, 4,84,84)
            x[1] = x[1] - 1
            x[3] = x[3].reshape(32, 4,84,84)
            
            self.trainer.qLearnMinibatch(x)
            self.trainer.add_learning_summaries(50004)

            expected_params = self._test_05_add_learning_sumarries_get_expected()

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
                with subTest(name):
                    call_args_list = getattr(mock_visualizer,name).call_args_list
                    self.assertEqual(len(call_args_list), 1)
                    call_arg = call_args_list[0]
                    args, _ = call_arg
                    args = args[0]
                    self.assertIsInstance(args, dict)
                    ex_params = expected_params[name]
                    for key in ex_params.keys():
                        with subTest('{}.{}'.format(name, key)):
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
    
    def _test_05_add_learning_sumarries_get_expected(self):
        raise NotImplementedError()

    def _test_05_add_learning_after(self):
        raise NotImplementedError()
    


class TestPytorchTrainer(AbstractTestTrainer, unittest.TestCase):
        
    def _get_trainer_info(self):
        from convnet_pytorch import PyTorchConvnet, PyTorchTrainer
        return 'pytorch', PyTorchConvnet, PyTorchTrainer

    def _test_01_get_parameters(self):
        from torch.legacy.nn import Linear, SpatialConvolution
        from convnet_pytorch import Rectifier

        expecteds_1 = []
        for i, mod in enumerate(self.network.model.modules):
            if isinstance(mod, SpatialConvolution) or isinstance(mod, Linear):
                mod.weight.copy_(torch.FloatTensor(mod.weight).fill_(i))
                mod.bias.copy_(torch.FloatTensor(mod.bias).fill_(i))
                mod.gradWeight.copy_(torch.FloatTensor(mod.gradWeight).fill_(i))
                mod.gradBias.copy_(torch.FloatTensor(mod.gradBias).fill_(i))
                expected = [
                    mod.name,
                    mod.output,
                    mod.weight,
                    mod.bias,
                    mod.gradWeight,
                    mod.gradBias
                ]
            
            elif isinstance(mod, Rectifier):
                expected = [mod.name, mod.output, None, None, None, None]
                
            else:
                continue

            expecteds_1.append(expected)

        for net_vars, expecteds in zip(self.network.get_parameters(), expecteds_1):
            print(len(net_vars))
            for name, net_var, expect_var in zip(['name', 'output', 'w', 'b', 'dw', 'db'],
                                                    net_vars,
                                                    expecteds):
                print('get_parameters numpy=True, clone=True:', name)
                sys.stdout.flush()
                with self.subTest('get_parameters numpy=True: {}'.format(name)):
                    if name == 'name':
                        self.assertEqual(net_var, expect_var)
                    else:
                        if expect_var is None:
                            self.assertIsNone(net_var)
                        else:
                            expect_var = expect_var.clone().numpy()
                            self.assertIsInstance(net_var, np.ndarray)
                            self.assertTrue(np.all(net_var.shape == expect_var.shape))
                            self.assertTrue(np.all(net_var == expect_var))


        for net_vars, expecteds in zip(self.network.get_parameters(numpy=False), expecteds_1):
            for name, net_var, expect_var in zip(['name', 'output', 'w', 'b', 'dw', 'db'],
                                                    net_vars,
                                                    expecteds):
                print('get_parameters numpy=False :', name)
                sys.stdout.flush()
                with self.subTest('get_parameters numpy=False {}'.format(name)):
                    if name == 'name':
                        self.assertEqual(net_var, expect_var)
                    else:
                        if expect_var is None:
                            self.assertIsNone(net_var)
                        else:
                            self.assertIsInstance(net_var, torch.FloatTensor)
                            self.assertEqual(net_var.data_ptr, expect_var.data_ptr)


    def _test_02_update_target_network_fill_params(self, fill_net_vars, fill_target_vars):
        for i, net_vars in enumerate(zip(self.trainer.network.get_parameters(numpy=False),
                                         self.trainer.target_network.get_parameters(numpy=False))):
            network_vars, target_network_vars = net_vars
            self.assertEqual(network_vars[0], target_network_vars[0])
            if not re.match('^relu.$', network_vars[0]):
                for net_nm, net_var, target_var in zip(['w', 'b', 'dw', 'db'],
                                            network_vars[2:],
                                            target_network_vars[2:]):
                    with self.subTest('{} before update_target_network setup : fill params {} {}'.format(i, net_nm, network_vars[0])):
                        net_var.fill_(fill_net_vars)
                        target_var.fill_(fill_target_vars)


    def _test_04_save_load(self):
    
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

        import os
        from datetime import datetime

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

    def _test_05_add_learning_before(self):

        org_qLearnMinibatch = self.trainer.qLearnMinibatch
        
        def inject_qLearnMinibatch(x):
            org_qLearnMinibatch(x)

        self.patcher = patch.object(self.trainer, 'qLearnMinibatch' , wraps=inject_qLearnMinibatch)
        self.mock_qLearnMinibatch = self.patcher.start()

    def _test_05_add_learning_sumarries_get_expected(self):
        
        from visualizer import LearningVisualizer
        
        params = {}
        
        x, _ = self.mock_qLearnMinibatch.call_args
        s, a, r, s2, t = x[0]
        params['addInputImages_network'] = s
        params['addInputImages_target_network'] = s2
        
        addNetworkParameters = {}
        for layer_params in self.trainer.network.get_parameters():
            addNetworkParameters[layer_params[0]] = layer_params[1:]

        params['addNetworkParameters'] = addNetworkParameters
        
        addTargetNetworkParameters = {}
        for layer_params in self.trainer.target_network.get_parameters():
            addTargetNetworkParameters[layer_params[0]] = layer_params[1:]

        params['addTargetNetworkParameters'] = addTargetNetworkParameters

        addGetQUpdateValues = {}
        for name in LearningVisualizer.GET_Q_UPDATE_VALUE_NAMES:
            addGetQUpdateValues[name] = getattr(self.trainer, name).numpy()

        params['addGetQUpdateValues'] = addGetQUpdateValues

        addRMSpropValues = {}
        for name in LearningVisualizer.RMS_PROP_VALUE_NAMES[:4]:
            addRMSpropValues[name] = getattr(self.trainer, name).numpy()
        addRMSpropValues[LearningVisualizer.RMS_PROP_VALUE_NAMES[-1]] = self.trainer.args.lr

        params['addRMSpropValues'] = addRMSpropValues

        return params

    def _test_05_add_learning_after(self):
        self.patcher.stop()