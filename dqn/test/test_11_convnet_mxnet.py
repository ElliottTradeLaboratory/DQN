import sys
import unittest
from unittest.mock import *

import numpy as np
from torch.utils.serialization import load_lua

import mxnet as mx

if sys.path.count('../') == 0:
    sys.path.append('../')
if sys.path.count('../../') == 0:
    sys.path.append('../../')

from testutils import *
from utils import *

class TestMXnetTrainer(unittest.TestCase):

    def setUp(self):
        get_random('pytorch', 1)

    def test_00_init_network(self):
        
        from common import create_networks
        from convnet_common import Convnet, Trainer
        
        from config import get_opt
        sys.argv += ['--backend', 'mxnet',
                     '--env', 'breakout',
                     '--logdir', '/tmp',
                     '--gpu', '-1',
                     '--random_type', 'pytorch',
                     '--loss_function', 'DQN3.0',
                     '--optimizer', 'DQN3.0',
                     #'--tf_debug'
                     ]
        args = get_opt()
        from initenv import setup
        
        game_env, agent, game_actions, args = setup(args)
        trainer = agent.trainer
        network = agent.network
        
        get_random().manualSeed(1)

        arg_dict, grad_dict = network.module.get_parameters()

        from convnet_pytorch import PyTorchConvnet
        args.backend='pytorch_legacy'
        pytorch_network = PyTorchConvnet(args, "network")

        py_params, _ = pytorch_network.model.parameters()

        for py, p in zip(py_params, list(arg_dict.items())[1:]):
            print('******* mxnet {}******'.format(p[0]))
            print(p[1])
            print('******* pytorch {}******'.format(p[0]))
            print(py.numpy())
            #assert_equal(p[1].asnumpy(), py.numpy(), verbose=1, msg=p[0])

    def test_01_update_target(self):

        
        from common import create_networks
        from convnet_common import Convnet, Trainer
        
        from config import get_opt
        sys.argv += ['--backend', "mxnet",
                     '--env', 'breakout',
                     '--logdir', '/tmp',
                     '--gpu', '0',
                     '--random_type', 'torch',
                     '--loss_function', 'DQN3.0',
                     '--optimizer', 'DQN3.0',
                     #'--tf_debug'
                     ]
        args = get_opt()
        args.n_actions = 4
        network, trainer = create_networks(args)
    
        for n_params, t_params in zip(trainer.network.get_trainable_parameters(numpy=True),
                                      trainer.target_network.get_trainable_parameters(numpy=True)):
            assert_not_equal(n_params[1:3], t_params[1:3])
            if n_params[3] is not None and not np.all(n_params[3] == 0) and not np.all(t_params[3] == 0):
                assert_not_equal(n_params[3], t_params[3])
            if n_params[4] is not None and not np.all(n_params[4] == 0) and not np.all(t_params[4] == 0):
                assert_not_equal(n_params[4], t_params[4])

        trainer.update_target_network()

        for n_params, t_params in zip(trainer.network.get_trainable_parameters(numpy=True),
                                      trainer.target_network.get_trainable_parameters(numpy=True)):
            assert_equal(n_params[1:], t_params[1:])

    def test_02_forward(self):
        from common import create_networks
        from convnet_common import Convnet, Trainer
        from convnet_mxnet import MODE_LEARNING, MODE_PREDICT, MODE_VALIDATE, MODE_OUTPUTS
        from config import get_opt
        
        sys.argv += ['--backend', 'mxnet',
                     '--env', 'breakout',
                     '--logdir','/tmp',
                     ]

        args = get_opt()
        args.n_actions = 4


        network, trainer = create_networks(args)

        fname = './dqn3.0_dump/cpu_batch_{:010d}.dat'.format(50004)
        batch = load_lua(fname)
        
        s, a, r, s2, term = [v.numpy() for v in batch]
        s = s.reshape(32, 4, 84, 84)
        s2 = s2.reshape(32, 4, 84, 84)
        a -= 1

        predict = network.forward([s[0]], MODE_PREDICT)

        print(predict)
        return
        learning = network.forward(s, MODE_LEARNING)

        assert_equal(predict, learning[0].asnumpy())

        valid_s = np.vstack((s,)*16)
        validate = network.forward(valid_s[:500], MODE_VALIDATE)

        assert_equal(predict, validate[0].asnumpy())

        trainer.qLearnMinibatch([s, a, r, s2, term], True)

        predict2 = network.forward([s[0]], MODE_PREDICT)

        assert_not_equal(predict, predict2)

        learning = network.forward(s, MODE_LEARNING)

        assert_equal(predict2, learning[0].asnumpy())

        validate = network.forward(valid_s[:500], MODE_VALIDATE)

        assert_equal(predict2, validate[0].asnumpy())

    def test_03_qLearnMinibatch(self):
        
        from common import create_networks
        from convnet_common import Convnet, Trainer
        
        from config import get_opt
        sys.argv += ['--backend', 'mxnet',
                     '--env', 'breakout',
                     '--logdir', '/tmp',
                     '--gpu', '-1',
                     '--random_type', 'pytorch',
                     '--loss_function', 'DQN3.0',
                     '--optimizer', 'DQN3.0',
                     #'--tf_debug'
                     ]
        args = get_opt()
        from initenv import setup
        
        game_env, agent, game_actions, args = setup(args)
        trainer = agent.trainer

        args.backend = 'pytorch_legacy'
        args.optimizer = 'DQN3.0'
        p_network, p_trainer = create_networks(args)
        
        trainer.update_target_network()

        for step in range(50004, 50016, 4):
            print('@@@@@@@@@@@@@ {} @@@@@@@@@@@@@@@@'.format(step))

            fname = './dqn3.0_dump/cpu_batch_{:010d}.dat'.format(step)
            batch = load_lua(fname)
            
            s, a, r, s2, term = [v.numpy() for v in batch]
            s = s.reshape(32, 4, 84, 84)
            s2 = s2.reshape(32, 4, 84, 84)
            a -= 1

            trainer.qLearnMinibatch([s, a, r, s2, term], True)
            p_trainer.qLearnMinibatch([s, a, r, s2, term], True)

            #print('q2_all\n', trainer.q2_all)
            #print('py q2_all\n', p_trainer.q2_all)
            #print('q2_max\n', trainer.q2_max)
            #print('q2\n', trainer.q2)
            #print('r\n', trainer.r)
            print('q_all\n', trainer.q_all)
            print('py q_all\n', p_trainer.optimizer.q_all.numpy())
            #print('q\n', trainer.q)
            #print('delta\n', trainer.delta)
            #print('targets\n', trainer.targets)
            #o, w, b, dw, db =trainer.network._get_params_dict(True)['q_all']
            #print(db)
            #fname = './dqn3.0_dump/getQUpdate_{:010d}.dat'.format(step)
            #dqn_getQUpdate = [v.numpy() for v in load_lua(fname)]
            #print('dqn3.0')
            #print(dqn_getQUpdate[-1])
            #print(dqn_getQUpdate[-1].sum(0))
            break
        #trainer.add_learning_summaries(1000)
