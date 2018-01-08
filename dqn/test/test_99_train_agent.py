import sys
import unittest
from unittest.mock import *

import numpy as np
from torch.utils.serialization import load_lua

sys.path.append('../')
sys.path.append('../../')

from utils import Namespace, get_random
from visualizer import CurrentStateVisualizer, EvaluationVisualizer
from testutils import *

class TestTrainAgent(unittest.TestCase):

    def test_01(self):
        
        opt = Namespace()
        opt.env = 'breakout'
        opt.backend = 'pytorch_legacy'
        opt.actrep = 4
        opt.random_starts = 30
        opt.steps= 15
        opt.eval_freq =  7
        opt.eval_steps = 5
        opt.save_freq =  20
        opt.prog_freq =   6
        opt.learn_start = 10
        opt.log_dir = '/tmp'
        opt.step_train_mode = 1

        get_random('pytorch', 1)

        s = np.zeros((210,160,3), dtype=np.float32)
        mock = MagicMock()
        mock.getState.return_value = [s+1, 0, False, {}]
        mock.perceive.side_effect = [i for i in range(0,4)]*5
        mock.step.side_effect=[
            [s+2, 1, False, {}],
            [s+3, 2, False, {}],
            [s+4, 3, False, {}],
            [s+5, 4, True, {}],
            [s+7, 6, False, {}],
            [s+8, 7, False, {}],
            [s+9, 8, False, {}],
            [s+10, 9, False, {}],
            [s+11, 10, False, {}],
            [s+12, 11, False, {}],
            [s+13, 12, False, {}],
            [s+14, 13, False, {}],
            [s+15, 14, False, {}],
            [s+16, 15, True, {}],
            [s+18, 17, False, {}],
            [s+19, 18, False, {}],
            [s+20, 19, False, {}],
            [s+21, 20, False, {}],
            [s+22, 21, False, {}],
            [s+23, 22, False, {}],
            [s+24, 23, False, {}],
            [s+25, 24, False, {}],
            [s+26, 25, False, {}],
            [s+27, 26, False, {}],
            [s+28, 27, False, {}],
        ]
        mock.nextRandomGame.side_effect= [
            [s+6, 5, False, {}],
            [s+17, 16, False, {}],
        ]
        mock.newGame.side_effect= [
            [s+15, 14, False, {}],
        ]
        ep = np.around(np.arange(1.0, 0.0, -0.1),1).tolist()
        ep =np.array(ep)
        ep = np.column_stack((ep, ep)).flatten()
        type(mock).ep = PropertyMock(side_effect=ep)
        type(mock).numSteps =  PropertyMock(side_effect=[i for i in range(6,24, 6)])
        type(mock).v_avg =  PropertyMock(return_value=1)
        type(mock).tderr_avg =  PropertyMock(return_value=2)
        type(mock).q_max =  PropertyMock(return_value=3)
        type(mock).episode_scores =  [1,2,3,4]
        type(mock).lr =  0.00025

        @patch('train_agent.CurrentStateVisualizer', return_value=mock)
        @patch('train_agent.EvaluationVisualizer', return_value=mock)
        def test_train_main(mock_cur_vis, mock_eval_vis):

            import train_agent
            
            train_agent.CurrentStateVisualizer.CURRENT_STATE_VALUE_NAMES = CurrentStateVisualizer.CURRENT_STATE_VALUE_NAMES
            train_agent.EvaluationVisualizer.EVAL_VALUE_NAMES = EvaluationVisualizer.EVAL_VALUE_NAMES
            train_agent.EvaluationVisualizer.VALID_VALUE_NAMES = EvaluationVisualizer.VALID_VALUE_NAMES
            
            train_agent.train_main(mock, mock, [0,1,3,4], opt)
        
        test_train_main()

        expected = [
            ('getState', (),{}),
            ('perceive', (s+1, 0, False), dict(testing=False, testing_ep=None)), # step 1
            ('step', (0,), dict(training=1)),
            ('perceive', (s+2, 1, False), dict(testing=False, testing_ep=None)), # step 2
            ('step', (1,), dict(training=1)),
            ('perceive', (s+3, 2, False), dict(testing=False, testing_ep=None)), # step 3
            ('step', (3,), dict(training=1)),
            ('perceive', (s+4, 3, False), dict(testing=False, testing_ep=None)), # step 4
            ('step', (4,), dict(training=1)),
            ('perceive', (s+5, 4, True), dict(testing=False, testing_ep=None)),  # step 5
            # In the training mode, train_main() calls nextRandomGame() after perceive()
            # in the next step when got terminal state from step()
            # because the network needs save the terminal state
            # into the transition table at perceive() for learning it.
            ('nextRandomGame', (),{}),                                           # reset game.
                                                                                 # progress output
            ('perceive', (s+6, 5, False), dict(testing=False, testing_ep=None)), # step 6
            ('step', (1,), dict(training=1)),
            ('addCurrentState',(dict(zip(['average_episode_scores', 'episode_count', 'epsilon'],[10.0, 1, 1.0])),), {}),
            ('flush', (6,), {}),
            ('perceive', (s+7, 6, False), dict(testing=False, testing_ep=None)), # step 7 eval_freq True but not in learning fase.
            ('step', (3,), dict(training=1)),
            ('perceive', (s+8, 7, False), dict(testing=False, testing_ep=None)), # step 8
            ('step', (4,), dict(training=1)),
            ('perceive', (s+9, 8, False), dict(testing=False, testing_ep=None)), # step 9
            ('step', (0,), dict(training=1)),
            ('perceive', (s+10, 9, False), dict(testing=False, testing_ep=None)), # step 10
            ('step', (1,), dict(training=1)),
            ('perceive', (s+11, 10, False), dict(testing=False, testing_ep=None)), # step 11
            ('step', (3,), dict(training=1)),
            ('perceive', (s+12, 11, False), dict(testing=False, testing_ep=None)),  # step 12
            ('step', (4,), dict(training=1)),
            ('addCurrentState',(dict(zip(['average_episode_scores', 'episode_count', 'epsilon'],[57.0, 1, 0.9])),), {}),
            ('flush', (12,), {}),
            ('perceive', (s+13, 12, False), dict(testing=False, testing_ep=None)),  # step 13
            ('step', (0,), dict(training=1)),
            ('perceive', (s+14, 13, False), dict(testing=False, testing_ep=None)),  # step 14
            ('step', (1,), dict(training=1)),
            ('start_recording', (14,), dict()),                                               # start evaluation
            ('newGame', (), dict()),
            ('perceive', (s+15, 14, False), dict(testing=True, testing_ep=0.05)),   # eval step 1
            ('step', (3,), dict(training=0)),
            # But, in the testing mode,  train_main() calls nextRandomGame() immediately
            # when got terminal state from step() because the network has not learning.
            ('nextRandomGame', (),{}),                                             # reset game.
            ('perceive', (s+17, 16, False), dict(testing=True, testing_ep=0.05)), # eval step 2
            ('step', (4,), dict(training=0)),
            ('perceive', (s+18, 17, False), dict(testing=True, testing_ep=0.05)), # eval step 3
            ('step', (0,), dict(training=0)),
            ('perceive', (s+19, 18, False), dict(testing=True, testing_ep=0.05)), # eval step 4
            ('step', (1,), dict(training=0)),
            ('perceive', (s+20, 19, False), dict(testing=True, testing_ep=0.05)), # eval step 5
            ('step', (3,), dict(training=0)),
            ('stop_recording', (), dict()),
            ('compute_validation_statistics', (), {}),
            ('addEvaluation',  (dict(episode_score=np.array([1,2,3,4]), episode_count=4),), {}),
            ('addValidation', (dict(TDerror=2, V=1),), {}),
            ('flush', (14,), {}),
            ('perceive', (s+21, 20, False), dict(testing=False, testing_ep=None)), # step 15
            ('step', (4,), dict(training=1)),
            ('save_network', ('/tmp/breakout_pytorch_network_step{:010d}.dat'.format(opt.steps),), {}),
            ]
        
        self.assertEqual(len(mock.method_calls), len(expected))
        for i, calls in enumerate(zip(mock.method_calls, expected)):
            print(i, mock.method_calls[i][0], expected[i][0])
            assert_call_equal(*calls, str(i))

    
    
    def test_02(self):
    
        from initenv import setup
        from config import get_opt
        from collections import deque

        sys.argv += ['--logdir', '/tmp', '--backend', 'pytorch_legacy', '--env','breakout']

        opt = get_opt()
        game_env, agent, game_actions, opt = setup(opt)

        org_perceive = agent.perceive

        steps = 0
        dqn_vars = deque(load_lua('./dqn3.0_dump/0-50Kstep_{:04d}.dat'.format(1000)))
        dat_step = deque([step for step in range(2000, 51000, 1000)])

        def inject_agent_perceive(screen, reward, terminal, testing, testing_ep):

            nonlocal dqn_vars, steps, dat_step

            steps += 1

            action_index = org_perceive(screen, reward, terminal, testing, testing_ep)

            dqn_s, dqn_a, dqn_r, dqn_t = dqn_vars.popleft()
            dqn_s = dqn_s.numpy()
            dqn_s = np.transpose(dqn_s, (0, 2,3,1))
            dqn_s = dqn_s.reshape((210, 160, 3))
            dqn_a = dqn_a if dqn_a >= 0 else 0

            assert float_equal(screen, dqn_s)
            assert action_index ==  dqn_a, '[{}] {} vs {}'.format(steps, action_index, dqn_a)
            assert reward == dqn_r
            assert terminal == dqn_t


            if len(dat_step) > 0 and len(dqn_vars) == 0:
                stp = dat_step.popleft()
                filename = './dqn3.0_dump/0-50Kstep_{:04d}.dat'.format(stp)
                print('loading', filename)
                dqn_vars = deque(load_lua(filename))
                print('done')


            return action_index


        dat_step2 = deque([step for step in range(1000, 51000, 1000)])

        def inject_addCurrentState(values):

            nonlocal dat_step2

            dqn_vars = load_lua('./dqn3.0_dump/current_state_{:010d}.dat'.format(dat_step2.popleft()))
            
            for key, dqn_var in dqn_vars.items():
                assert key in values, '[{}] {} no in values{}'.format(steps, key, values.keys())
                
                assert values[key] == dqn_var, '[{}] {} {} vs {}'.format(steps, key, values[key], dqn_var)

        @patch('train_agent.CurrentStateVisualizer.addCurrentState', wraps=inject_addCurrentState)
        @patch.object(agent, 'perceive', wraps=inject_agent_perceive)
        def test(mock_perceive, mock_cur_vis):

            import train_agent
            
            opt.steps = 20000
            
            train_agent.train_main(game_env, agent, game_actions, opt)

        test()

    def test_03(self):

        sys.argv += ['--backend','pytorch_legacy',
                     '--env', 'breakout',
                     '--step_train_mode', '0',
                     '--steps', '2000',
                     '--debug']
        from config import get_opt
        from initenv import setup
        opt = get_opt()
        game_env, agent, game_actions, opt = setup(opt)

        get_random('pytorch', 1)

        org_step = game_env.env.step
        step_returns = []
        def inject_step(action, training):
            nonlocal step_returns
            step_returns.append(org_step(action, training))
            return step_returns[-1]

        org_envStep = game_env.env.env.envStep
        def inject_envStep(action):
            return org_envStep(action)

        @patch.object(game_env.env.env, 'envStep', wraps=inject_envStep)
        @patch.object(game_env.env, 'step', wraps=inject_step)
        def test_train_main(mock_step, mock_envStep):

            import train_agent
            
            train_agent.train_main(game_env, agent, game_actions, opt)

        test_train_main()

        self.assertGreater(len(step_returns), 0)
        for ret in step_returns:
            s, r, t, info = ret
            
            if t:
                self.assertEqual(info['ale.lives'], 0)

