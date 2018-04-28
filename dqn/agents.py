import sys
import numpy as np

if sys.path.count('../') == 0:
    sys.path.append('../')


class DQNAgent:

    def __init__(self, args):
        import common
        from transition_table import TransitionTable
        from utils import get_random
        from collections import deque

        # copy config props
        for key, value in args.items():
            setattr(self, key ,value)

        self.ep         = self.ep_start

        # create Q network ,target Q network and Q network trainer
        self.network, self.trainer = common.create_networks(args)

        # Load preprocessing network.
        PreprocessClass = common.get_preprocess(args.preproc)
        self.preproc = PreprocessClass(args.input_height, args.input_width, args)
        assert self.preproc is not None

        if args.backend == 'tensorflow':
            from common import get_extensions
            Ex = get_extensions(args)
            Ex.variable_initialization()

        # Load network parameters.
        if not args.file_name is None:
            self.network.load(args.file_name)
            self.trainer.update_target_network()

        if args.maximization == 'agent':
            from alewrap_py.game_screen import GameScreen
            self._screen = GameScreen(args)
        else:
            class Nop:
                def paint(self, img):
                    self.img = img
                def grab(self):
                    return self.img
            self._screen = Nop()

        # create transition table
        shapes = {'s': (self.input_height, self.input_width),
                  'a': (1, ),
                  'r': (1, ),
                  't': (1, )}

        self.transitions = TransitionTable(args, shapes)

        self.numSteps = 0 # Number of perceived states.
        self.lastState = None
        self.lastAction = None
        self.v_avg = 0 # V running average.
        self.tderr_avg = 0 # TD error running average.
        self.max_q2 = 0
        self.avg_q2 = 0
        self.q2 = np.array([])

        self.q_max = 1.0
        self.r_max = self.max_reward

        self.normalize = args.screen_normalize == 'agent'
        self.random = get_random()

        if self.target_q:
            self.trainer.update_target_network()

        self._do_greedy = DefaultGreedyMethod(self)

    def _preprocess(self, rawstate):
        self._screen.paint(rawstate)
        rawstate = self._screen.grab()
        if self.preproc:
            s = self.preproc.forward(rawstate)
            return s

        return rawstate

    def _qLearnMinibatch(self):
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)

        if self.rescale_r:
            r = r.astype(np.float32) / self.r_max

        if self.training_summary == 'non':
            do_summary = False
        elif self.training_summary == 'as_prog_freq':
            do_summary = self.numSteps % self.prog_freq == 0
        elif self.training_summary == 'all':
            do_summary = True
        else:
            assert False

        self.trainer.qLearnMinibatch([s, a, r, s2, term], do_summary)

        if do_summary:
            self.trainer.add_learning_summaries(self.numSteps)

    def _sample_validation_data(self):

        s, a, r, s2, term = self.transitions.sample(self.valid_size)

        self.valid_s    = s
        self.valid_a    = a
        self.valid_r    = r
        self.valid_s2   = s2
        self.valid_term = term

    def compute_validation_statistics(self):

        if self.rescale_r:
            self.valid_r = self.valid_r.astype(np.float32) / self.r_max

        delta, q_max =self.trainer.compute_validation_statistics([self.valid_s, self.valid_a, self.valid_r, self.valid_s2, self.valid_term])

        self.v_avg = self.q_max * np.mean(q_max)
        self.tderr_avg = np.mean(np.abs(delta))

    def perceive(self, rawstate, rawreward, terminal, testing, testing_ep):

        state = self._preprocess(rawstate)

        curState = None

        reward = rawreward
        if not self.normalized_dqn:

            if self.clip_reward:
                reward = min(max(rawreward, -self.clip_reward), self.clip_reward)

            if  self.rescale_r:
                self.r_max = max(self.r_max, reward)

        self.transitions.add_recent_state(state, terminal)

        currentFullState = self.transitions.get_recent()

        if self.lastState is not None and not testing:
            self.transitions.add(self.lastState, self.lastAction, reward, self.lastTerminal)

        if self.numSteps == self.learn_start + 1 and not testing:
            self._sample_validation_data()

        curState = self.transitions.get_recent()
        curState = curState.reshape((1,)+self.input_dims) 

        if terminal:
            action = -1 
        else:
            action = self._eGreedy(curState, testing_ep)

        if self.numSteps > self.learn_start and not testing and\
            self.numSteps % self.update_freq == 0:
            for _ in range(self.n_replay):
                self._qLearnMinibatch()

        if not testing:
            self.numSteps += 1

        self.lastState = state
        self.lastAction = action
        self.lastTerminal = terminal

        if not self.target_q is None and self.numSteps % self.target_q == 1:
            print('network W -> terget network W')
            self.trainer.update_target_network()

        return 0 if terminal else action

    def _eGreedy(self, state, testing_ep):
        if self._do_greedy(testing_ep):
            return self._greedy(state)
        else:
            self.q = [0 for _ in range(self.n_actions)]
            return self.random.random(0, self.n_actions)

    def _greedy(self, state):

        if len(self.state_dim) == 2:
            assert False, 'Input must be at least 3D'

        self.q = self.network.forward(state)[0]

        # get action
        if self.use_tie_break:
            # Evaluate all other actions (with random tie-break)
            maxq = self.q.max()
            results = np.where(self.q == maxq)
            besta = results[0]

            r = self.random.random(len(besta))

            self.lastAction = besta[r]

        else:
            # only argmax
            self.lastAction = np.argmax(self.q)

        return self.lastAction


    def save_network(self, filepath):
        self.network.save(filepath)


# epsilon-greedy methods
class BaseGreedyMethod(object):
    def __init__(self, agent):
        self.agent = agent

    def now_greeding(self):
        return self.agent.ep > self.agent.ep_end

    def __call__(self, testing_ep):
        if self.agent.not_use_egreedy:
            # use in debug.
            print('skip egreedy')
            return True
        return self.call(testing_ep)

    def call(self, testing_ep):
        raise NotImplementedError
  
class DefaultGreedyMethod(BaseGreedyMethod):
    def __init__(self, agent):
        super(DefaultGreedyMethod, self).__init__(agent)

    def call(self, testing_ep):
        self.agent.ep = testing_ep if testing_ep is not None \
                  else (self.agent.ep_end + \
                        max(0, (self.agent.ep_start - self.agent.ep_end) * (self.agent.ep_endt - \
                        max(0, self.agent.numSteps - self.agent.learn_start))/self.agent.ep_endt))
        return self.agent.random.uniform() >= self.agent.ep

