import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from alewrap_py import Render, ACTION_MEANING

if sys.path.count('../') == 0:
    sys.path.append('../')

from utils.tensorboard_logger import Logger

_WRITER = None
_FAMILIES = [None, '1.current_state', '2.network', '3.target_network', '4.getQUpdate', '5.RMSprop', '6.evaluation', '7.validation']

class Visualizer(object):
    
    def __init__(self, opt):
        global _WRITER
        if _WRITER is None:
            _WRITER = tf.summary.FileWriter(opt.log_dir, flush_secs=2, max_queue=2000)
        if opt.backend == 'tensorflow':
            from keras import backend as K
            _WRITER.add_graph(K.get_session().graph)

        self.logger = Logger(_WRITER)
    
    def add_graph(self, graph):
        global _WRITER
        _WRITER.add_graph(graph)

    def _validate_values(self, names, values):
        if len(names) != len(values) or \
            np.any(np.array([not k in names for k in list(values.keys())])):
            raise ValueError('given invalid value name, expect {} got {}'.format(names, list(values.keys())))

    def _add_scaler(self, family_id, tag, value):
        if value is None:
            return
        tag = '{}/{}'.format(_FAMILIES[family_id], tag)
        self.logger.add_scalar(tag, value)

    def _add_max_scaler(self, family_id, tag, value):
        if value is None:
            return
        tag = '{}/{}/max'.format(_FAMILIES[family_id], tag)
        self.logger.add_scalar(tag, value.max())

    def _add_min_scaler(self, family_id, tag, value):
        if value is None:
            return
        tag = '{}/{}/min'.format(_FAMILIES[family_id], tag)
        self.logger.add_scalar(tag, value.min())

    def _add_avg_scaler(self, family_id, tag, value):
        if value is None:
            return
        tag = '{}/{}/avg'.format(_FAMILIES[family_id], tag)
        self.logger.add_scalar(tag, np.mean(value))

    def _add_histogram(self, family_id, tag, value):
        if value is None:
            return
        tag = '{}/{}'.format(_FAMILIES[family_id], tag)
        self.logger.add_histogram(tag, value)

    def _add_images(self, family_id, tag, images):
        if images is None:
            return
        tag = '{}/{}'.format(_FAMILIES[family_id], tag)
        self.logger.add_images(tag, images)

    def flush(self, step):
        self.logger.flush(step)

class EvaluationVisualizer(Visualizer):
    EVAL_VALUE_NAMES = ['episode_score', 'episode_count', ]
    VALID_VALUE_NAMES = ['TDerror', 'V']
    def __init__(self, opt):
        super(EvaluationVisualizer, self).__init__(opt)

    def addEvaluation(self, values):
    
        self._validate_values(EvaluationVisualizer.EVAL_VALUE_NAMES, values)

        for name, val in values.items():
            if name == 'episode_score':
                self._add_max_scaler(6, name, val)
                self._add_min_scaler(6, name, val)
                self._add_avg_scaler(6, name, val)
            else:
                self._add_scaler(6, name, val)

    def addValidation(self, values):
    
        self._validate_values(EvaluationVisualizer.VALID_VALUE_NAMES, values)

        for k, v in values.items():
            self._add_scaler(7, k, v)


class CurrentStateVisualizer(Visualizer):
    CURRENT_STATE_VALUE_NAMES = ['average_episode_scores', 'episode_count', 'epsilon']
    def __init__(self, opt):
        super(CurrentStateVisualizer, self).__init__(opt)

    def addCurrentState(self, values):

        self._validate_values(CurrentStateVisualizer.CURRENT_STATE_VALUE_NAMES, values)

        for name, val in values.items():
            self._add_scaler(1, name, val)




class LearningVisualizer(Visualizer):
    GET_Q_UPDATE_VALUE_NAMES = ['q2_all', 'q2_max', 'q2', 'r', 'q_all', 'q', 'delta', 'targets']
    RMS_PROP_VALUE_NAMES = ['g', 'g2', 'tmp', 'deltas', 'learning_rate']
    def __init__(self, opt, layer_names):
        super(LearningVisualizer, self).__init__(opt)
        self.layer_names = layer_names
    
    def addInputImages(self, family_id, s):
        assert family_id in [2, 3]
        self._add_images(family_id, 'input', s)

    def addTargetNetworkParameters(self, parameters):
        self._addParameters(3, parameters)
        
    def addNetworkParameters(self, parameters):
        self._addParameters(2, parameters)

    def _addParameters(self, family_id, parameters):
        if not family_id in [2, 3]:
            raise ValueError('given invalid family_id, expect 2 or 3, got {}'.format(family_id))

        self._validate_values(self.layer_names, parameters)

        for layer_name in self.layer_names:
            params = parameters.get(layer_name)

            for p, p_name in zip(params, ['output', 'weight', 'bias', 'weight/gradients', 'bias/gradients']):
                if p is None:
                    continue
                tag = '{}/{}'.format(layer_name, p_name)
                self._add_max_scaler(family_id, tag, p)
                self._add_min_scaler(family_id, tag, p)
                self._add_avg_scaler(family_id, tag, p)
                self._add_histogram(family_id, tag, p)

            if layer_name == 'q_all':
                tag = '{}/action'.format(layer_name)
                q_all = np.mean(params[0], axis=0)
                for i in range(len(q_all)):
                    self._add_scaler(family_id, '{}/{}'.format(tag, i), q_all[i])


    def addGetQUpdateValues(self, values):

        self._validate_values(LearningVisualizer.GET_Q_UPDATE_VALUE_NAMES, values)

        for val_name, val in values.items():
            if val is None:
                continue
            self._add_max_scaler(4, val_name, val)
            self._add_min_scaler(4, val_name, val)
            self._add_avg_scaler(4, val_name, val)
            self._add_histogram(4, val_name, val)
    
    def addRMSpropValues(self, values):

        self._validate_values(LearningVisualizer.RMS_PROP_VALUE_NAMES, values)

        for val_name, val in values.items():
            if val is None:
                continue
            if val_name == 'learning_rate':
                self._add_scaler(5, val_name, val)
            else:
                self._add_max_scaler(5, val_name, val)
                self._add_min_scaler(5, val_name, val)
                self._add_avg_scaler(5, val_name, val)
                self._add_histogram(5, val_name, val)
    

class QValueVisualizer(Render):

    def __init__(self, args):
        self.test_recording_q_value = args.test_recording_q_value
        self.show_q_value = args.test_recording_q_value_show

        self.action = 0
        self.actions = args.actions
        self.n_actions = args.n_actions
        self.action_meaning = ACTION_MEANING[self.actions]
        self.index=np.arange(self.n_actions)
        os.environ['DISPLAY'] = ':0'

        import matplotlib.pyplot as plt
        self.plt = plt
        self.q_values = np.zeros((5000, self.n_actions+4), dtype=np.float16)
        self.q_value = np.zeros((self.n_actions, ), dtype=np.float32)
        self.action_indexes = np.zeros((self.n_actions, ), dtype=np.uint8)

    def collect_q_value(self, q_value):
        if isinstance(q_value, list):
            q_value = np.array(q_value)
        self.q_value = q_value

    def step(self, frame, action, reward, term, info, episode_score, total_steps):
        self.total_steps = total_steps
        if self.test_recording_q_value:
            self.action = action
            if isinstance(self.q_value, list):
                q_value = np.array(self.q_value)
            if self.total_steps >= len(self.q_values):
                self.q_values = np.concatenate((self.q_values, np.zeros((5000, self.n_actions+4), dtype=np.float16)), axis=0)
            self.q_values[self.total_steps, 0] = self.episode_id
            self.q_values[self.total_steps, 3] = action
            self.q_values[self.total_steps, 4:] = self.q_value.astype(np.float16)
            self.action_indexes[action] += 1
            self.q_values[self.total_steps, 1] = info['ale.lives']
            self.q_values[self.total_steps, 2] = episode_score

    def write_summmary(self, f):
        if self.test_recording_q_value:
            f.write('---------------------------\n')
            f.write('action count\n')
            for i, num in enumerate(self.action_indexes):
                f.write('{}:{}\n'.format(i, num))
            f.write('---------------------------\n')
            f.write('Q Values\n')
            random_step_count = 0
            for epi_q_value in self.q_values[:self.total_steps]:
                epi_id = epi_q_value[0]
                lives = epi_q_value[1]
                score = epi_q_value[2]
                act = epi_q_value[3]
                q_value = epi_q_value[4:]
                q_val_str = ''
                for qv in q_value:
                    q_val_str = '{}, {}'.format(q_val_str, qv)
                f.write('episode:{} lives:{} score:{} act:{} q_value:{}\n'.format(int(epi_id), int(lives), score, int(act), q_val_str))
                if np.all(q_value == 0):
                    random_step_count += 1
            f.write('---------------------------\n')
            f.write('random count:{}(%{:.2f})\n'.format(random_step_count, random_step_count/self.total_steps*100))

    def __call__(self, frame):

        if self.show_q_value:
            fig = self.plt.figure()

            ax1 = self.plt.subplot2grid((3,2), (0,0), rowspan=3)
            ax1.imshow(frame)
            ax1.set_title('Game Screen')

            actions = np.zeros(self.n_actions)
            actions[self.action] = 1
            ax2 = self.plt.subplot2grid((3,2), (0,1))
            ax2.bar(self.index, actions, 0.8, color='b',)
            ax2.set_xticks([])
            ax2.set_yticks([0,1])
            ax2.set_title('Selected action')

            ax3 = self.plt.subplot2grid((3,2), (1,1), rowspan=2)
            q_value = self.q_values[self.total_steps, 4:]
            q_max = q_value.max()
            bars = ax3.bar(self.index, q_value, 0.8, alpha=0.5)
            for i, bar in enumerate(bars):
                bar.set_facecolor('r' if q_max == q_value[i] else 'b')

            ax3.set_xticks(self.index)
            ax3.set_xticklabels(self.action_meaning)
            ax3.set_title('Q-Value')

            self.plt.tight_layout()
            fig.canvas.draw()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            if self.show_q_value:
                cv2.imshow('Q Value', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
            self.plt.close()
        return frame
