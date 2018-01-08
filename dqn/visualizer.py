import sys
import numpy as np
import tensorflow as tf

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
    

