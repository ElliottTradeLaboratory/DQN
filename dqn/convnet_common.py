import re
import numpy as np
from visualizer import LearningVisualizer

BASE_LAYER_NAMES = ['Conv1', 'relu1', 'Conv2', 'relu2', 'Conv3', 'relu3', 'flatten', 'Linear', 'relu4', 'q_all']
BASE_EXCLUDE_LAYER_NAMES_FOR_SUMMARY = ['flatten']
class Convnet(object):
    def __init__(self, args,
                 network_name,
                 layer_names=BASE_LAYER_NAMES,
                 exclude_layer_name_for_summary=BASE_EXCLUDE_LAYER_NAMES_FOR_SUMMARY):
        self.args = args
        self.network_name = network_name
        self.layer_names = layer_names
        self.summarizable_layer_names = [lname for lname in layer_names if not lname in exclude_layer_name_for_summary]
        self.trainable_layer_names = [lname for lname in self.summarizable_layer_names if not re.match(r"^relu\d$", lname)]

    def forward(self, s):
        raise NotImplementedError()

    def get_summarizable_parameters(self, numpy=True):
        return self._get_selected_parameters(self.summarizable_layer_names, numpy)

    def get_trainable_parameters(self, numpy=True):
        return self._get_selected_parameters(self.trainable_layer_names, numpy)

    def set_trainable_parameters(self, weights, numpy=True):
        raise NotImplementedError()

    def _get_selected_parameters(self, selecter, numpy):
        params_dict = self._get_params_dict(numpy)
        return [params for layer_name, params in params_dict.items() if layer_name in selecter]

    def _get_params_dict(self, numpy):
        # This method returns OrderedDict of params that {layer_name:(output, w, b, dw, db),...}.
        # If numpy is True returns np.ndarray parameters otherwise returns native variable parameters for framework.
        # If layer does not have trainable parameters, w, b, dw and db are None.
        raise NotImplementedError()

    def save(self, filepath):
        self._save(filepath)

    def _save(self, filepath):
        raise NotImplementedError()

    def load(self, filepath):
        self._load(filepath)

    def _load(self, filepath):
        raise NotImplementedError()


class Trainer(object):

    def __init__(self, args, network, target_network):
        self.args = args
        self.network = network
        self.target_network = target_network

        self.learning_visualizer = LearningVisualizer(args, self.network.summarizable_layer_names)

    def qLearnMinibatch(self, x, do_summary):

        return self._qLearnMinibatch(x, do_summary)

    def add_learning_summaries(self, numSteps):
        
        dict_getQUpdate_values = dict(zip(LearningVisualizer.GET_Q_UPDATE_VALUE_NAMES,
                                          self._getQUpdate_values))
        dict_rms_prop_values = dict(zip(LearningVisualizer.RMS_PROP_VALUE_NAMES[:-1],
                                        self._rmsprop_values))
        dict_rms_prop_values[LearningVisualizer.RMS_PROP_VALUE_NAMES[-1]] = self.args.lr

        dict_network_parameters = dict(zip(self.network.summarizable_layer_names,
                                           self.network.get_summarizable_parameters(numpy=True)))
        dict_target_network_parameters = dict(zip(self.target_network.summarizable_layer_names,
                                                  self.target_network.get_summarizable_parameters(numpy=True)))

        self.learning_visualizer.addInputImages(2, self.s[0])
        self.learning_visualizer.addInputImages(3, self.s2[0])
        self.learning_visualizer.addGetQUpdateValues(dict_getQUpdate_values)
        self.learning_visualizer.addRMSpropValues(dict_rms_prop_values)
        self.learning_visualizer.addNetworkParameters(dict_network_parameters)
        self.learning_visualizer.addTargetNetworkParameters(dict_target_network_parameters)

        self.learning_visualizer.flush(numSteps)


    def compute_validation_statistics(self, x):

        ret= self._getQUpdate(x)

        return ret[1:3]

    def update_target_network(self):
        self._update_target_network()

    @property
    def _getQUpdate_values(self):
        raise NotImplementedError()

    @property
    def _rmsprop_values(self):
        raise NotImplementedError()

    def _getQUpdate(self, x):
        raise NotImplementedError()

    def _qLearnMinibatch(self, x, do_summary):
        raise NotImplementedError()

    def _update_target_network(self):
        raise NotImplementedError()

