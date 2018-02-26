from collections import deque
import sys
import os
import re
import numpy as np

from common import get_extensions
from convnet_common import Trainer, Convnet, BASE_LAYER_NAMES, BASE_EXCLUDE_LAYER_NAMES_FOR_SUMMARY
from visualizer import LearningVisualizer

class KerasConvnet(Convnet):

    def __init__(self, args, network_name):
        Convnet.__init__(self,
                         args,
                         network_name,
                         layer_names=['premute']+BASE_LAYER_NAMES,
                         exclude_layer_name_for_summary=['premute']+BASE_EXCLUDE_LAYER_NAMES_FOR_SUMMARY)

        self.Ex = get_extensions(self.args)
        self.model = self._create_model()
        
        self.outputs = []
        for layer in self.model.layers:
            if layer.name in self.summarizable_layer_names:
                self.outputs.append(layer.output)

    def _create_model(self):
        from keras import backend as K
        from keras.models import Model
        from keras.layers import Input, Dense, Flatten, Conv2D, Activation, Permute
        from initializer import get_initializer
        from utils import get_random


        with K.name_scope(self.network_name):
            
            initializer = get_initializer(self.args.initializer)
            padding = 'valid'
            layer_names = deque(self.layer_names)

            get_random().manualSeed(1)

            s = Input(shape=self.args.input_dims, name='s')

            premuted_s = Permute((2, 3, 1), name=layer_names.popleft())(s)
            Conv1 = Conv2D(32, (8,8), strides=(4, 4),
                            kernel_initializer=initializer.kernel_initializer,
                            bias_initializer=initializer.bias_initializer,
                            padding=padding,
                            name=layer_names.popleft())(premuted_s)
            relu1 = self.Ex.Rectifier(name=layer_names.popleft())(Conv1)
            Conv2 = Conv2D(64, (4,4), strides=(2, 2),
                            kernel_initializer=initializer.kernel_initializer,
                            bias_initializer=initializer.bias_initializer,
                            padding=padding,
                            name=layer_names.popleft())(relu1)
            relu2 = self.Ex.Rectifier(name=layer_names.popleft())(Conv2)
            Conv3 = Conv2D(64, (3,3), strides=(1, 1),
                            kernel_initializer=initializer.kernel_initializer,
                            bias_initializer=initializer.bias_initializer,
                            padding=padding,
                            name=layer_names.popleft())(relu2)
            relu3 = self.Ex.Rectifier(name=layer_names.popleft())(Conv3)
            flatten=Flatten(name=layer_names.popleft())(relu3)
            Linear = Dense(512,
                            kernel_initializer=initializer.kernel_initializer,
                            bias_initializer=initializer.bias_initializer,
                            name=layer_names.popleft())(flatten)
            relu4 = self.Ex.Rectifier(name=layer_names.popleft())(Linear)
            output      = Dense(self.args.n_actions,
                            kernel_initializer=initializer.kernel_initializer,
                            bias_initializer=initializer.bias_initializer,
                            name=layer_names.popleft())(relu4)

            assert len(layer_names) == 0

            model = Model(inputs=s, outputs=output, name=self.network_name)

            if self.args.verbose >= 2:
                from keras.utils import plot_model
                model.summary()
                if self.args.verbose >= 3:
                    plot_model(model, to_file='{}.png'.format(model.name), show_shapes=True)

            return model

    def forward(self, s):
        return self.model.predict_on_batch(s)

    def _save(self, filepath):
        self.model.save_weights(filepath)

    def _load(self, filepath):
        self.model.load_weights(filepath)

    def set_summarizable_parameters(self, params):
        self.params = params

    def get_trainable_parameters(self, numpy=True):
        if numpy:
            return self.model.get_weights()
        else:
            return self.trainable_weights()

    def set_trainable_parameters(self, weights, numpy=True):
        if numpy:
            if self.Ex.get_device(self.args.gpu):
                self.model.set_weights(weights)
        else:
            raise NotImplementedError()

    @property
    def input(self):
        return self.model.inputs[0]

    @property
    def output(self):
        return self.model.outputs[0]

    @property
    def trainable_weights(self):
        return self.model.trainable_weights

    def get_grads(self, numpy=False):
        return self.grad_values if numpy else self.grads

    def set_grads(self, grads):
        self.grads = grads

    def set_grad_values(self, grad_values):
        grad_values = np.array(grad_values).reshape(-1, 2)
        self.grad_values = grad_values

    def get_outputs(self):
        outputs = []
        for name in self.summarizable_layer_names:
            layer = self.model.get_layer(name)
            outputs.append(layer.output)
        return outputs

    def get_output_values(self):
        return self.output_values

    def set_output_values(self, output_values):
        pass
        #self.output_values = np.array(output_values).reshape(9,-1)

class KerasTrainer(Trainer):
    def __init__(self, args, network, target_network):
        Trainer.__init__(self, args, network, target_network)

        self.func_getQUpdate = self._create_getQUpdate_ops()
        self.func_qLearnMinibatch, \
        self.func_qLearnMinibatch_for_summary = self._create_qLeanMinibatch_ops()
        self.func_update_target_network = self._create_update_target_network_ops()


    def _compute_q2(self):
        from keras import backend as K

        r = K.placeholder((None,), dtype='float32', name='r')
        term = K.placeholder((None,), dtype='float32', name='term')
        gamma = K.constant(self.args.discount, dtype='float32', name='gamma')

        q2_all = self.target_network.output

        with K.name_scope('compute_q2_max'):
            q2_max = K.max(q2_all, axis=1)

        with K.name_scope('preprocess_term'):
            _term = term * -1.0 + 1.0
        
        with K.name_scope('compute_q2'):
            q2 = q2_max * gamma * _term + r

        self.getQUpdate_ops = [q2_all, q2_max, q2, r]

        return q2, q2_max, r, term

    def _compute_q(self):
        from keras import backend as K

        q_all = self.network.output
        a = K.placeholder((None,), dtype='int32', name='a')

        with K.name_scope('a_to_one_hot'):
            a_one_hot = K.one_hot(a, self.args.n_actions)
        
        with K.name_scope('compute_q'):
            q_values_wrt_a = q_all * a_one_hot
            q = K.sum(q_values_wrt_a, axis=1)

        self.getQUpdate_ops += [q_all, q]

        return q_all, q, a, a_one_hot

    def _create_getQUpdate_ops(self):
        from keras import backend as K
        Ex = get_extensions(self.args)

        q2, q2_max, r, term = self._compute_q2()
        q_all, q, a, a_one_hot = self._compute_q()

        with K.name_scope('get_batch_size'):
            batch_size = K.shape(q2)[0]

        with K.name_scope('compute_delta'):
            q2 = K.stop_gradient(q2)
            delta = q2 - q
            delta = K.clip(delta, -1.0, 1.0)

        if self.args.loss_function in ['DQN3.0', 'BFT1', 'BFT2']:
            print('choice loss_function', 'DQN3.0', 'BFT1', 'BFT2')
            with K.name_scope('compute_targets'):
                _targets = a_one_hot * K.reshape(delta, [batch_size, 1])
                if self.args.loss_function == 'DQN3.0':
                    _targets = K.stop_gradient(_targets)

                # The 'Ex.start_gradient' function will be able to specify entry point
                # of the gradient for avoiding gradients of DQN loss function in CNTK.
                self.targets = Ex.start_gradient(q_all, _targets)

        elif self.args.loss_function == 'huber':
            print('choice loss_function huber')
            q2 = K.stop_gradient(q2)
            self.targets = Ex.huber_loss(q, q2)

        self.getQUpdate_ops += [delta, self.targets]

        self.inputs = (
            self.network.input,
            a,
            r,
            self.target_network.input,
            term
        )

        outputs = [self.targets,
                   delta,
                   q2_max]
        return  K.function(self.inputs, outputs)



    def _create_qLeanMinibatch_ops(self):
        assert hasattr(self, 'targets'), 'This method must runs after '
        'self._create_getQUpdate_ops() for computing targets.'

        from keras import backend as K
        Ex = get_extensions(self.args)

        _grads = []

        def _DQN_RMSprop(vars, grads):

            nonlocal _grads

            with K.name_scope('DQNRMSprop'):


                with K.name_scope('constants'):
                    grad_momentum = K.constant(self.args.grad_momentum, dtype='float32', name='grad_momentum')
                    sqared_grad_momentum = K.constant(self.args.sqared_grad_momentum, dtype='float32', name='sqared_grad_momentum')
                    mini_squared_gradient = K.constant(self.args.mini_squared_gradient, dtype='float32', name='mini_squared_gradient')
                    lr = K.constant(self.args.lr, dtype='float32', name='lr')
                    momentum = K.constant(self.args.momentum, dtype='float32', name='momentum')
                    wc = K.constant(self.args.wc, dtype='float32', name='wc')

                g = []
                g2 = []
                deltas = []
                with K.name_scope('states'):
                    for i, grad in enumerate(grads):
                        g.append(K.variable(K.zeros_like(grad), name='g_' + str(i)))
                        g2.append(K.variable(K.zeros_like(grad), name='g2_' + str(i)))
                        deltas.append(K.variable(K.zeros_like(grad), name='deltas_' + str(i)))
                        _grads.append(grad)

                update_g = []
                update_g2 = []
                update_deltas = []
                update_var = []
                tmp = []
                for grad, var, _g, _g2, _deltas in zip(grads, vars, g, g2, deltas):

                    with K.name_scope('apply_wc'):
                        grad = grad - (wc * var)

                    with K.name_scope('compute_g'):
                        new_g = _g * grad_momentum + (1.0-grad_momentum) * grad
                    with K.name_scope('compute_g2'):
                        new_g2 = _g2 * sqared_grad_momentum + (1.0-sqared_grad_momentum) * K.square(grad)
                    with K.name_scope('compute_tmp'):
                        new_tmp = Ex.sqrt(new_g2 - K.square(new_g) + mini_squared_gradient)
                    with K.name_scope('compute_deltas'):
                        new_deltas = _deltas * momentum + lr * (grad / new_tmp)

                    with K.name_scope('update_var'):
                        update_var.append(K.update_add(var, new_deltas))

                    update_g.append(K.update(_g, new_g))
                    update_g2.append(K.update(_g2, new_g2))
                    tmp.append(new_tmp)
                    update_deltas.append(K.update(_deltas, new_deltas))

            self.RMSprop_ops = [g, g2, tmp, deltas]

            return Ex.group(update_var + update_g + update_g2 + update_deltas)

        rmsprop = _DQN_RMSprop if self.args.optimizer == 'DQN3.0' else None

        vars = self.network.trainable_weights
        tvars = self.target_network.trainable_weights
        func, grads = Ex.create_qLeanMinibatch_func(self.args,
                                                    self.network.output,
                                                    self.inputs,
                                                    self.targets,
                                                    vars,
                                                    rmsprop)
        if grads is None:
            grads = _grads

        # For summaries
        target_network_vars = self.target_network.trainable_weights
        func_for_summary = Ex.get_for_summary_func(self.inputs,
                                                   self.network.outputs,
                                                   vars,
                                                   grads,
                                                   self.target_network.outputs,
                                                   target_network_vars,
                                                   self.getQUpdate_ops,
                                                   self.RMSprop_ops if hasattr(self, 'RMSprop_ops') else None)
        return func, func_for_summary

    def _create_update_target_network_ops(self):
        from keras import backend as K

        if self.args.backend == 'cntk':
            # NOTE: CNTK will not be updates variables without loss.
            return None
        
        with K.name_scope('update_target_network'):
            self.update_target_network_ops = [
                K.update(tv, nv) for tv, nv in zip(self.target_network.trainable_weights,
                                                   self.network.trainable_weights)
            ]

            return K.function([], [], self.update_target_network_ops)

    def _getQUpdate(self, x):
        # This method is for validation.
        return self.func_getQUpdate(x)


    def _qLearnMinibatch(self, x, do_summary):
        # NOTE: This method will not call _getQUpdate() like DQN3.0.
        # Because graph of qLearnMinibatch ops are includes getQUpdate ops.

        self.s, _, _, self.s2, _ = x
        self.func_qLearnMinibatch(x)
        if do_summary:
            self.outputs = self.func_qLearnMinibatch_for_summary(x)

    def _update_target_network(self):

        if self.args.backend == 'cntk':
            params = self.network.model.get_weights()
            self.target_network.model.set_weights(params)
        else:
            self.func_update_target_network([])

    def add_learning_summaries(self, numSteps):

        network_outputs,\
        network_vars,\
        network_grads,\
        target_network_outputs,\
        target_network_vars,\
        getQUpdate_vals,\
        getRMSprop_vals = self.outputs
        
        # concatenate variables in g, g2, tmp and deltas.
        getRMSprop_vals = [tuple([var.flatten() for var in vars]) for vars in getRMSprop_vals]
        getRMSprop_vals = [np.concatenate(vars) for vars in getRMSprop_vals]

        dict_getQUpdate_values = dict(zip(LearningVisualizer.GET_Q_UPDATE_VALUE_NAMES,
                                          getQUpdate_vals))
        dict_rms_prop_values = dict(zip(LearningVisualizer.RMS_PROP_VALUE_NAMES[:-1],
                                        getRMSprop_vals))
        dict_rms_prop_values[LearningVisualizer.RMS_PROP_VALUE_NAMES[-1]] = self.args.lr


        dict_network_parameters = self._get_params(network_outputs, network_vars, network_grads)
        dict_target_network_parameters = self._get_params(network_outputs, network_vars, [None]*len(network_grads))

        self.learning_visualizer.addInputImages(2, self.s[0])
        self.learning_visualizer.addInputImages(3, self.s2[0])
        self.learning_visualizer.addGetQUpdateValues(dict_getQUpdate_values)
        self.learning_visualizer.addRMSpropValues(dict_rms_prop_values)
        self.learning_visualizer.addNetworkParameters(dict_network_parameters)
        self.learning_visualizer.addTargetNetworkParameters(dict_target_network_parameters)

        self.learning_visualizer.flush(numSteps)

    def _get_params(self, outputs, vars, grads):
        params = {}
        outputs_deq = deque(outputs)
        vars_deq = deque(vars)
        grads_deq = deque(grads) 
        for layer_name in self.network.summarizable_layer_names:
            if re.match(r"^relu\d$", layer_name):
                params[layer_name] = (
                    outputs_deq.popleft(),
                    None,
                    None,
                    None,
                    None,
                )
            else:
                params[layer_name] = (
                    outputs_deq.popleft(),
                    vars_deq.popleft(),
                    vars_deq.popleft(),
                    grads_deq.popleft(),
                    grads_deq.popleft()
                )
        return params


    @property
    def _getQUpdate_values(self):
        return self.getQUpdate_values

    @property
    def _rmsprop_values(self):
        return self.RMSprop_values

def create_networks(args):
    from common import get_extensions
    Ex = get_extensions(args)

    with Ex.get_device(args.gpu):

        network        = KerasConvnet(args, 'network')
        target_network = KerasConvnet(args, 'target_network')
        trainer        = KerasTrainer(args, network, target_network)

    return network, trainer
