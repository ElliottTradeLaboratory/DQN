import re
import numpy as np

import mxnet as mx
from collections import namedtuple, deque, OrderedDict
from mxnet.symbol import Variable, FullyConnected, Custom, Convolution, flatten
from mxnet_extensions import Torch_nn_DefaultInitialiser, ExBucketingModule, DQNRMSProp

from convnet_common import Convnet, Trainer
from visualizer import LearningVisualizer

MODE_LEARNING = '1'
MODE_PREDICT  = '2'
MODE_VALIDATE = '3'
MODE_OUTPUTS  = '4'

class MXnetConvnet(Convnet):
    def __init__(self, args,
                 network_name):
        super(MXnetConvnet, self).__init__(args, network_name)

        self.Batch = namedtuple('Batch', ['data', 'provide_data', 'bucket_key', 'provide_label'])
        self.module = self._create_module()
        self._output_dict = None

    def _create_module(self):

        names = deque(self.layer_names)

        # layer creation.
        self.layers = []
        s = Variable('s')
        self.layers.append(Convolution(data=s,
                                       num_filter=32,
                                       kernel=(8,8),
                                       stride=(4,4),
                                       pad=(1,1),
                                       name=names.popleft()))
        self.layers.append(Custom(data=self.layers[-1], op_type='Rectifier', name=names.popleft()))
        self.layers.append(Convolution(data=self.layers[-1],
                                       num_filter=64,
                                       kernel=(4,4),
                                       stride=(2,2),
                                       pad=(0,0),
                                       name=names.popleft()))
        self.layers.append(Custom(data=self.layers[-1], op_type='Rectifier', name=names.popleft()))
        self.layers.append(Convolution(data=self.layers[-1],
                                       num_filter=64,
                                       kernel=(3,3),
                                       stride=(1,1),
                                       pad=(0,0),
                                       name=names.popleft()))
        self.layers.append(Custom(data=self.layers[-1], op_type='Rectifier', name=names.popleft()))
        self.layers.append(flatten(data=self.layers[-1], name=names.popleft()))
        self.layers.append(FullyConnected(data=self.layers[-1],
                                          num_hidden=512,
                                          name=names.popleft()))
        self.layers.append(Custom(data=self.layers[-1], op_type='Rectifier', name=names.popleft()))
        self.layers.append(FullyConnected(data=self.layers[-1],
                                          num_hidden=self.args.n_actions,
                                          name=names.popleft()))


        self.shapes = {
            MODE_LEARNING:[('s', (self.args.minibatch_size,) + self.args.input_dims)],
            MODE_OUTPUTS :[('s', (self.args.minibatch_size,) + self.args.input_dims)],
            MODE_PREDICT :[('s', (1,) + self.args.input_dims)],
            MODE_VALIDATE:[('s', (self.args.valid_size,) + self.args.input_dims)],
        }

        def sym_gen(bucket_key):
            # Return the parameters for creating Module at BucketingModule.
            return (mx.symbol.Group(self.layers[:-1]), ('s',), None) if bucket_key == MODE_OUTPUTS \
                   else (self.layers[-1], ('s',), None)
        
        # Initialize a Model for validate because it is most large batch_size.
        # mxnet recommends default_bucket use most large batch size for memory allocation.
        module = ExBucketingModule(sym_gen, default_bucket_key=MODE_VALIDATE, context=self.args.device)
        module.bind(data_shapes=self.shapes[MODE_VALIDATE], for_training=True)
        module.init_params(Torch_nn_DefaultInitialiser())

        # Add other Modules
        for mode in [MODE_PREDICT, MODE_OUTPUTS, MODE_LEARNING]:
            module.switch_bucket(mode, self.shapes[mode])

        if self.args.verbose >= 2:
            output_shapes = dict(self.shapes[MODE_LEARNING])
            for name, v in module.get_params()[0].items():
                output_shapes[name] = v.shape
            print('--------{}--------'.format(self.network_name))
            mx.visualization.print_summary(self.layers[-1], output_shapes)

        return module

    def forward(self, s, mode=MODE_PREDICT):

        if mode == MODE_LEARNING:
            # If mode == MODE_LEARNING, execute MODE_OUTPUTS before execute MODE_LEARNING.
            # Because, if execute it after execute MODE_LEARNING, current_module in BucketingModule is
            # MODE_OUTPUTS, because of this, occurs error at backword, because shape of MODE_OUTPUTS is
            # not match to shape of targets.
            self.module.forward(self.Batch([mx.ndarray.array(s)], self.shapes[MODE_OUTPUTS], MODE_OUTPUTS, None))
            self._output_dict = OrderedDict(zip(self.layer_names[:-1],
                                            self.module.get_outputs()))
        self.module.forward(self.Batch([mx.ndarray.array(s)], self.shapes[mode], mode, None))
        outputs = self.module.get_outputs()

        if mode == MODE_PREDICT:
            return outputs[0].asnumpy()
        elif mode == MODE_LEARNING:
            self._output_dict['q_all'] = outputs[0]
            return outputs[0]
        elif mode == MODE_VALIDATE:
            return outputs[0]
        else:
            assert False, "mode'{}' is not supported. MODE_OUTPUTS is included in MODE_LEARNING".format(mode)

    def backward(self, outgrad):
        self.module.backward([outgrad])

    def update(self):
        self.module.update()

    def _get_params_dict(self, numpy):
        arg_dict, grad_dict = self.module.get_parameters()
        params_dict = OrderedDict()
        for layer_name in self.layer_names:
            params = (self._output_dict[layer_name] if self._output_dict is not None else None,
                      arg_dict.get(layer_name + '_weight', None),
                      arg_dict.get(layer_name + '_bias', None),
                      grad_dict.get(layer_name + '_weight', None),
                      grad_dict.get(layer_name + '_bias', None))
            if numpy:
                params = tuple([p if p is None else p.asnumpy() for p in params])

            params_dict[layer_name] = params

        return params_dict

    def _save(self, filepath):
        self.module.save_params(filepath)

    def _load(self, filepath):
        self.module.load_params(filepath)


class MXnetTrainer(Trainer):

    def __init__(self, args, network, target_network):
        super(MXnetTrainer, self).__init__(args, network, target_network)

        optimizer = DQNRMSProp(learning_rate=args.lr,
                               grad_momentum=args.grad_momentum,
                               sqared_grad_momentum=args.sqared_grad_momentum,
                               mini_squared_gradient=args.mini_squared_gradient,
                               momentum=args.momentum,
                               wc=args.wc)
        import warnings
        with warnings.catch_warnings():
            # The Module outputs warning about how to use rescale_grad
            # because I specified rescale_grad with default that 1.
            # mxnet recommends use rescale_grad with 1/batch_size.
            # However, I never use it because DQN3.0 does not use it too.
            # That's none of your business!
            warnings.simplefilter("ignore")
            self.network.module.init_optimizer(optimizer=optimizer)

    def compute_validation_statistics(self, x):

        _, delta, q2_max = self._getQUpdate(x, mode=MODE_VALIDATE)

        return delta.asnumpy(), q2_max.asnumpy()

    @property
    def _getQUpdate_values(self):
        return [self.q2_all.asnumpy(),
                self.q2_max.asnumpy(),
                self.q2.asnumpy(),
                self.r.asnumpy(),
                self.q_all.asnumpy(),
                self.q.asnumpy(),
                self.delta.asnumpy(),
                self.targets.asnumpy()]

    @property
    def _rmsprop_values(self):
        g = np.array([])
        g2 = np.array([])
        deltas = np.array([])
        for k, v in self.network.module._curr_module._updater.states.items():
            g = np.hstack((g, v[0].asnumpy().flatten()))
            g2 = np.hstack((g2, v[1].asnumpy().flatten()))
            deltas = np.hstack((deltas, v[2].asnumpy().flatten()))
        return [g, g2, None, deltas]

    def _getQUpdate(self, x, mode):
        s, a, r, s2, term = [mx.ndarray.array(v, self.args.device) for v in x]
        
        self.r = r

        # delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
        self.term = term * -1 + 1

        # Compute max_a Q(s_2, a).
        self.q2_all = self.target_network.forward(s2, mode)

        self.q2_max =self.q2_all.max(1)

        # Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
        q2 = self.q2_max * self.args.discount * self.term

        self.delta = r + q2
        
        self.q2 = self.delta

        # q = Q(s,a)
        self.q_all = self.network.forward(s, mode)

        batch_size = self.q_all.shape[0]
        a_one_hot = mx.ndarray.one_hot(a, self.args.n_actions)
        q_action = self.q_all * a_one_hot
        self.q = q_action.sum(1)

        self.delta = self.delta - self.q

        self.delta = self.delta.clip(-self.args.clip_delta,
                                     self.args.clip_delta)
        
        self.targets = self.delta.reshape((q2.shape[0],1)) * a_one_hot

        return self.targets, self.delta, self.q2_max


    def _qLearnMinibatch(self, x, do_summary):

        self.s, _, _, self.s2, _ = x

        targets , _, _ = self._getQUpdate(x, MODE_LEARNING)

        self.network.backward(targets)

        self.network.update()


    def _add_learning_summaries(self, numSteps):
        raise NotImplementedError()

    def _update_target_network(self):
        params = self.network.module.get_params()
        self.target_network.module.set_params(*params)


def create_networks(args):
    args.device = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    from utils import get_random
    get_random().manualSeed(1)
    network = MXnetConvnet(args, 'network')
    target_network = MXnetConvnet(args, 'target_network')
    trainer = MXnetTrainer(args, network, target_network)

    return network, trainer
