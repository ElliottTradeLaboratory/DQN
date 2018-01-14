import logging
import numpy as np

from mxnet.ndarray import NDArray, zeros
import mxnet.initializer
import mxnet.operator
import mxnet.optimizer
from mxnet import random
from mxnet.module import BucketingModule
from mxnet import context as ctx
from mxnet.symbol import sign, sqrt, Activation, Custom
from initializer import Torch_nn_DefaultInitializer as torch_nn_init

def setup_before_package_loading(opt):
    pass

def setup(opt):
    random.seed(opt.seed)
    return opt

# Reproduce initializer of Torch7 nn layer.
@mxnet.initializer.register
@mxnet.initializer.alias('torch_nn_init')
class Torch_nn_DefaultInitialiser(mxnet.initializer.Initializer):

    def __init__(self):
        super(Torch_nn_DefaultInitialiser, self).__init__()
        torch_nn_init.channel_last = False

    def _init_weight(self, name, arr):
        shape = arr.shape
        weight = torch_nn_init.kernel_initializer(shape)
        arr[:] = weight

    def _init_bias(self, name, arr):
        shape = arr.shape
        bias = torch_nn_init.bias_initializer(shape)
        arr[:] = bias

# Extension class of BucketingModule with get_parameters() added
class ExBucketingModule(BucketingModule):
    def get_parameters(self):
        return (self._curr_module._exec_group.execs[0].arg_dict,
                self._curr_module._exec_group.execs[0].grad_dict)

# Reproduce Rectifier Layer of DQN3.0.
class Rectifier(mxnet.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        output = (in_data[0].abs() + in_data[0]) / 2.0
        self.assign(out_data[0], req[0], output)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_data[0].sign() * out_grad[0])

@mxnet.operator.register("Rectifier")
class RectifierProp(mxnet.operator.CustomOpProp):
    def __init__(self):
        super(RectifierProp, self).__init__(need_top_grad=True)
    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [in_shape[0]], [in_shape[0]], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return Rectifier()


def get_relu(opt, inputs, name):
    if opt.relu:
        return Activation(data=inputs, act_type='relu', name=name)
    else:
        return Custom(data=inputs, op_type='Rectifier', name=name)

@mxnet.optimizer.register
class DQNRMSProp(mxnet.optimizer.Optimizer):
    def __init__(self,
                 learning_rate,
                 grad_momentum,
                 sqared_grad_momentum,
                 mini_squared_gradient,
                 momentum,
                 wc,
                 **kwargs):
        super(DQNRMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.grad_momentum = grad_momentum
        self.sqared_grad_momentum = sqared_grad_momentum
        self.mini_squared_gradient = mini_squared_gradient
        self.momentum = momentum
        self.wc = wc
        self.state = []

    def create_state(self, index, weight):
        self.state += [
            (zeros(weight.shape, weight.context, stype=weight.stype),  # g
             zeros(weight.shape, weight.context, stype=weight.stype),  # g2
             zeros(weight.shape, weight.context, stype=weight.stype))] # deltas
        return self.state[-1]
    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)

        g, g2, deltas = state

        grad[:] = grad - (self.wc * weight)

        g[:] = g * self.grad_momentum + (1-self.grad_momentum) * grad
        g2[:] = g2 * self.sqared_grad_momentum + (1-self.sqared_grad_momentum) * grad.square()
        tmp = (g2 - g.square() + self.mini_squared_gradient).sqrt()
        deltas[:] = deltas * self.momentum + lr * (grad / tmp)
        weight[:] = weight + deltas
