import numpy as np
from contextlib import contextmanager
import theano
import theano.tensor as T

def setup_before_package_loading(opt):
    device = 'cuda{}'.format(opt.gpu) if opt.gpu >= 0 else 'cpu'

    import os
    os.environ['KERAS_BACKEND'] = 'theano'
    os.environ['THEANO_FLAGS'] = "floatX=float32,device={}".format(device)

    # After setting the environment variables,
    # Create a class instance of Rectifier.
    _build_Rectifier_class()

def setup(opt):
    # Nop
    return opt

@contextmanager
def get_device(gpu=0):
    # Nop
    pass
    yield
    pass


# Reproduce DQN3.0 Rectifier layer
class RectifierOp(theano.Op):
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        self.output = (x.abs() + x) * 0.5
        z[0] = self.output

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [self.output.sgn() * output_grads[0]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


Rectifier = None
def _build_Rectifier_class():
    from keras.layers import Activation

    global Rectifier
    class _Rectifier(Activation):
        def __init__(self, **kwargs):
            super(Rectifier, self).__init__(activation='relu', **kwargs)

        def call(self, inputs):
            rectifier_op = RectifierOp()
            return rectifier_op(input)

    Rectifier = _Rectifier

def group(updates):
    # Nop
    return updates

def _convert_arguments(x):
    s, a, r, s2, term = x
    
    # eval() needs type casting from uint8 to float32.
    a = a.astype(np.float32)
    r = r.astype(np.float32)
    term = term.astype(np.float32)

    return [s, a, r, s2, term]

_known_grads = None
def start_gradient(q_all, targets):
    # Create gradient argument
    global known_grads
    known_grads = {q_all: targets}
    return targets

def create_qLeanMinibatch_func(args, inputs, targets, vars, rmsprop):
    import theano.tensor as T

    grads = T.grad(None, wrt=params, known_grads=_known_grads)
    updates = _rmsprop_for_theano(grads, vars, args)
    trainer = theano.function([], [], updates=updates)
    def function(x):
        arguments = dict(zip(inputs, x))
        trainer(arguments)
    return function

def get_for_summary_func(inputs,
                         network_outputs,
                         network_vars,
                         network_grads,
                         target_network_outputs,
                         target_network_vars,
                         getQUpdate_ops,
                         getRMSprop_ops):

    def function(x):

        s, a, r, s2, term = _convert_arguments(x)

        input_s, _, _, input_s2, _ = inputs

        args = dict(zip(inputs, [s, a, r, s2, term]))

        ret = (  [v for k, v in C.combine(network_outputs).eval({input_s:s}).items()],
                 [v for k, v in C.combine(network_vars).eval().items()],
                 [v for k, v in C.combine(network_grads).eval().items()],
                 [v for k, v in C.combine(target_network_outputs).eval({input_s2:s2}).items()],
                 [v for k, v in C.combine(target_network_vars).eval().items()],
                 [v for k, v in C.combine(getQUpdate_ops).eval(args).items()],
                 [v for k, v in C.combine(getRMSprop_ops).eval().items()],
                )
        return ret

    return function

def _rmsprop_for_theano(grads, vars, args):

    updates = OrderedDict()

    for var, grad in zip(vars, grads):
        value = param.get_value(borrow=True)

        g = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        g2 = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        deitas = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)

        new_g = args.grad_momentum * g + (1 - args.grad_momentum) * grad
        new_g2 = args.squared_momentum * g2 + (1 - args.squared_momentum) * T.square(grad)
        tmp = T.sqrt(new_g2 - T.square(new_g) + self.mini_squared_gradient))
        new_deltas = (deltas * args.momentum) + (args.lr * (grad / tmp))
        
        updates[g] = new_g
        updates[g2] = new_g2
        updates[deltas] = new_deltas
        updates[var] = var + new_deltas

    return updates
