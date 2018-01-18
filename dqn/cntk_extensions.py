import numpy as np
from contextlib import contextmanager
import cntk as C
from cntk.ops.functions import UserFunction
from cntk.contrib.deeprl.agent.shared.cntk_utils import huber_loss

def setup_before_package_loading(opt):
    import os
    os.environ['KERAS_BACKEND'] = 'cntk'
    
    # After setting the environment variables,
    # Create a class instance of Rectifier.
    _build_Rectifier_class(opt)

def setup(opt):
    return opt

@contextmanager
def get_device(gpu=0):

    # CNTK can arbitrarily select a GPU with the CNTK process
    # executed the first time after OS startup.
    # After that CNTK process inherits the GPU setting
    # of the first CNTK process.
    try:
        device = C.device.gpu(gpu) if gpu >= 0 else C.device.cpu()
        C.device.try_set_default_device(device)
    except Exception:
        pass

    yield
    pass

def variable_initialization():
    pass

def sqrt(x):
    return C.sqrt(x)

class StartGradient(UserFunction):

    def __init__(self, q_all, targets, name='start_gradient'):
        super(StartGradient, self).__init__([q_all, targets], as_numpy=False, name=name)

    def forward(self, arguments, device=None, outputs_to_retain=None):
        return arguments[1], arguments[1]

    def backward(self, state, root_gradients, variables):
        variables[self.inputs[0]] = state

    def infer_outputs(self):
        return [C.output_variable(self.inputs[1].shape, self.inputs[1].dtype, self.inputs[1].dynamic_axes)]

    @staticmethod
    def deserialize(inputs, name, state):
        return StartGradient(inputs[0], inputs[1], name)

def start_gradient(q_all, targets):
    return C.user_function(StartGradient(q_all, targets))

def huber_loss(output, target):
    return huber_loss(output, target)



Rectifier = None
def _build_Rectifier_class(opt):
    from keras.layers import Activation

    # Reproduce DQN3.0 Rectifier layer
    class RectifierFunction(UserFunction):

        def __init__(self,input, name='Rectifier_function'):
            super(RectifierFunction, self).__init__([input], as_numpy=False, name=name)

            # In UserFunction of CNTK, Must define the mini compute graph
            # for implementation of forward() and backward() when you use as_numy=False.
            # This manners can be seen in test class of UserFunction
            # that https://github.com/Microsoft/CNTK/blob/01eb0fb1aef3834557c296e95be09f62c4f2c521/bindings/python/cntk/ops/tests/userfunction_test.py#L651.

            # Define forward function.
            self.forward_arg = C.input_variable(input.shape, input.dtype)
            if opt.rectifier_div:
                self.forward_func = (C.abs(self.forward_arg) + self.forward_arg) / 2.0
            else:
                self.forward_func = (C.abs(self.forward_arg) + self.forward_arg) * 0.5

            # Define backwrod function.
            backward_arg1 = C.input_variable(input.shape, input.dtype)
            backward_arg2 = C.input_variable(input.shape, input.dtype)

            # compute sign manually.
            greater = C.greater(backward_arg1, 0.0)
            tmp_val = C.element_select(greater, 1.0, backward_arg1)
            less = C.less(tmp_val, 0.0)
            signed_val = C.element_select(less, -1.0, tmp_val)

            self.backward_func = signed_val * backward_arg2

        def forward(self, argument, device=None, outputs_to_retain=None):
            output = self.forward_func.eval({self.forward_func.arguments[0]:argument}, as_numpy=False)
            return output, output

        def backward(self, state, root_gradients):
            feed_dict = dict(zip(self.backward_func.arguments, [state, root_gradients]))
            grad_input = self.backward_func.eval(feed_dict, as_numpy=False)
            return grad_input

        def infer_outputs(self):
            return [C.output_variable(self.inputs[0].shape, self.inputs[0].dtype,
                self.inputs[0].dynamic_axes)]

        @staticmethod
        def deserialize(inputs, name, state):
            return RectifierFunction(inputs[0], name)

    global Rectifier
    class _Rectifier(Activation):
        def __init__(self, **kwargs):
            super(Rectifier, self).__init__(activation='relu', **kwargs)

        def call(self, inputs):
            if opt.relu:
                return super(Rectifier, self).call(inputs)
            else:
                return C.user_function(RectifierFunction(inputs))

    Rectifier = _Rectifier

def group(updates):
    return C.combine(updates)

def _convert_arguments(x):
    s, a, r, s2, term = x
    
    # eval() needs type casting from uint8 to float32.
    a = a.astype(np.float32)
    r = r.astype(np.float32)
    term = term.astype(np.float32)

    return [s, a, r, s2, term]

def create_qLeanMinibatch_func(args, q_all, inputs, targets, vars, rmsprop):

    if args.optimizer == 'DQN3.0':
        assert rmsprop is not None
        print('choise learner DQN3.0')
        learner = C.universal(rmsprop, vars)
    else:
        learner = _get_learner(args, vars)

    # NOTE:
    # The trainer showed a note "10 of the model parameters are not covered by any of the specified Learners".
    # However, the "10 of the model parameters" are target_network's parameters.
    # Because of this, this note can be ignored.
    trainer = C.Trainer(targets, (targets), [learner])

    def function(x):
        arguments = dict(zip(inputs,_convert_arguments(x)))
        trainer.train_minibatch(arguments)
    return function, None

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
                 [[v for k, v in C.combine(ops).eval().items()] for ops in getRMSprop_ops],
                )
        return ret

    return function

def _get_learner(args, vars):

    if args.optimizer == 'RMSpropNonCentered':
        print('choise learner RMSpropNonCentered')
        return C.learners.rmsprop(vars,
                                  args.lr,
                                  args.grad_momentum)
    else:
        raise NotImplementedError()
