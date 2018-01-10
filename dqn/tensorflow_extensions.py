import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python import debug as tf_debug

K = None

def setup_before_package_loading(opt):
    import os
    os.environ['KERAS_BACKEND'] = opt.backend

    from keras import backend
    global K

    K = backend

    _create_RectifierClass(opt)

def setup(opt):
    sess = get_session(opt.log_device_placement,
                       debug=opt.tf_debug)
    K.set_session(sess)
    tf.set_random_seed(opt.seed)
    K.manual_variable_initialization(True)

    return opt

def variable_initialization():
    K.get_session().run(tf.global_variables_initializer())


def _get_config(log_device_placement=False):
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = True
    return config

_SESS = None
def get_session(log_device_placement=False, graph=None, new_session=False, debug=False):
    global _SESS

    if new_session:
        config = _get_config(log_device_placement)
        sess = tf.Session(config=config, graph=graph)

    else:
        if _SESS is None:
            config = _get_config(log_device_placement)
            _SESS = tf.Session(config=config)

        sess = _SESS

    if debug:
        print('*********** TFDBG MODE *************')
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    return sess

def get_device(gpu=0):
    return tf.device('/cpu:0' if gpu == -1 else '/gpu:{}'.format(gpu if gpu is not None else 0))

def sqrt(x):
    return tf.sqrt(x)

# Define of polymorphismic user function 'start_gradient' for CNTK ver.
def start_gradient(q_all, targets):
    return targets

def huber_loss(labels, predictions):
    return tf.losses.huber_loss(labels, predictions)

Rectifier = None
def _compute_output(opt, input):
    if opt.rectifier_div:
        output = (tf.abs(input) + input) / 2.0
    else:
        output = (tf.abs(input) + input) * 0.5
    return output
def _create_RectifierClass(opt):
    from keras.layers import Layer

    @function.Defun()
    def rectifier_func_grad(input, grad):
        return _compute_output(opt, input) * grad

    def shape_func(op):
        return [op.inputs[0].get_shape()]

    @function.Defun(shape_func=shape_func, grad_func=rectifier_func_grad)
    def rectifier_func(input):
        return _compute_output(opt, input)

    class _Rectifier(Layer):
        def __init__(self, **kwargs):
            super(_Rectifier, self).__init__(**kwargs)
            self.func = tf.nn.relu if opt.relu else rectifier_func 

        def build(self, input_shape):
            super(_Rectifier, self).build(input_shape)

        def call(self, inputs):
            return self.func(inputs)

        def compute_output_shape(self, input_shape):
            return (input_shape)

    global Rectifier
    Rectifier = _Rectifier


def group(ops):
    return tf.group(*ops)

def create_qLeanMinibatch_func(args, q_all, inputs, targets, vars, rmsprop):

    if args.optimizer == 'DQN3.0':
        if args.loss_function == 'DQN3.0':
            print('choice gradient DQN3.0')
            grads = tf.gradients(q_all, vars, targets)
        else:
            if args.loss_function == 'BFT1':
                print('choice gradient BFT1')
                grads = tf.gradients(targets, vars)
            elif args.loss_function == 'BFT2':
                print('choice gradient BFT2')
                grads = tf.gradients(targets, vars, targets)

            grads = [g for g in grads if g is not None]

        training_updates = rmsprop(vars, grads)

    else:
        print('choice gradient other')
        optimizer = _get_optimizer(args, vars)
        grads_and_vars = optimizer.compute_gradients(targets, var_list=vars)
        training_updates = optimizer.apply_gradients(grads_and_vars)

        grads = []
        for g, v in grads_and_vars:
            grads.append(g)

    return K.function(inputs, [targets], updates=[training_updates]), grads

def get_for_summary_func(inputs,
                         network_outputs,
                         network_vars,
                         network_grads,
                         target_network_outputs,
                         target_network_vars,
                         getQUpdate_ops,
                         getRMSprop_ops):

    # same as get_for_summary_func() in cntk_extensions.py.
    def function(x):
        nonlocal getRMSprop_ops
        sess = K.get_session()
        if getRMSprop_ops is not None:
            rmsprop_ops = tuple([sess.run(ops, feed_dict={inputs: x}) for ops in getRMSprop_ops])
        else:
            rmsprop_ops = None
        return ((sess.run(network_outputs, feed_dict={inputs: x})),
                (sess.run(network_vars, feed_dict={inputs: x})),
                (sess.run(network_grads, feed_dict={inputs: x})),
                (sess.run(target_network_outputs, feed_dict={inputs: x})),
                (sess.run(target_network_vars, feed_dict={inputs: x})),
                (sess.run(getQUpdate_ops, feed_dict={inputs: x})),
                rmsprop_ops)
    return function

def _get_optimizer(args, vars):

    if args.optimizer == 'RMSpropCentered':
        optimizer = tf.train.RMSPropOptimizer(args.lr,
                                              decay=args.grad_momentum,
                                              momentum=args.momentum,
                                              epsilon=args.mini_squared_gradient,
                                              centered=True)
    elif args.optimizer == 'RMSpropNonCentered':
        optimizer = tf.train.RMSPropOptimizer(args.lr,
                                              decay=args.grad_momentum,
                                              momentum=args.momentum,
                                              epsilon=args.mini_squared_gradient,
                                              centered=False)
    else:
        raise NotImplementedError()

    return optimizer
