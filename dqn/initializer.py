import math
import numpy as np

from utils import get_random

def _compute_fans(shape, data_format):
    # this function is refer to keras.initializers._compute_fans

    if len(shape) == 2:
        # In this case of the parameter of Linear layer,
        # it shape does not include the channel,
        # but if the data_format is 'channels_first',
        # it is probably a column vector
        # such as the FullyConnected layer in MXNet.
        if data_format == 'channels_first':
            fan_in = shape[1]
            fan_out = shape[0]
        elif data_format == 'channels_last':
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    elif len(shape) in {3, 4, 5}:
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))

    return fan_in, fan_out


class Initializer:
    channel_last = True


class Keras_DefaultInitializer(Initializer):
    kernel_initializer = 'glorot_uniform'

    bias_initializer = 'zeros'

    @staticmethod
    def print():
        print('initializer : keras initializer (w:{}, b:{})'.format(Keras_DefaultInitializer.kernel_initializer, Keras_DefaultInitializer.bias_initializer))

class Torch_nn_DefaultInitializer(Initializer):

    stdv = 0

    @staticmethod
    def kernel_initializer(shape, dtype=None):
        cls = Torch_nn_DefaultInitializer
        fan_in, _ = _compute_fans(shape, 'channels_last' if cls.channel_last else 'channels_first')
        stdv = 1.0 / np.sqrt(fan_in)
        cls.stdv = stdv

        return cls._uniform(shape, -stdv, stdv)

    @staticmethod
    def bias_initializer(shape, dtype=None):
        stdv = Torch_nn_DefaultInitializer.stdv

        return Torch_nn_DefaultInitializer._uniform(shape, -stdv, stdv)

    @staticmethod
    def _uniform(shape, low, hight):

        random = get_random()

        if Torch_nn_DefaultInitializer.channel_last:
            shape = [int(s) for s in shape]
            shape.reverse()
            v = np.float32(random.uniform(low, hight, shape=shape))
            v = np.require(v.T, requirements='C') # For CNTK
            return v
        else:
            return np.float32(random.uniform(low, hight, shape=shape))

    @staticmethod
    def print():
        print('initializer : torch nn layers default (w: uniform 1 / sqrt(fan_in), b: uniform 1 / sqrt(fan_in of weight))')

class Uniform_Initializer(Initializer):
    @staticmethod
    def kernel_initializer(shape, dtype=None):
        cls = Uniform_Initializer
        fan_in, _ = _compute_fans(shape, 'channels_last' if cls.channel_last else 'channels_first')
        stdv = 1 / np.sqrt(fan_in)

        from keras import backend as K
        return K.random_uniform_variable(shape, -stdv, stdv)

    @staticmethod
    def bias_initializer(shape, dtype=None):
        cls = Uniform_Initializer
        fan_in, _ = _compute_fans(shape, 'channels_last' if cls.channel_last else 'channels_first')
        stdv = 1 / np.sqrt(fan_in)

        from keras import backend as K
        return K.random_uniform_variable(shape, -stdv, stdv)
    
    @staticmethod
    def print():
        print("initializer : torch nn layers default uniform (w: uniform 1 / sqrt(fan_in), b: uniform sqrt(prod(bias shape))")

class Deep_q_rl_Initializer(Initializer):
    @staticmethod
    def kernel_initializer(shape, dtype=None):
        fan_in, _ = _compute_fans(shape)
        
        stdv = np.sqrt(1 / fan_in)
        from keras import backend as K
        return K.random_uniform_variable(shape, 
                                         -stdv, 
                                         stdv)

    @staticmethod
    def bias_initializer(shape, dtype=None):
        return np.full(shape, 0.1, dtype=np.float32)
    
    @staticmethod
    def print():
        print("initializer : same as deep_q_rl as lasagne.init.HeUniform (w: uniform sqrt(1/fan_in), b:0.1")

initializer_selecter = {'torch_nn_default': Torch_nn_DefaultInitializer,
                        'uniform'         : Uniform_Initializer,
                        'deep_q_rl'       : Deep_q_rl_Initializer}

def get_initializer(type):
    if isinstance(type, str):
        if type in initializer_selecter:
            initializer = initializer_selecter.get(type)
        else:
            initializer = Keras_DefaultInitializer()

            if not type is None:
                initializer_str = type.split(':')

                if len(initializer_str) == 2:
                    w, b = initializer_str
                elif len(initializer_str) == 1:
                    w, b = initializer_str, ''
                else:
                    assert False, '--initializer: got value that invalid format {}'.format(type)

                if len(w) > 0:
                    initializer.kernel_initializer = w
                
                if len(b) > 0:
                    initializer.bias_initializer = b
 
        initializer.print()
    else:
        initializer = type
        print('{} : keras built-in initializer'.format(type))

    return initializer
