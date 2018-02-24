import torch
from torch import nn
import torch.legacy.nn as legacy_nn
from torch.autograd import Function

def setup_before_package_loading(opt):
    pass

def setup(opt):
    torch.manual_seed(opt.seed)
    _create_rectifier(opt)
    _create_weight_init(opt)
    return opt

class LegacyInterface(object):

    @property
    def gradWeight(self):
        return self.weight.grad
    @property
    def gradBias(self):
        return self.bias.grad

# Stateful Layers
class ExConv2d(nn.Conv2d, LegacyInterface):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, name=None):
        super(ExConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias)
        self.output = None
        self._name = name

    def forward(self, input):
        self.output = super(ExConv2d, self).forward(input)
        return self.output

    @property
    def name(self):
        return self._name

class ExLinear(nn.Linear, LegacyInterface):
    def __init__(self, in_features, out_features, bias=True, name=None):
        super(ExLinear, self).__init__(in_features, out_features, bias)
        self.output = None
        self._name = name

    def forward(self, input):
        self.output = super(ExLinear, self).forward(input)
        return self.output

    @property
    def name(self):
        return self._name

weight_init = None
def _create_weight_init(opt):
    global weight_init

    print(opt.initializer)

    if opt.initializer == 'torch_nn_default':
        def _weight_init_torch_nn_default(m):
            pass
        weight_init = _weight_init_torch_nn_default

    else:

        from initializer import get_initializer
        initializer = get_initializer(opt.initializer)

        def _init(data, weight):
            if callable(initializer):
                np_data = data.numpy()
                np_init = initializer.kernel_initializer(np_data.shape, np_data.dtype) if weight else \
                          initializer.bias_initializer(np_data.shape, np_data.dtype)
                data.copy_(torch.from_numpy(np_init).float())
            else:
                raise NotImplementedError()

        def _weight_init_other_initializer(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                _init(m.weight.data, weight=True)
                _init(m.bias.data,  weight=False)
            elif isinstance(m, (legacy_nn.SpatialConvolution, legacy_nn.Linear)):
                _init(m.weight,  weight=True)
                _init(m.bias,  weight=False)

        weight_init = _weight_init_other_initializer


Rectifier = None
def _create_rectifier(opt):
    global Rectifier
    if opt.backend == 'pytorch_legacy':
        if opt.relu:
            Rectifier = legacy_nn.ReLU
        else:
            # DQN3.0 Rectifier reproduced by PyTorch(Legacy)
            class _Rectifier(legacy_nn.Module):
                def updateOutput(self, input):
                    return self.output.resize_as_(input).copy_(input).abs_().add_(input).div_(2)
                def updateGradInput(self, input, gradOutput):
                    self.gradInput.resize_as_(self.output)
                    return self.gradInput.set_(torch.sign(self.output) * gradOutput)
            Rectifier = _Rectifier
    elif opt.backend == 'pytorch':
        # DQN3.0 Rectifier reproduced by PyTorch(New)
        class RectifierFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                output = input.abs().add(input).div(2.0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output = ctx.saved_variables
                return output[0].sign() * grad_output

        class RectifierNew(nn.ReLU):
            def __init__(self, name=None):
                super(RectifierNew, self).__init__()
                self.output = None
                self._name = name

            def forward(self, input):
                if opt.relu:
                    self.output = super(RectifierNew, self).forward(input)
                else:
                    self.output = RectifierFunction.apply(input)
                return self.output

            @property
            def name(self):
                return self._name

            def __repr__(self):
                if opt.relu:
                    class_name = 'nn.ReLU'
                else:
                    class_name = self.__class__.__name__

                inplace_str = ', inplace' if self.inplace else ''
                return class_name + '(' \
                    + str(0) \
                    + ', ' + str(0) \
                    + inplace_str + ')'
            
        Rectifier = RectifierNew
    else:
        assert False
