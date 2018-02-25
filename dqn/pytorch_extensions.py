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

        from initializer import compute_fans
        """
        initializer = get_initializer(opt.initializer)

        class WeightInitializer(object):
            def __call__(self, m):
                w = None
                b = None
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    w = m.weight.data
                    b = m.bias.data
                elif isinstance(m, (legacy_nn.SpatialConvolution, legacy_nn.Linear)):
                    w = m.weight
                    b = m.bias

                if w is not None:
                    self.init_w = initializer.kernel_initializer(w.numpy().shape)
                    w.copy_(torch.from_numpy(self.init_w).float())
                    self.init_b = initializer.bias_initializer(b.numpy().shape)
                    b.copy_(torch.from_numpy(self.init_b).float())
        """

        class WeightInitializer(object):
            def __call__(self, m):
                w = None
                b = None
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    w = m.weight.data
                    b = m.bias.data
                elif isinstance(m, (legacy_nn.SpatialConvolution, legacy_nn.Linear)):
                    w = m.weight
                    b = m.bias

                if w is not None:
                    self._uniform(w)
                    self._uniform(b)

            def _uniform(self, data):
                fan_in, _ = compute_fans(data.numpy().shape, 'channels_first')
                stdv = 1 / np.sqrt(fan_in)
                self.uniform_(-stdv, stdv)

        weight_init = WeightInitializer()


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

        class RectifierNew(nn.Module):
            def __init__(self, name=None):
                super(RectifierNew, self).__init__()
                self.output = None
                self._name = name

            def forward(self, input):
                if opt.relu:
                    self.output = nn.functional.relu(input)
                else:
                    self.output = RectifierFunction.apply(input)
                return self.output

            @property
            def name(self):
                return self._name

            def __repr__(self):
                if opt.relu:
                    class_name = 'nn.functional.relu'
                else:
                    class_name = 'DQN3.0 Rectifer'

                return class_name
            
        Rectifier = RectifierNew
    else:
        assert False
