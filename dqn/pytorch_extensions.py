import torch
from torch import nn
from torch.legacy.nn import Module as Legacy_Module
from torch.autograd import Function

def setup_before_package_loading(opt):
    pass

def setup(opt):
    torch.manual_seed(opt.seed)
    return opt

# DQN3.0 Rectifier reproduced by PyTorch(Legacy)
class Rectifier(Legacy_Module):
    def updateOutput(self, input):
        return self.output.resize_as_(input).copy_(input).abs_().add_(input).div_(2)
    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(self.output)
        return self.gradInput.set_(torch.sign(self.output) * gradOutput)

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
        super(RectifierNew, self).__init__(False)
        self.output = None
        self._name = name

    def forward(self, input):
        self.output = RectifierFunction.apply(input)
        return self.output

    @property
    def name(self):
        return self._name

