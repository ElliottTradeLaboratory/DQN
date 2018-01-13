from collections import deque, OrderedDict
import numpy as np

import torch
from torch.legacy.nn import Module
from torch.autograd import Variable


from convnet_common import Convnet, Trainer
from visualizer import LearningVisualizer

class Utils(object):
    def __init__(self, args=None):
        if args is not None:
            self.args = args
    def _get_numpy(self, v):
        if v is None:
            return v
        elif np.isscalar(v):
            return np.array([v])
        elif isinstance(v, list):
            return np.hstack(tuple([self._get_numpy(_v).flatten() for _v in v]))
        elif isinstance(v, torch.autograd.Variable):
            return v.data.clone().cpu().numpy() if self.args.gpu >= 0 else \
                   v.data.clone().numpy()
        return v.clone().cpu().numpy() if self.args.gpu >= 0 else \
               v.clone().numpy()

    def _conv_numpy_to_pytorch(self, x):
        s, a, r, s2, _term = x

        self.s = s
        self.s2 = s2

        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r.astype(np.float32)).float()
        s2 = torch.from_numpy(s2).float()
        term = torch.from_numpy(_term).float()

        if self.args.backend == 'pytorch':
            s, a, r, s2, term = [Variable(t) for t in [s, a, r, s2, term]]

        if self.args.gpu >= 0:
            s, a, r, s2, term = [v.cuda(self.args.gpu) for v in [s, a, r, s2, term]]

        return s, a, r, s2, term

class PyTorchConvnet(Convnet, Utils):

    def __init__(self, args, network_name):
        super(PyTorchConvnet, self).__init__(args, network_name)
        if args.backend == 'pytorch_legacy':
            self.model, self.saver, self.loader = self._create_model_legacy()
        else:
            self.model, self.saver, self.loader = self._create_model()

    def _create_model(self):
        from collections import deque
        from torch.nn import Sequential, Module
        from pytorch_extensions import ExLinear, ExConv2d, Rectifier

        names = deque(self.summarizable_layer_names)

        self._modules = [ExConv2d(4, 32, kernel_size=8, stride=4, padding=1, name=names.popleft()),
                         Rectifier(name=names.popleft()),
                         ExConv2d(32, 64, kernel_size=4, stride=2, name=names.popleft()),
                         Rectifier(name=names.popleft()),
                         ExConv2d(64, 64, kernel_size=3, stride=1, name=names.popleft()),
                         Rectifier(name=names.popleft()),
                         ExLinear(3136, 512, name=names.popleft()),
                         Rectifier(name=names.popleft()),
                         ExLinear(512, self.args.n_actions, name=names.popleft())]

        features = Sequential()
        for name_and_mod in zip(self.summarizable_layer_names[:6], self._modules[:6]):
            features.add_module(*name_and_mod)

        classifier = Sequential()
        for  name_and_mod in zip(self.summarizable_layer_names[6:], self._modules[6:]):
            classifier.add_module(*name_and_mod)
        
        class Model(Module):
            def __init__(self, args, features, classifier):
                super(Model, self).__init__()
                self.features = features
                self.classifier = classifier
                self.gpu = args.gpu

            def forward(self, s):
                return self.forward_train(s).data

            def forward_train(self, s):
                if isinstance(s, np.ndarray):
                    s = torch.from_numpy(s).float()
                if not isinstance(s, Variable):
                    s = Variable(s, requires_grad=False)
                if self.gpu >= 0:
                    s = s.cuda(self.gpu)
                x = self.features.forward(s)
                x = x.view(x.size(0), 3136)
                x = self.classifier.forward(x)
                return x

            def parameters(self):
                params = []
                grads = []
                for p in super(Model, self).parameters():
                    params.append(p)
                    grads.append(p.grad)
                return params, grads

            def zeroGradParameters(self):
                self.features.zero_grad()
                self.classifier.zero_grad()

            @property
            def modules(self):
                mods = [mod for mod in super(Model, self).modules() \
                            if not isinstance(mod, Sequential) and not isinstance(mod, Model)]
                return mods

        model = Model(self.args, features, classifier)

        if self.args.gpu >= 0:
            model = model.cuda(self.args.gpu)

        if self.args.verbose >= 2:
            print('--------{}--------'.format(self.network_name))
            print(model)
            print('Convolutional layers flattened output size:', 3136)

        def saver(filepath, model):
            state_dict = model.state_dict()
            torch.save(state_dict, filepath)

        def loader(filepath, model):
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict)

        return model, saver, loader

    def _create_model_legacy(self):
        from torch.legacy.nn import Sequential, Linear, SpatialConvolution, Reshape
        from torch.legacy.nn import Module
        from pytorch_extensions import Rectifier

        layers = [SpatialConvolution(4, 32, 8, 8, 4, 4, 1),
                  Rectifier(),
                  SpatialConvolution(32, 64, 4, 4, 2, 2),
                  Rectifier(),
                  SpatialConvolution(64, 64, 3, 3, 1, 1),
                  Rectifier(),
                  Reshape(3136),
                  Linear(3136, 512),
                  Rectifier(),
                  Linear(512, self.args.n_actions)]
                  
        model = Sequential()

        for mod, layer_name in zip(layers, self.layer_names):
            mod.name = layer_name
            model.add(mod)

        if self.args.gpu >= 0:
            model = model.cuda()

        if self.args.verbose >= 2:
            print('--------{}--------'.format(self.network_name))
            print(model)
            print('Convolutional layers flattened output size:', 3136)
        
        def saver(filepath, model):
            params = model.flattenParameters()
            torch.save(params, filepath)

        def loader(filepath, model):
            loaded_params = torch.load(filepath)
            w, dw = model.flattenParameters()
            w.copy_(loaded_params[0])
            dw.copy_(loaded_params[1])

        return model, saver, loader

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()

        if self.args.gpu >= 0:
            s = s.cuda(self.args.gpu)

        predict = self.model.forward(s)
        if self.args.gpu >= 0:
            return predict.cpu().numpy()
        else:
            return predict.numpy()

    def forward_train(self, s):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()

        if self.args.gpu >= 0:
            s = s.cuda(self.args.gpu)

        if self.args.backend == 'pytorch':
            predict = self.model.forward_train(s)
        else:
            predict = self.model.forward(s)
        return predict

    def get_summarizable_modules(self):
        for mod in self.model.modules:
            if mod.name in self.summarizable_layer_names:
                yield mod

    def _get_params_dict(self, numpy):
        params_dict = OrderedDict()
        for mod in self.model.modules:
            params = [mod.output,
                      getattr(mod, 'weight', None),
                      getattr(mod, 'bias', None),
                      getattr(mod, 'gradWeight', None),
                      getattr(mod, 'gradBias', None)]
            if numpy:
                params = [self._get_numpy(p) for p in params]
            params_dict[mod.name] = tuple(params)
        return params_dict

    def _save(self, filepath):
        self.saver(filepath, self.model)
    def _load(self, filepath):
        self.loader(filepath, self.model)

class PyTorchTrainer(Trainer, Utils):
    def __init__(self, args, network, target_network):
        super(PyTorchTrainer, self).__init__(args, network, target_network)
        from pytorch_optimizer import get_optimizer

        self.optimizer = get_optimizer(self.args)(self)

        if args.backend =='pytorch':
            def updater(network, target_network):
                state_dicts = (network.model.features.state_dict(),
                          network.model.classifier.state_dict())
                target_network.model.features.load_state_dict(state_dicts[0])
                target_network.model.classifier.load_state_dict(state_dicts[1])
            self.updater = updater
        else:
            def updater(network, target_network):
                w, dw = network.model.flattenParameters()
                tw, tdw = target_network.model.flattenParameters()
                tw.copy_(w.clone())
                tdw.copy_(dw.clone())
            self.updater = updater

    def compute_validation_statistics(self, x):

        x = self._conv_numpy_to_pytorch(x)

        ret= self.optimizer.getQUpdate(x)

        return [self._get_numpy(v) for v in ret[1:3]]


    def _qLearnMinibatch(self, x, do_summary):

        x = self._conv_numpy_to_pytorch(x)

        self.optimizer.qLearnMinibatch(x)

    def _update_target_network(self):
        self.updater(self.network, self.target_network)

    def _get_trainer_attrs(self, names):
        params = []
        for name in names:
            p = getattr(self.optimizer, name, None)
            if p is not None:
                p = self._get_numpy(p)
            params.append(p)
            
        return params

    @property
    def _getQUpdate_values(self):
        return self._get_trainer_attrs(LearningVisualizer.GET_Q_UPDATE_VALUE_NAMES)

    @property
    def _rmsprop_values(self):
        return self._get_trainer_attrs(LearningVisualizer.RMS_PROP_VALUE_NAMES[:-1])


def create_networks(args):
    from utils import get_random

    get_random().manualSeed(args.seed)
    network = PyTorchConvnet(args, 'network')
    target_network = PyTorchConvnet(args, 'target_network')
    trainer = PyTorchTrainer(args, network, target_network)

    return network, trainer
