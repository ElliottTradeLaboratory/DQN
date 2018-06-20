import torch
import numpy as np

class PytorchLegacy_DQN30_Optimizer(object):
    # Reproduced DQN3.0 using PyTorch(Legacy)
    
    def __init__(self, trainer):
        self.network = trainer.network
        self.target_network = trainer.target_network
        self.optimizer = trainer.args.optimizer
        self.loss_function = trainer.args.loss_function
        self.discount = trainer.args.discount
        self.backend = trainer.args.backend
        self.lr = trainer.args.lr
        self.wc = trainer.args.wc
        self.grad_momentum = trainer.args.grad_momentum
        self.sqared_grad_momentum = trainer.args.sqared_grad_momentum
        self.mini_squared_gradient = trainer.args.mini_squared_gradient
        self.momentum = trainer.args.momentum

        self.clip_delta = trainer.args.clip_delta
        self.n_actions = trainer.args.n_actions
        self.minibatch_size = trainer.args.minibatch_size

        self.gpu = trainer.args.gpu

        self._init_param()

    def _init_param(self):

        ws, dws = self.network.model.parameters()

        dws = [dw.zero_() for dw in dws]

        self.deltas = [dw.clone() for dw in dws]

        self.g  = [dw.clone() for dw in dws]
        self.g2 = [dw.clone() for dw in dws]

        if self.gpu >= 0:
            self.deltas = [v.cuda(self.gpu) for v in self.deltas]
            self.g  = [v.cuda(self.gpu) for v in self.g]
            self.g2 = [v.cuda(self.gpu) for v in self.g2]

    def compute_q2(self, s2, r, term):

        self.r = r.clone()

        # delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
        term = term.clone().float().mul(-1).add(1)


        # Compute max_a Q(s_2, a).
        q2_all = self.target_network.forward_train(s2)
        if self.backend == 'pytorch':
            # detach autograd graph
            q2_all = q2_all.detach()

        q2_max, _ = q2_all.float().max(1)

        self.q2_all = q2_all
        self.q2_max = q2_max

        # Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
        q2 = q2_max.mul(self.discount).mul(term)

        q2.add_(r.clone().float())

        self.q2 = q2

        return q2, q2_max

    def compute_q(self, s, a):

        # q = Q(s,a)
        self.q_all = self.network.forward_train(s).float()

        # same as DQN3.0
        if isinstance(self.q_all, torch.autograd.Variable):
            q_all = self.q_all.data
        else:
            q_all = self.q_all.clone()

        q = torch.FloatTensor(q_all.size(0))
        for i in range(q_all.size(0)):
            q[i] = q_all[i][a[i]]
        self.q = q.clone()
        if self.gpu >= 0:
            q = q.cuda(self.gpu)

        return q
 
    def getQUpdate(self, x):
        s, a, r, s2, term = x

        delta, q2_max = self.compute_q2(s2, r, term)

        q = self.compute_q(s, a)

        delta.add_(q.mul(-1))

        if self.clip_delta :
            delta[delta.ge(self.clip_delta)] = self.clip_delta
            delta[delta.le(-self.clip_delta)] = -self.clip_delta

        targets = torch.zeros(self.minibatch_size, self.n_actions).float()
        if self.gpu >= 0:
            targets = targets.cuda(self.gpu)

        for i in range(min(self.minibatch_size, a.size(0))):
            targets[i][a[i]] = delta[i]

        if self.gpu >= 0:
             targets = targets.cuda(self.gpu)

        self.delta = delta.clone()
        self.targets = targets.clone()

        return targets, delta, q2_max

    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, _, _ = self.getQUpdate(x)

        self.network.model.zeroGradParameters()

        self.network.model.backward(s, self.targets)

        ws, dws = self.network.model.parameters()

        self.rmsprop(ws, dws)

    def rmsprop(self, ws, dws):
        self.tmp = []
        for w, dw, g, g2, deltas in zip(ws, dws, self.g, self.g2, self.deltas):

            # add weight cost to gradient
            if isinstance(w, torch.autograd.Variable):
                dw = dw.data
                dw.add_(-self.wc, w.data)
            else:
                dw.add_(-self.wc, w)

            g.mul_(self.grad_momentum).add_(1 - self.grad_momentum, dw)
            g2.mul_(self.sqared_grad_momentum).addcmul_(1 - self.sqared_grad_momentum, dw, dw)
            tmp = g2.addcmul(-1, g, g).add_(self.mini_squared_gradient).sqrt()
            deltas.mul_(self.momentum).addcdiv_(self.lr, dw, tmp)
            if isinstance(w, torch.nn.parameter.Parameter):
                w.data.add_(deltas)
            else:
                w.add_(deltas)
            self.tmp.append(tmp)

class PytorchLegacy_DQN30loss_RMSprop(PytorchLegacy_DQN30_Optimizer):
    # PyTorch(Legacy) DQN3.0 loss & built-in RMSprop
    def __init__(self, trainer):
        super(PytorchLegacy_DQN30loss_RMSprop, self).__init__(trainer)
        self.rmsprop_config = {
            'learningRate'  : self.lr,
            'alpha'         : self.grad_momentum,
            'epsilon'       : self.mini_squared_gradient,
            'weightDecay'   : self.wc
        }

        def rmsprop(ws, dws):
            # The parameter of layer in Torch7 are flattened & concatenated all layers parameters.
            # But pytorch is not.
            # Because of this, optimize each layer parameters using for loop.
            for w, dw, g, deitas in zip(ws, dws, self.g, self.deltas):
                state = {
                    'm'   : g,
                    'tmp' : delta
                }
                torch.legacy.optim.rmsprop(lambda x: (None, dw), w, rmsprop_config, state)
        self.rmsprop = rmsprop

class PytorchLegacy_HuberLoss_DQN30RMSprop(PytorchLegacy_DQN30_Optimizer):
    # PyTorch(Legacy) Huber loss & DQN3.0 RMSprop
    def __init__(self, trainer):
        super(PytorchLegacy_HuberLoss_DQN30RMSprop, self).__init__(trainer)

        self.huber_loss = torch.legacy.nn.SmoothL1Criterion()
        if self.gpu >= 0:
            self.huber_loss = self.huber_loss.cuda(self.gpu)

    def getQUpdate(self, x):
        s, a, r, s2, term = x

        q2, q2_max = self.compute_q2(s2, r, term)
        q = self.compute_q(s, a)
        
        self.delta = self.huber_loss.forward(q2, q)

        delta_grad = self.huber_loss.backward(q2, q)

        targets = torch.zeros(self.minibatch_size, self.n_actions).float()
        if self.gpu >= 0:
            targets = targets.cuda(self.gpu)

        for i in range(min(self.minibatch_size, a.size(0))):
            targets[i][a[i]] = delta_grad[i]

        self.targets = targets.clone()

        return self.targets, self.delta, q2_max


class Pytorch_DQN30_Optimizer(PytorchLegacy_DQN30_Optimizer):
    # PyTorch DQN3.0 optimizer
    def __init__(self, trainer):
        super(Pytorch_DQN30_Optimizer, self).__init__(trainer)

    def _init_param(self):

        ws, dws = self.network.model.parameters()

        dws = [torch.zeros(w.size()).float() for w in ws]

        self.deltas = [dw.clone() for dw in dws]

        self.g  = [dw.clone() for dw in dws]
        self.g2 = [dw.clone() for dw in dws]

        if self.gpu >= 0:
            self.deltas = [v.cuda(self.gpu) for v in self.deltas]
            self.g  = [v.cuda(self.gpu) for v in self.g]
            self.g2 = [v.cuda(self.gpu) for v in self.g2]


    def compute_q(self, s, a):

        # q = Q(s,a)
        q_all = self.network.model.forward_train(s).float()
        
        # compute one hot a
        batch_size = q_all.size(0)
        a_one_hot = torch.zeros(batch_size, self.n_actions)
        a_one_hot = torch.autograd.Variable(a_one_hot, requires_grad=False)
        if self.gpu >= 0:
            a_one_hot = a_one_hot.cuda(self.gpu)
        a = a.resize(batch_size, 1)
        a_one_hot.scatter_(1, a, 1)

        q = q_all.mul(a_one_hot)
        q = q.sum(1)
        self.q_all = q_all
        self.q = q

        if self.gpu >= 0:
            q = q.cuda(self.gpu)

        return q, a_one_hot
            
    def getQUpdate(self, x):
        s, a, r, s2, term = x

        delta, q2_max = self.compute_q2(s2, r, term)

        q, a_one_hot = self.compute_q(s, a)

        delta = delta.add(q.mul(-1))

        if self.clip_delta :
            delta = delta.clamp(-self.clip_delta, self.clip_delta)

        targets = delta.resize(delta.size(0), 1) * a_one_hot

        return targets, delta, q2_max

    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, self.delta, self.q2_max = self.getQUpdate(x)

        self.network.model.zeroGradParameters()

        self.q_all.backward(self.targets)

        ws, dws = self.network.model.parameters()

        self.rmsprop(ws, dws)

class Pytorch_BackwardFromTargets1_DQN30RMSprop(Pytorch_DQN30_Optimizer):
    def __init__(self, trainer):
        super(Pytorch_BackwardFromTargets1_DQN30RMSprop, self).__init__(trainer)


    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, self.delta, self.q2_max = self.getQUpdate(x)

        self.network.model.zeroGradParameters()

        # Rise RuntimeError because PyTorch couldn't grad from tensor that num-element > 1
        self.targets.backward()

        ws, dws = self.network.model.parameters()

        self.rmsprop(ws, dws)

class Pytorch_BackwardFromTargets2_DQN30RMSprop(Pytorch_DQN30_Optimizer):
    def __init__(self, trainer):
        super(Pytorch_BackwardFromTargets2_DQN30RMSprop, self).__init__(trainer)


    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, self.delta, self.q2_max = self.getQUpdate(x)

        self.network.model.zeroGradParameters()

        self.targets.backward(0-self.targets)

        ws, dws = self.network.model.parameters()

        self.rmsprop(ws, dws)

class Pytorch_HuberLoss_DQN30RMSprop(Pytorch_DQN30_Optimizer):
    # PyTorch Huber loss & DQN3.0 RMSprop with Autograd
    def __init__(self, trainer):
        super(Pytorch_HuberLoss_DQN30RMSprop, self).__init__(trainer)

        self.huber_loss = torch.nn.SmoothL1Loss()
        if self.gpu >= 0:
            self.huber_loss.cuda(self.gpu)

    def getQUpdate(self, x):
        s, a, r, s2, term = x

        q2, q2_max = self.compute_q2(s2, r, term)
        q, _ = self.compute_q(s, a)
        
        self.targets = self.huber_loss(q, q2.detach())
        self.delta = self.targets

        return self.targets, self.delta, self.q2_max

    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, _, _ = self.getQUpdate(x)

        self.network.model.zeroGradParameters()

        self.targets.backward()

        ws, dws = self.network.model.parameters()

        self.rmsprop(ws, dws)

class Pytorch_DQN30Loss_RMSpropCentered(PytorchLegacy_DQN30_Optimizer):
    # PyTorch DQN3.0 loss & built-in RMSprop centered w/o Autograd
    def __init__(self, trainer):
        super(Pytorch_DQN30Loss_RMSpropCentered, self).__init__(trainer)

        self.rmsprop = RMSprop([{'params':self.network.model.features.parameters()},
                                {'params':self.network.model.classifier.parameters()}],
                               lr=self.lr,
                               alpha =self.grad_momentum,
                               eps=self.mini_squared_gradient,
                               momentum=self.momentum,
                               weight_decay=self.wc,
                               centered=True)
    def _init_param(self):
        pass

    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, _, _ = self.getQUpdate(x)

        self.rmsprop.zero_grad()

        self.q_all.backward(s, self.targets)

        self.rmsprop.step()

class Pytorch_HuberLoss_RMSpropCentered(Pytorch_HuberLoss_DQN30RMSprop):
    # PyTorch Huber loss & built-in RMSpropCentered with Autograd
    def __init__(self, trainer):
        super(Pytorch_HuberLoss_RMSpropCentered, self).__init__(trainer)

        self.rmsprop = RMSprop([{'params':self.network.model.features.parameters()},
                                {'params':self.network.model.classifier.parameters()}],
                               lr=self.lr,
                               alpha =self.grad_momentum,
                               eps=self.mini_squared_gradient,
                               momentum=self.momentum,
                               weight_decay=self.wc,
                               centered=True)

    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, _, _ = self.getQUpdate(x)

        self.rmsprop.zero_grad()

        self.targets.backward(s, self.targets)

        self.rmsprop.step()


class Pytorch_HuberLoss_RMSprop(Pytorch_HuberLoss_DQN30RMSprop):
    # PyTorch Huber loss & built-in RMSprop with Autograd
    def __init__(self, trainer):
        super(Pytorch_HuberLoss_RMSprop, self).__init__(trainer)
        print('Select Pytorch_HuberLoss_RMSprop')

        self.rmsprop = torch.optim.RMSprop([{'params':self.network.model.features.parameters()},
                                            {'params':self.network.model.classifier.parameters()}],
                                           lr=self.lr,
                                           alpha =self.grad_momentum,
                                           eps=self.mini_squared_gradient,
                                           momentum=self.momentum,
                                           weight_decay=self.wc,
                                           centered=False)

    def qLearnMinibatch(self, x):

        s, a, r, s2, term = x

        self.targets, _, _ = self.getQUpdate(x)

        self.rmsprop.zero_grad()

        self.targets.backward()

        self.rmsprop.step()

def get_optimizer(args):
    optimizer_factory = {
        'pytorch_legacy:DQN3.0:DQN3.0'            :PytorchLegacy_DQN30_Optimizer,
        'pytorch_legacy:huber:DQN3.0'             :PytorchLegacy_HuberLoss_DQN30RMSprop,
        'pytorch_legacy:DQN3.0:RMSprop'           :PytorchLegacy_DQN30loss_RMSprop,
        'pytorch:DQN3.0:DQN3.0'                   :Pytorch_DQN30_Optimizer,
        'pytorch:BFT1:DQN3.0'                     :Pytorch_BackwardFromTargets1_DQN30RMSprop,
        'pytorch:BFT2:DQN3.0'                     :Pytorch_BackwardFromTargets2_DQN30RMSprop,
        'pytorch:huber:DQN3.0'                    :Pytorch_HuberLoss_DQN30RMSprop,
        'pytorch:DQN3.0:RMSpropCentered'          :Pytorch_DQN30Loss_RMSpropCentered,
        'pytorch:huber:RMSpropCentered'           :Pytorch_HuberLoss_RMSpropCentered,
        'pytorch:huber:RMSprop'                   :Pytorch_HuberLoss_RMSprop,

    }
    try:
        optimizer_type = '{}:{}:{}'.format(args.backend, args.loss_function, args.optimizer)
        optimizer = optimizer_factory[optimizer_type]
    except KeyError:
        assert False, 'optimizer:{} is not supported'.format(optimizer_type)
    
    return optimizer
