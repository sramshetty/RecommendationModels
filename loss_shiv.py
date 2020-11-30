import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1', use_cuda=True):
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        self.use_cuda = use_cuda

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        if loss_type == 'XE':
            self._loss_fn = SampledCrossEntropyLoss(use_cuda)
        elif loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        elif loss_type == 'BPR':
            self._loss_fn = BPRLoss()
        elif loss_type == 'TOP1-max':
            self._loss_fn = TOP1_max()
        elif loss_type == 'BPR-max':
            self._loss_fn = BPR_max()
        else:
            raise NotImplementedError

    def forward(self, logit):
        return self._loss_fn(logit)


class SampledCrossEntropyLoss(nn.Module):
    def __init__(self, use_cuda):
        super(SampledCrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda

    def forward(self, logit):
        batch_size = logit.size(1)
        target = Variable(torch.arange(batch_size).long())
        if self.use_cuda: target = target.cuda()

        return self.xe_loss(logit, target)

class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, logit):
        diff = logit - logit.diag().view(-1, 1).expand_as(logit)
        return torch.mean(torch.sigmoid(diff) + torch.sigmoid(logit ** 2))


class TOP1_max(nn.Module):
    def __init__(self):
        super(TOP1_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit - logit.diag().view(-1, 1).expand_as(logit)
        return torch.mean(logit_softmax *  (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, logit):
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        return - torch.mean(F.logsigmoid(diff))


class BPR_max(nn.Module):
    def __init__(self):
        super(BPR_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        return - torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))