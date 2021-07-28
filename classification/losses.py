import torch.nn as nn


class Criterion(object):
    def __init__(self,
                 loss_type='softmax',
                 loss_weight=1.0,
                 reduction='mean'):
        super(Criterion, self).__init__()
        if loss_type == 'softmax':
            loss_builder = CrossEntropyLoss
        elif loss_type == 'sigmoid':
            loss_builder = BCEWithLogitsLoss
        else:
            raise ValueError('Unknown type %s' % loss_type)
        self.criteria = loss_builder(loss_weight, reduction)

    def __call__(self,
                 pred,
                 true):
        return self.criteria(pred, true)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        # Scalar weight of loss
        self.loss_weight = loss_weight
        self.criteria = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self,
                pred,
                true):
        return self.loss_weight * self.criteria(pred, true)


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        # Scalar weight of loss
        self.loss_weight = loss_weight
        self.criteria = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self,
                pred,
                true):
        return self.loss_weight * self.criteria(pred, true)
