import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        # Scalar weight of loss
        self.loss_weight = loss_weight
        self.criteria = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, true):
        return self.loss_weight * self.criteria(pred, true)


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        # Scalar weight of loss
        self.loss_weight = loss_weight
        self.criteria = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, true):
        return self.loss_weight * self.criteria(pred, true)
