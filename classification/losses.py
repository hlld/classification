import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 loss_weight,
                 reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        # Scalar weight of classification loss
        self.loss_weight = loss_weight
        self.criteria = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, true):
        return self.loss_weight * self.criteria(pred, true)
