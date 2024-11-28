import torch
from torch import nn
import torch.nn.functional as F

class FeatureMatchingLoss(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, label_feats: torch.Tensor, pred_feats: torch.Tensor, **batch):
        loss = 0
        for label_feat, pred_feat in zip(label_feats, pred_feats):
            for label, pred in zip(label_feat, pred_feat):
                loss = loss + F.l1_loss(pred, label)
        loss = self.l * loss
        return {"fm_loss": loss}