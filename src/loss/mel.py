import torch
from torch import nn
import torch.nn.functional as F

class MelSpecLoss(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, labels: torch.Tensor, preds: torch.Tensor, **batch):
        loss = self.l * F.l1_loss(preds, labels)
        return {"mel_loss": loss}