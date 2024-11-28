import torch
from torch import nn


class DiscGANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels: torch.Tensor, preds: torch.Tensor, **batch):
        loss = 0.0
        for label, pred in (labels, preds):
            loss = loss + torch.mean((label - 1) ** 2) +  torch.mean(pred ** 2)
        return {"disc_loss": loss}
    
    
