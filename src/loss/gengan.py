import torch
from torch import nn

class GenGANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, **batch):
        loss = 0.0
        for pred in preds:
            loss = loss + torch.mean((pred - 1) ** 2)
        return {"gen_loss": loss}