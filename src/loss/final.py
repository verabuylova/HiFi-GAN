import torch
from torch import nn
import torch.nn.functional as F

from src.loss.discgan import DiscGANLoss
from src.loss.gengan import GenGANLoss
from src.loss.mel import MelSpecLoss
from src.loss.fm import FeatureMatchingLoss

class FinalLoss(nn.Module):
    def __init__(self, fm_l, mel_l):
        super().__init__()
        self.discgan = DiscGANLoss()
        self.gengan = GenGANLoss()
        self.mel = MelSpecLoss(mel_l)
        self.fm = FeatureMatchingLoss(fm_l)
    