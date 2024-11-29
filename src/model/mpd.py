# https://arxiv.org/pdf/2010.05646.pdf, 2 HiFi-GAN, Appendix A
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm


class MPD(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p = p
        kernel, stride = (5, 1), (3, 1)
        channels = [32, 64, 128, 512]

        def conv_block(in_channels, out_channels, kernel, stride, padding=(2, 0)):
            return nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, out_channels, kernel, stride, padding=padding)),
                nn.LeakyReLU()
            )

        layers = [
            conv_block(1, channels[0], kernel, stride) if i == 0 else conv_block(channels[i-1], channels[i], kernel, stride)
            for i in range(4)
        ]
        layers.extend([
            weight_norm(nn.Conv2d(512, 1024, kernel, padding=(2, 0))),
            nn.LeakyReLU(),
            weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))
        ])

        self.layers = nn.ModuleList(layers)


    def forward(self, wav):

        if wav.shape[-1] % self.p != 0:
            wav = F.pad(wav, (0, self.p - wav.shape[-1] % self.p), mode="reflect")
        
        x = wav.view(wav.shape[0], wav.shape[1], -1, self.p) 

        d = [
            x if isinstance(layer, nn.LeakyReLU) else layer(x) 
            for layer in self.layers
        ]

        x = torch.flatten(d[-1], 1, -1)
        return x, d
