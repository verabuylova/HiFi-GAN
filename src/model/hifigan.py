from typing import Dict

import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm

import torch.nn.functional as F

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            convs = nn.ModuleList([
                nn.Sequential(
                    spectral_norm(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
                )
            ])
            
            self.discriminators.append(convs)

    def forward(self, audio) :
        features = []
        for period, convs in zip(self.periods, self.discriminators):
            pad_right = period - (audio.size(1) % period)
            if pad_right > 0 and pad_right != period:
                padded_audio = F.pad(audio, (0, pad_right), mode="reflect")
            else:
                padded_audio = audio
            x = padded_audio.view(audio.size(0), 1, -1, period)
            
            res = []
            for conv in convs:
                x = conv(x)
                res.append(x)
            
            features.append(res)
        
        return features

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(3):
            convs_d = nn.ModuleList([
                nn.Sequential(
                    weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, dilation=1, groups=4)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, dilation=1, groups=16)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, dilation=1, groups=64)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, dilation=1, groups=256)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=2))
                )
            ])
            self.convs.append(convs_d)

        self.avg_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])
    
    def forward(self, audio):
        res = []
        
        for scale in range(len(self.convs)):
            if scale == 0:
                x = audio
            else:
                x = self.avg_pools[scale - 1](audio)
            x = x.unsqueeze(1)
            
            r = []
            for layer in self.convs[scale]:
                x = layer(x)
                r.append(x)
            
            res.append(r)
        
        return res

class MRF(nn.Module):
    def __init__(self, in_channels, kernels=[3, 7, 11], dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super(MRF, self).__init__()

        self.layers = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            layer = nn.Sequential(*[
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.Conv1d(in_channels, in_channels, k, 1, 'same', dilation))
                ) for dilation in d
            ])
            self.layers.append(layer)

    def forward(self, x):
        res = 0
        for layer in self.layers:
            res = res + x + layer(x)
        return res / len(self.layers)

class HiFiGanGenerator(nn.Module):
    def __init__(self, h_u=128, k_u=[16, 16, 4, 4]):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(80, h_u, kernel_size=7, padding=3))
        
        layers = []
        for i, k in enumerate(k_u):
            in_channels = h_u // (2 ** i)
            out_channels = h_u // (2 ** (i + 1))
            layers += [
                nn.LeakyReLU(0.1),
                weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=k, stride=k//2, padding=k//4)),
                MRF(out_channels)
            ]
        self.layers = nn.Sequential(*layers)
        
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(h_u // (2 ** len(k_u)), 1, kernel_size=7, padding=3))
        )
        self.tanh = nn.Tanh()

    def forward(self, spectrogram: torch.Tensor, **batch) -> Dict:
        x = self.conv1(spectrogram)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.tanh(x)

        return {"output_audio": torch.flatten(x, start_dim=1)}


class HiFiGanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()

    def forward(self, output_audio, audio, detach_generated=False, **batch) -> Dict:
        if detach_generated:
            output_audio = output_audio.detach()
        return {
            "msd_gt": self.msd(audio),
            "mpd_gt": self.mpd(audio),
            "msd_pred": self.msd(output_audio),
            "mpd_pred": self.mpd(output_audio),
        }