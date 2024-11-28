from torch import nn
from torch.nn import Sequential
from torch.nn.utils import weight_norm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, k_r, D_r):
        super().__init__()
        self.ResBlock = nn.ModuleList([
            nn.Sequential(
                *[nn.LeakyReLU(), 
                  weight_norm(nn.Conv1d(in_channels, in_channels, k, padding="same", dilation=d))
                 for k, d in zip(k_r, D_r[m])]
            ) 
            for m in range(len(D_r))
        ])

    def forward(self, x):
        for Block in self.ResBlock:
            x = x + Block(x)
        return x
    
class MRF(nn.Module):
    def __init__(self, k_r, D_r):
        super().__init__()
        self.ResBlock = nn.ModuleList([ResidualBlock(k_r[i], D_r[i]) for i in range(len(k_r))])

    def forward(self, x):
        res = self.ResBlock[0](x)
        for i in range(1, len(self.ResBlock)):
            res = res + self.ResBlock[i](x)
        return res
    
class Generator(nn.Module):
    def __init__(self, k_u, h_u, k_r, D_r):
        super().__init__()
        layers = [nn.Conv1d(inchannels=80, 
                            out_channels=h_u, 
                            kernel_size=7, 
                            padding="same")]
        for i in range(len(k_u)):
            layers.append(nn.LeakyReLu())
            layers.append(nn.ConvTranspose1d(in_channels=2 * (h_u // (2 ** (i + 1))), 
                                             out_channels=(h_u // (2 ** (i + 1))),
                                             kernel_size=k_u[i],
                                             stride=k_u[i] // 2,
                                             stride=k_u[i] // 4
            ))
            layers.append(MRF(k_r, D_r))
        layers.append(nn.LeakyReLU())
        layers.append(weight_norm(nn.Conv1d(in_channels=h_u // (2 ** len(k_u)),
                                            out_channels=1,
                                            kernel_size=7,
                                            padding="same")))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
