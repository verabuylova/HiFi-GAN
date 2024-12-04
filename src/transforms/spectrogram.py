from dataclasses import dataclass

import librosa
import torch
import torchaudio
from torch import nn


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    center: bool = False

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    def __init__(self, config: MelSpectrogramConfig, normalize_audio: bool = False):
        super().__init__()

        self.config = config
        self.normalize_audio = normalize_audio

        # Set the default device to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=config.power,
            center=config.center,
        ).to(self.device)  # Move the transform to the device

        # needed for collate_fn paddings
        self.pad_value = config.pad_value

        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
        ).T
        mel_basis = torch.tensor(mel_basis, dtype=torch.float32).to(self.device)
        self.mel_spectrogram.mel_scale.fb.copy_(mel_basis)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        audio = audio.to(self.device)

        if self.normalize_audio:
            audio = audio / torch.abs(audio).max(dim=1, keepdim=True)[0]

        audio = torch.nn.functional.pad(
            audio,
            (
                (self.config.n_fft - self.config.hop_length) // 2,
                (self.config.n_fft - self.config.hop_length) // 2,
            ),
            mode="reflect",
        )

        mel = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()

        return mel