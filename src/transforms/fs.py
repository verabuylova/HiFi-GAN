from dataclasses import dataclass
import torch
import torchaudio
from torch import nn
from speechbrain.inference.TTS import FastSpeech2


@dataclass
class MelSpectrogramFSConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    center: bool = False
    pad_value: float = -11.5129251


class MelSpectrogramFS(nn.Module):
    def __init__(self, config: MelSpectrogramFSConfig):
        super().__init__()
        self.config = config
        self.fastspeech2 = FastSpeech2.from_hparams(
            source="speechbrain/tts-fastspeech2-ljspeech", savedir="pretrained_models/tts-fastspeech2-ljspeech"
        )
        self.pad_value = config.pad_value

    def forward(self, input_text) -> torch.Tensor:
        if isinstance(input_text, list):
            mel_output, durations, pitch, energy = self.fastspeech2.encode_text(
                input_text,
                pace=1.0,
                pitch_rate=1.0,
                energy_rate=1.0,
            )
        else:
            mel_output, durations, pitch, energy = self.fastspeech2.encode_text(
                [input_text],
                pace=1.0,
                pitch_rate=1.0,
                energy_rate=1.0,
            )

        return mel_output

