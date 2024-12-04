import torch
from torch import nn

from src.transforms import MelSpectrogram, MelSpectrogramConfig

import torch.nn.functional as F

class MPDLoss:
    def __init__(self):
        pass 
    
    def __call__(self, label, pred):
        return torch.mean((label - torch.ones_like(label)) ** 2) + torch.mean((pred - torch.zeros_like(pred)) ** 2)

class MSDLoss:
    def __init__(self):
        pass
    
    def __call__(self, label, pred):
        return torch.mean((label - torch.ones_like(label)) ** 2) + torch.mean((pred - torch.zeros_like(pred)) ** 2)
    

class DiscLoss(nn.Module):
    def __init__(self):
        super(DiscLoss, self).__init__()
        self.mpd_loss_fn = MPDLoss()
        self.msd_loss_fn = MSDLoss()

    def forward(self, msd_gt, mpd_gt, msd_pred, mpd_pred, **kwargs):
        mpd_loss = sum(
            self.mpd_loss_fn(pred_mpd, label_mpd)
            for pred_mpd, label_mpd in zip(mpd_pred[-1], mpd_pred[-1])
        )
        
        msd_loss = sum(
            self.msd_loss_fn(label_msd, pred_msd)
            for label_msd, pred_msd in zip(msd_gt[-1], msd_gt[-1])
        )
        
        return {"d_loss": mpd_loss + msd_loss}

    

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        return sum(F.l1_loss(p, l) for pred, label in zip(preds, labels) for p, l in zip(pred, label))


class GenLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.mel_spectrogram = MelSpectrogram(MelSpectrogramConfig()).to(self.device)
        self.feature_matching_loss = FeatureMatchingLoss()

    def forward(
        self,
        msd_gt,
        mpd_gt,
        msd_pred,
        mpd_pred,
        output_audio,
        spectrogram,
        **batch,
    ):
        # Ensure spectrogram is on the same device
        spectrogram = spectrogram.to(self.device)

        # Compute Mel-spectrogram of the generated audio
        mel_output = self.mel_spectrogram(output_audio)

        # Mel-spectrogram loss
        mel_loss = F.l1_loss(mel_output, spectrogram)

        # Adversarial loss
        adversarial_loss = 0
        for preds in msd_pred[-1]:
            adversarial_loss += torch.mean((preds - 1) ** 2)
        for preds in mpd_pred[-1]:
            adversarial_loss += torch.mean((preds - 1) ** 2)

        # Feature matching loss
        feature_loss = self.feature_matching_loss(msd_pred[:-1], msd_gt[:-1]) + \
                       self.feature_matching_loss(mpd_pred[:-1], mpd_gt[:-1])

        # Total generator loss
        loss = adversarial_loss + 2 * feature_loss + 45 * mel_loss
        return {"loss": loss}

