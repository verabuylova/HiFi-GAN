from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms import MelSpectrogram, MelSpectrogramConfig

class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

        outputs = self.generator(**batch)
        batch.update(outputs)

        discriminator_outputs = self.discriminator(**batch, detach_generated=True)
        batch.update(discriminator_outputs)

        all_discriminator_losses = self.d_criterion(**batch)
        batch.update(all_discriminator_losses)

        if self.is_train:
            batch["d_loss"].backward()
            self._clip_grad_norm()
            self.d_optimizer.step()
            if self.d_lr_scheduler is not None:
                self.d_lr_scheduler.step()

        discriminator_outputs = self.discriminator(**batch, detach_generated=False)
        batch.update(discriminator_outputs)

        all_losses = self.g_criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.g_optimizer.step()
            if self.g_lr_scheduler is not None:
                self.g_lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        
        return batch


    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, output_audio, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        output_spectrogram = MelSpectrogram(MelSpectrogramConfig())(
            output_audio.detach().cpu()
        )[0]

        self.writer.add_image("gt", plot_spectrogram(spectrogram_for_plot))
        self.writer.add_image("pred", plot_spectrogram(output_spectrogram))

    def log_audio(self, audio, audio_name):

        audio = (audio / torch.max(torch.abs(audio))).detach().cpu()
        self.writer.add_audio(
            audio_name,
            audio.float(),
            sample_rate=self.config.writer.audio_sample_rate,
        )

    def log_predictions(self, output_audio, audio, examples_to_log=1, **batch):
        for i, (pred, gt) in enumerate(zip(output_audio, audio)):
            self.log_audio(pred, f"pred_{i}")
            self.log_audio(gt, f"gt_{i}")

