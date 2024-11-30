from pathlib import Path

import pandas as pd
import torch
import numpy as np
import random

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer



class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            self.disc_optimizer.zero_grad()
            batch.update(self.model.disc_forward(batch["pred"].detach(), batch["gt"]))
            disc_loss = self.criterion.disc(**batch)
            batch.update(disc_loss)
            batch["disc_loss"].backward()
            self._clip_grad_norm(self.model.mpds)
            self._clip_grad_norm(self.model.msd)
            self.disc_optimizer.step()
            self.train_metrics.update("MPDs grad_norm", self.get_grad_norm(self.model.mpds))
            self.train_metrics.update("MSD grad_norm", self.get_grad_norm(self.model.msd))

            batch.update(self.model.disc_forward(**batch))
            self.gen_optimizer.zero_grad()
            gen_loss = self.criterion.gen(**batch)
            batch.update(gen_loss)
            batch["gen_loss"].backward()
            self._clip_grad_norm(self.model.gen)
            self.gen_optimizer.step()
            self.train_metrics.update("Gen grad_norm", self.get_grad_norm(self.model.gen))


        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

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
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(self, preds, gts, examples_to_log=10, **batch):
        rows = {}
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            if i >= examples_to_log:
                break
            pred_audio = self.writer.wandb.Audio(pred.cpu().squeeze().numpy(), sample_rate=16000)
            gt_audio = self.writer.wandb.Audio(gt.cpu().squeeze().numpy(), sample_rate=16000)
            
            rows[i] = {
                "pred": pred_audio,
                "gt": gt_audio
            }

        self.writer.add_table("logs", pd.DataFrame.from_dict(rows, orient="index"))
