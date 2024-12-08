import torch
from wvmos import get_wvmos
from src.metrics.base_metric import BaseMetric

class WV_MOS_Metric(BaseMetric):
    def __init__(self, audio_paths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_wvmos(cuda=False)
        self.audio_paths = audio_paths

    def __call__(self, audio_paths: list[str], **kwargs):
        audio_paths = self.audio_paths
        mos_score = self.model.calculate_dir(audio_paths)
        return mos_score

