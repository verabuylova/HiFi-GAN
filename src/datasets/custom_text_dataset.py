from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirTextDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        data = []
        transcription_dir = Path(data_dir)
        for path in Path(transcription_dir).iterdir():
            entry = {}
            entry["path"] = str(path)
            entry["text"] = None
            entry["audio_len"] = 0
            if path.suffix in [".txt"]:
                with path.open() as f:
                    entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
