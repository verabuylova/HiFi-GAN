from pathlib import Path
from src.datasets.base_dataset import BaseDataset


class CustomDirTextDataset(BaseDataset):
    def __init__(self, data_dir: str = None, text: str = None, *args, **kwargs):
        data = []
        
        if text:
            entry = {"text": text}
            data.append(entry)
        elif data_dir:
            transcription_path = Path(data_dir) / "transcriptions"
            for path in transcription_path.iterdir():
                if path.suffix.lower() == ".txt":
                    entry = {}
                    entry["path"] = str(path)
                    with path.open("r", encoding="utf-8") as f:
                        entry["text"] = f.read().strip()
                    if entry["text"]:
                        data.append(entry)
        
        super().__init__(data, *args, **kwargs)
