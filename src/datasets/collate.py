import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    wavs = [item["wav"] for item in dataset_items]
    specs = [item["spec"] for item in dataset_items]

    wavs_padded = pad_sequence(wavs, batch_first=True, padding_value=0.0)
    specs_padded = pad_sequence(specs, batch_first=True, padding_value=0.0)

    return {
        "wav": wavs_padded,
        "spec": specs_padded,
    }
