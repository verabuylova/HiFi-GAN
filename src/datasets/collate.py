import torch

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
    wavs = torch.stack([item["wav"] for item in dataset_items])
    specs = torch.stack([item["spec"] for item in dataset_items])

    return {"wav": wavs, "spec": specs}
