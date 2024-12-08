import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)

import torch._dynamo
torch._dynamo.config.suppress_errors = True


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    if config.text is not None:
        config.datasets = {
            "test": {
                "_target_": "src.datasets.BaseDataset",
                "instance_transforms": config.transforms.instance_transforms.inference,
                "segment": 'inference',
                "index": [
                    {"text": config.text, "path": "text.txt", "audio_len": 0}
                ],
            }
        }

        config.dataloader.batch_size = 1

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    generator = instantiate(config.generator).to(device)
    # print(model)

    torch.backends.cudnn.benchmark = True  # optimize slow_dilated_conv2d

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("test", []):
        metrics["inference"].append(instantiate(metric_config))

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        generator=generator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()