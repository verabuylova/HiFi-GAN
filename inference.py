import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. 

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    generator = instantiate(config.generator).to(device)
    discriminator = instantiate(config.discriminator).to(device) 
    print(generator)

    if config.inferencer.get("from_pretrained"):
        checkpoint = torch.load(config.inferencer.from_pretrained, map_location=device)
        generator.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint from {config.inferencer.from_pretrained}")

    metrics = instantiate(config.metrics) 

    save_path = ROOT_PATH / "data" / "generated_audio" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        generator=generator,
        discriminator=discriminator, 
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=True,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
