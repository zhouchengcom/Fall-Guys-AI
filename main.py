from omegaconf import DictConfig, OmegaConf

import hydra
import logging
from hydra.core.hydra_config import HydraConfig
import pprint
from hydra.core.global_hydra import GlobalHydra

from utils.augmentation import augmente

import albumentations as A

logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="conf")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    logger.info("hahah")
    open("test.txt", "w")

    if cfg.augmentation.run:
        augmente(cfg)

    # pprint.pprint(HydraConfig.get().job)
    # loaded_transform = A.load("/tmp/transform.json")
    # print([v for v in GlobalHydra.instance().config_loader().get_load_history()])
    # print(dir(cfg.model))


if __name__ == "__main__":
    my_app()