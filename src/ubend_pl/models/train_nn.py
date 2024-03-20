import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from ubend_pl.configs import NnTrainConfig
from ubend_pl.models.model_list import PtLossRegressor


log = logging.getLogger("main")


@hydra.main(
    version_base=None, config_name="config", config_path="../../../configs"
)
def run_train(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    model = PtLossRegressor(**cfg.get("model"))
    nn_train_config = NnTrainConfig(**cfg.get("train"))

    if os.path.exists(nn_train_config.artifact_dir):
        os.remove(nn_train_config.artifact_dir)
        os.remove(nn_train_config.model_dir)
        log.warning("Модель уже обучена. Перезаписываю...")
        return None

    os.makedirs(nn_train_config.artifact_dir)
    os.makedirs(nn_train_config.model_dir)

    train()

    return None


def train() -> None:
    return None


if __name__ == "__main__":
    run_train()
