import logging
import os
from pathlib import Path
import shutil

import hydra
from joblib import dump
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from ubend_pl.configs import ALTrainConfig
from ubend_pl.models.model_list import create_model, get_query_strategy
from ubend_pl.models.trainer import ALTrainer


log = logging.getLogger("main")


@hydra.main(
    version_base=None, config_name="config", config_path="../../../configs"
)
def run_train_gpr(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    gpr_train_config = ALTrainConfig(**cfg.get("train"))

    if os.path.exists(gpr_train_config.artifact_dir):
        shutil.rmtree(gpr_train_config.artifact_dir)
        shutil.rmtree(gpr_train_config.model_dir)
        log.warning("Модель уже обучена. Перезаписываю...")

    os.makedirs(gpr_train_config.artifact_dir)
    os.makedirs(gpr_train_config.model_dir)

    train_data = pd.read_csv(gpr_train_config.train_data_path)
    val_data = pd.read_csv(gpr_train_config.val_data_path)
    train_X, train_y = train_data.drop("pt_loss", axis=1), train_data["pt_loss"]
    val_X, val_y = val_data.drop("pt_loss", axis=1), val_data["pt_loss"]
    train_X, val_X = train_X.values, val_X.values
    train_y, val_y = train_y.values, val_y.values

    model = create_model(
        model_name=gpr_train_config.model_name, model_config=cfg.get("models")
    )
    query_strategy = get_query_strategy(gpr_train_config.query_strategy_type)

    trainer = ALTrainer(
        model=model,
        query_strategy=query_strategy,
        model_type=gpr_train_config.model_type,
        n_start_points=gpr_train_config.n_start_points,
        n_query=gpr_train_config.n_query,
        estimation_step=gpr_train_config.estimation_step,
        log=log,
    )

    al_gpr, estimation_info = trainer.train(train_X, train_y, val_X, val_y)

    estimation_info_path = str(
        Path(gpr_train_config.artifact_dir, "estimation_info.csv")
    )
    al_gpr_path = str(Path(gpr_train_config.model_dir, "al_gpr.joblib"))
    estimation_info.to_csv(estimation_info_path, index=False)
    dump(al_gpr, al_gpr_path)
    return None


if __name__ == "__main__":
    run_train_gpr()
