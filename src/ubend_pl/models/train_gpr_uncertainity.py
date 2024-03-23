import logging
import os
from pathlib import Path
import shutil
from typing import Union
import warnings

import hydra
from joblib import dump
from modAL import ActiveLearner
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

from ubend_pl.configs import GPrTrainConfig, GPrConfig
from ubend_pl.models.model_list import GP_regression, GP_regression_std


log = logging.getLogger("main")


@hydra.main(
    version_base=None, config_name="config", config_path="../../../configs"
)
def run_train_gpr_uncertainity(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    gpr_config = GPrConfig(**cfg.get("models"))
    gpr_train_config = GPrTrainConfig(**cfg.get("train"))

    gpr = GP_regression(
        n_features=gpr_config.n_features,
        c_min=gpr_config.c_min,
        c_max=gpr_config.c_max,
        nu=gpr_config.nu,
        ck_const=gpr_config.ck_const,
        ck_min=gpr_config.ck_min,
        ck_max=gpr_config.ck_max,
        whtk_constant=gpr_config.whtk_constant,
        whtk_min=gpr_config.whtk_min,
        whtk_max=gpr_config.whtk_max,
        n_restarts_optimizer=gpr_config.n_restarts_optimizer,
    )

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

    al_gpr, estimation_info = train_gpr_uncertainity(
        gpr,
        train_X,
        train_y,
        val_X,
        val_y,
        gpr_train_config.n_start_points,
        gpr_train_config.n_query,
        gpr_train_config.estimation_step,
    )

    estimation_info_path = str(
        Path(gpr_train_config.artifact_dir, "estimation_info.csv")
    )
    al_gpr_path = str(Path(gpr_train_config.model_dir, "al_gpr.joblib"))
    estimation_info.to_csv(estimation_info_path, index=False)
    dump(al_gpr, al_gpr_path)
    return None


def train_gpr_uncertainity(
    gpr: GaussianProcessRegressor,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    n_start_points: int,
    n_query: int,
    estimation_step: int,
) -> Union[ActiveLearner, pd.DataFrame]:
    estimation_info = {
        "n_points": [],
        "r2": [],
        "mse": [],
        "mae": [],
        "mape": [],
    }

    initial_X_i = np.random.choice(
        train_X.shape[0], size=n_start_points, replace=False
    )
    initial_X = train_X[initial_X_i]
    initial_y = train_y[initial_X_i].reshape(-1, 1)

    al_gpr = ActiveLearner(
        estimator=gpr,
        query_strategy=GP_regression_std,
        X_training=initial_X,
        y_training=initial_y,
    )

    warnings.filterwarnings("ignore")
    for i in range(n_query):
        log.info(f"Start {i + 1} / {n_query} iteration")
        query_idx, query_instance = al_gpr.query(train_X)
        al_gpr.teach(
            train_X[query_idx].reshape(1, -1), train_y[query_idx].reshape(1, -1)
        )

        if (i + 1) % estimation_step == 0:
            val_preds = al_gpr.predict(val_X)
            estimation_info["n_points"].append(n_start_points + i + 1)
            estimation_info["r2"].append(r2_score(val_y, val_preds))
            estimation_info["mape"].append(
                mean_absolute_percentage_error(val_y, val_preds)
            )
            estimation_info["mse"].append(mean_squared_error(val_y, val_preds))
            estimation_info["mae"].append(mean_absolute_error(val_y, val_preds))

    estimation_info = pd.DataFrame.from_dict(estimation_info)

    return al_gpr, estimation_info


if __name__ == "__main__":
    run_train_gpr_uncertainity()
