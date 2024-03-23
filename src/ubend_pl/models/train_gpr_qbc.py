import logging
import os
from pathlib import Path
import shutil
from typing import Union
import warnings

import hydra
from joblib import dump
from modAL import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling
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

from ubend_pl.configs import GPrTrainConfig, GPrQBCConfig
from ubend_pl.models.model_list import GP_regression_committee


log = logging.getLogger("main")


@hydra.main(
    version_base=None, config_name="config", config_path="../../../configs"
)
def run_train_gpr_qbc(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    gpr_qbc_config = GPrQBCConfig(**cfg.get("models"))
    gpr_train_config = GPrTrainConfig(**cfg.get("train"))

    gpr_committee = GP_regression_committee(gpr_qbc_config.regressors_params)

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

    gpr_committee, estimation_info = train_gpr_qbc(
        gpr_committee,
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
    gpr_committee_path = str(
        Path(gpr_train_config.model_dir, "gpr_committee.joblib")
    )
    estimation_info.to_csv(estimation_info_path, index=False)
    dump(gpr_committee, gpr_committee_path)
    return None


def train_gpr_qbc(
    gpr_committee: list[GaussianProcessRegressor],
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    n_start_points: int,
    n_query: int,
    estimation_step: int,
) -> Union[ActiveLearner, pd.DataFrame]:
    n_models = len(gpr_committee)
    estimation_info = {
        "n_points": [],
        "r2": [],
        "mse": [],
        "mae": [],
        "mape": [],
    }

    initial_pool = np.random.choice(
        range(train_X.shape[0]), size=n_models * n_start_points, replace=False
    )
    initial_idx = [
        initial_pool[i * n_start_points : (i + 1) * n_start_points]
        for i in range(n_models)
    ]

    learner_list = [
        ActiveLearner(
            estimator=regressor,
            X_training=train_X[idx],
            y_training=train_y[idx].reshape(-1, 1),
        )
        for idx, regressor in zip(initial_idx, gpr_committee)
    ]

    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling,
    )

    warnings.filterwarnings("ignore")
    for i in range(n_query):
        log.info(f"Start {i + 1} / {n_query} iteration")
        query_idx, query_instance = committee.query(train_X)

        # idk why but need to reshape X
        committee.teach(
            train_X[query_idx].reshape(1, -1), train_y[query_idx].reshape(-1, 1)
        )

        if (i + 1) % estimation_step == 0:
            val_preds = committee.predict(val_X)
            estimation_info["n_points"].append(n_start_points + i + 1)
            estimation_info["r2"].append(r2_score(val_y, val_preds))
            estimation_info["mape"].append(
                mean_absolute_percentage_error(val_y, val_preds)
            )
            estimation_info["mse"].append(mean_squared_error(val_y, val_preds))
            estimation_info["mae"].append(mean_absolute_error(val_y, val_preds))

    estimation_info = pd.DataFrame.from_dict(estimation_info)

    return committee, estimation_info


if __name__ == "__main__":
    run_train_gpr_qbc()
