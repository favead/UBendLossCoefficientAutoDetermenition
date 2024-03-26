import logging
from typing import Union
import warnings
from modAL import ActiveLearner, CommitteeRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from torch import nn


class ALTrainer:
    def __init__(
        self,
        model: Union[
            GaussianProcessRegressor,
            list[GaussianProcessRegressor],
            GradientBoostingRegressor,
            nn.Module,
        ],
        query_strategy: callable,
        model_type: str,
        n_start_points: int,
        n_query: int,
        estimation_step: int,
        log: logging.Logger,
    ) -> None:
        self.model_type = model_type
        self.n_start_points = n_start_points
        self.n_query = n_query
        self.estimation_step = estimation_step
        self.log = log
        self.model = model
        self.query_strategy = query_strategy

    def _create_al_model(
        self, train_X: np.ndarray, train_y: np.ndarray, n_start_points: int
    ) -> None:
        if self.model_type == "ActiveLearner":
            self._create_solo_al_model(train_X, train_y, n_start_points)
        elif self.model_type == "CommitteeRegressor":
            self._create_qbc_model(train_X, train_y, n_start_points)

    def _create_solo_al_model(
        self, train_X: np.ndarray, train_y: np.ndarray, n_start_points: int
    ) -> None:
        initial_X_i = np.random.choice(
            train_X.shape[0], size=n_start_points, replace=False
        )
        initial_X = train_X[initial_X_i]
        initial_y = train_y[initial_X_i].reshape(-1, 1)

        self.al_model = ActiveLearner(
            estimator=self.model,
            query_strategy=self.query_strategy,
            X_training=initial_X,
            y_training=initial_y,
        )
        return None

    def _create_qbc_model(
        self, train_X: np.ndarray, train_y: np.ndarray, n_start_points: int
    ) -> CommitteeRegressor:
        n_models = len(self.model)
        initial_pool = np.random.choice(
            range(train_X.shape[0]),
            size=n_models * n_start_points,
            replace=False,
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
            for idx, regressor in zip(initial_idx, self.model)
        ]

        self.al_model = CommitteeRegressor(
            learner_list=learner_list,
            query_strategy=self.query_strategy,
        )
        return None

    def train(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
    ) -> tuple[ActiveLearner, pd.DataFrame]:

        estimation_info = {
            "n_points": [],
            "r2": [],
            "mse": [],
            "mae": [],
            "mape": [],
        }

        self._create_al_model(train_X, train_y, self.n_start_points)

        warnings.filterwarnings("ignore")
        for i in range(self.n_query):
            self.log.info(f"Start {i + 1} / {self.n_query} iteration")
            query_idx, _ = self.al_model.query(train_X)
            self.al_model.teach(
                train_X[query_idx].reshape(1, -1),
                train_y[query_idx].reshape(1, -1),
            )

            if (i + 1) % self.estimation_step == 0:
                val_preds = self.al_model.predict(val_X)
                estimation_info["n_points"].append(self.n_start_points + i + 1)
                estimation_info["r2"].append(r2_score(val_y, val_preds))
                estimation_info["mape"].append(
                    mean_absolute_percentage_error(val_y, val_preds)
                )
                estimation_info["mse"].append(
                    mean_squared_error(val_y, val_preds)
                )
                estimation_info["mae"].append(
                    mean_absolute_error(val_y, val_preds)
                )

        estimation_info = pd.DataFrame.from_dict(estimation_info)
        return self.al_model, estimation_info
