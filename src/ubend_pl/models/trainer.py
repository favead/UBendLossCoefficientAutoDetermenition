import logging
from typing import Union
import warnings
from modAL import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

from ubend_pl.configs import GPrConfig, GPrQBCConfig
from ubend_pl.models.model_list import (
    GP_regression,
    GP_regression_committee,
    GS_x,
    GS_xy,
    GS_y,
    random_strategy,
)


class GPRTrainer:
    def __init__(
        self,
        model_type: str,
        query_strategy_type: str,
        n_start_points: int,
        n_query: int,
        estimation_step: int,
        log: logging.Logger,
    ) -> None:
        self.model_type = model_type
        self.query_strategy_type = query_strategy_type
        self.n_start_points = n_start_points
        self.n_query = n_query
        self.estimation_step = estimation_step
        self.log = log

    def _initialize_gpr(self, gpr_config: GPrConfig) -> None:
        self.model = GP_regression(
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
        return None

    def _initialize_gpr_list(self, gpr_qbc_config: GPrQBCConfig) -> None:
        self.model = GP_regression_committee(gpr_qbc_config.regressors_params)

    def _create_al_model(
        self, train_X: np.ndarray, train_y: np.ndarray, n_start_points: int
    ) -> None:
        if self.model_type == "ActiveLearner":
            self._create_solo_al_model(train_X, train_y, n_start_points)
        elif self.model_type == "CommitteeRegressor":
            self._create_qbc_model()

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

    def initialize(self, params: Union[GPrConfig, GPrQBCConfig]) -> None:
        if self.model_type == "ActiveLearner":
            self._initialize_gpr(params)
        elif self.model_type == "CommitteeRegressor":
            self._initialize_gpr_list(**params)

        if self.query_strategy_type == "uncertainity":
            self.query_strategy = max_std_sampling
        elif self.query_strategy_type == "gs_x":
            self.query_strategy = GS_x
        elif self.query_strategy_type == "gs_y":
            self.query_strategy = GS_y
        elif self.query_strategy_type == "gs_xy":
            self.query_strategy = GS_xy
        elif self.query_strategy_type == "random":
            self.query_strategy = random_strategy
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
