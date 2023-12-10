"""
Model evaluation and metrics collection
"""

import logging
import os
import time
from typing import Dict, Union
from clearml import Task
import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.base import r2_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
import torch


log = logging.getLogger(__name__)


def save_metric_figure(title: str, save_path: str, metric: float) -> None:
    """
    Plot scalar metric to bar and save to artifacts
    """
    plt.xlim(left=-0.3, right=0.3)
    plt.bar_label(
        plt.bar([""], metric, width=0.1, color="green"),
        labels=[metric],
        label_type="center",
        color="white",
        size=16,
    )
    plt.title(title, size=24)
    plt.savefig(save_path)
    return None


def save_hist_figure(
    gts: np.ndarray, preds: np.ndarray, save_path: str
) -> None:
    """
    Plot residuals histogram
    """
    residuals = gts - preds
    plt.hist(residuals, size="green")
    plt.title("Residuals", size=24)
    plt.savefig(save_path)
    return None


def save_scatter_figure(
    gts: np.ndarray, preds: np.ndarray, save_path: str
) -> None:
    """
    Plot pairplot for gts and preds
    """
    plt.scatter(preds, gts)
    plt.ylabel("Predictions")
    plt.xlabel("Ground Truth's")
    plt.title("Pair plot", size=24)
    plt.savefig(save_path)
    return None


def evaluate_model(
    model_params: Dict[str, Union[str, int]],
    data_path: str,
    targets_path: str,
    artifact_name: str,
    clearml_task: Task | None = None,
    **kwargs,
) -> None:
    """
    Model evaluation: plot diagrams with metrics
    """
    model = build_model(**model_params)
    gts = pd.read_csv(targets_path).values

    start_time = time.time()
    preds = model.predict(data_path)
    end_time = time.time()
    inference_time = end_time - start_time

    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()

    mape = mean_absolute_percentage_error(gts, preds)
    mae = mean_absolute_error(gts, preds)
    mse = mean_squared_error(gts, preds)
    r2 = r2_score(gts, preds)

    try:
        os.chdir(f"artifacts/{artifact_name}")
    except OSError:
        log.info("Directory already exist, rewriting metrics!")

    logger = clearml_task.get_logger() if clearml_task else None

    for title, metric_value in zip(
        ["MAPE", "MAE", "MSE", "R2", "Inference_Time"],
        [mape, mae, mse, r2, inference_time],
    ):
        save_metric_figure(
            title, f"artifacts/{artifact_name}/{title}.png", metric_value
        )

        if logger:
            logger.report_media(
                title,
                iteration=0,
                local_path=f"artifacts/{artifact_name}/{title}.png",
                file_extension="png",
            )

    save_hist_figure(gts, preds, f"artifacts/{artifact_name}/Residuals.png")
    save_scatter_figure(gts, preds, f"artifacts/{artifact_name}/Pair_Plot.png")

    if logger:
        logger.report_media(
            "Residuals",
            iteration=0,
            local_path=f"artifacts/{artifact_name}/Residuals.png",
            file_extension="png",
        )
        logger.report_media(
            "Pair_Plot",
            iteration=0,
            local_path=f"artifacts/{artifact_name}/Pair_Plot.png",
            file_extension="png",
        )
    return None


@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def run_evaluation(cfg: DictConfig) -> None:
    """
    Model evaluation initialization and logging
    """
    OmegaConf.to_yaml(cfg)
    evaluate_params = cfg["evaluate"]
    task = Task.init(
        project_name=evaluate_params["project_name"],
        task_name=evaluate_params["task_name"],
        tags=evaluate_params["tags"],
    )
    evaluate_model(
        cfg["model"],
        evaluate_params["data_path"],
        evaluate_params["targets_path"],
        artifact_name=evaluate_params["artifact_name"],
        clearml_task=task,
    )
    task.close()
    return None
