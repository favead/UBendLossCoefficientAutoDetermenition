"""
Data processing and separation
"""
import os
from pathlib import Path
from typing import Dict, List

from clearml import Task
import hydra
from joblib import dump, load
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessingSteps:
    """
    Container for data processing steps
    """

    @staticmethod
    def split_data(
        data_input_path: str,
        test_size: float,
        val_size: float,
        features_output_path: str,
        targets_output_path: str,
        target_column: str,
        clearml_task: Task | None = None,
    ) -> None:
        """
        Split data to train/val/test features and targets
        """
        data = pd.read_csv(data_input_path)

        def create_output_path(original_path: str, prefix: str) -> str:
            fname = os.path.basename(original_path).split(".")[0]
            return str(Path(original_path.replace(fname, f"{prefix}_{fname}")))

        def separate_and_save_data(
            data: pd.DataFrame,
            target_column: str,
            features_output_path: str,
            targets_output_path: str,
        ) -> None:
            data.drop(target_column, axis=1).to_csv(
                features_output_path, index=False
            )
            data[target_column].to_csv(targets_output_path, index=False)
            return None

        if test_size > 0:
            data, test_data = train_test_split(
                data,
                test_size=test_size,
                random_state=1234,
                shuffle=True,
            )
            separate_and_save_data(
                test_data,
                target_column,
                create_output_path(features_output_path, "test"),
                create_output_path(targets_output_path, "test"),
            )
            if clearml_task:
                clearml_task.upload_artifact(
                    "test_data", artifact_object=test_data.describe()
                )

        if val_size > 0:
            data, val_data = train_test_split(
                data,
                test_size=val_size,
                random_state=1234,
                shuffle=True,
            )

            separate_and_save_data(
                val_data,
                target_column,
                create_output_path(features_output_path, "val"),
                create_output_path(targets_output_path, "val"),
            )

            if clearml_task:
                clearml_task.upload_artifact(
                    "val_data", artifact_object=val_data.describe()
                )

        separate_and_save_data(
            data,
            target_column,
            create_output_path(features_output_path, "train"),
            create_output_path(targets_output_path, "train"),
        )

        if clearml_task:
            clearml_task.upload_artifact(
                "train_data", artifact_object=data.describe()
            )

        return None

    @staticmethod
    def normalize_data(
        data_input_path: str,
        data_output_path: str,
        scaler_path: str | None = None,
        save_scaler_path: str | None = None,
        clearml_task: Task | None = None,
    ) -> None:
        """
        Normalize features
        """

        data = pd.read_csv(data_input_path)
        scaler = load(scaler_path) if scaler_path else StandardScaler()
        normalized_data = (
            scaler.transform(data)
            if scaler_path
            else scaler.fit_transform(data)
        )
        data = pd.DataFrame(
            normalized_data, index=data.index, columns=data.columns
        )

        if save_scaler_path:
            dump(scaler, save_scaler_path)

        if clearml_task:
            clearml_task.upload_artifact(
                name="normalized_data", artifact_object=data.describe()
            )

        data.to_csv(data_output_path, index=False)
        return None

    @staticmethod
    def create_features(
        data_input_path: str,
        data_output_path: str,
        clearml_task: Task | None = None,
    ) -> None:
        """
        Hand crafted features
        """

        data = pd.read_csv(data_input_path)
        for column in data.columns:
            if data[column].dtype != object:
                data[f"{column}**2"] = data[column].values ** 2
                data[f"{column}**3"] = data[column].values ** 3
                data[f"exp_{column}"] = np.exp(data[column].values)
                data[f"sin_{column}"] = np.sin(data[column].values)
                data[f"cos_{column}"] = np.cos(data[column].values)

        if clearml_task:
            clearml_task.upload_artifact(
                name="features_data", artifact_object=data.describe()
            )
        data.to_csv(data_output_path, index=False)
        return None


@hydra.main(
    config_path="../../../configs", config_name="config", version_base=None
)
def run_transforms(cfg: DictConfig) -> None:
    """
    Data processing pipeline
    """
    OmegaConf.to_yaml(cfg)
    np.random.seed(1234)
    data_pipeline: List[Dict[str, dict]] = cfg["data"]["pipeline"]
    data_pipeline_params = cfg["data"]
    task = Task.init(
        project_name=data_pipeline_params["project_name"],
        task_name=data_pipeline_params["task_name"],
        tags=data_pipeline_params["tags"],
    )
    for processing_step in data_pipeline:
        for processing_step_name, params in processing_step.items():
            if processing_step := getattr(
                DataProcessingSteps, processing_step_name
            ):
                processing_step(**params, clearml_task=task)
    task.close()
    return None
