"""
Data processing and separation
"""
from typing import Dict
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    data_input_path: str,
    test_size: float,
    stratify_column: str,
    val_size: float,
    features_output_path: str,
    targets_output_path: str,
    target_column: str,
) -> None:
    """
    Split data to train/val/test features and targets
    """
    data = pd.read_csv(data_input_path)

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
            stratify=stratify_column,
        )
        separate_and_save_data(
            test_data,
            target_column,
            f"test_{features_output_path}",
            f"test_{targets_output_path}",
        )

    if val_size > 0:
        data, val_data = train_test_split(
            data,
            test_size=val_size,
            random_state=1234,
            shuffle=True,
            stratify=stratify_column,
        )

        separate_and_save_data(
            val_data,
            target_column,
            f"val_{features_output_path}",
            f"val_{targets_output_path}",
        )

    separate_and_save_data(
        data,
        target_column,
        f"train_{features_output_path}",
        f"train_{targets_output_path}",
    )

    return None


def normalize_data(
    data_input_path: str, target_column: str, data_output_path: str
) -> None:
    """
    Normalize features
    """
    data = pd.read_csv(data_input_path)
    features = data.drop(target_column, axis=1)
    features = (features - features.mean(axis=0)) / (
        features.std(axis=0) + 1e-6
    )
    data = pd.concat((features, data[target_column]))
    data.to_csv(data_output_path, index=False)
    return None


PROCESSING_STEPS = {"normalize_data": normalize_data, "split_data": split_data}


@hydra.main(
    config_path="../../configs", config_name="config", version_base=None
)
def run_transforms(cfg: DictConfig) -> None:
    """
    Data processing pipeline
    """
    OmegaConf.to_yaml(cfg)
    np.random.seed(1234)
    data_pipeline: Dict[str, dict] = cfg["data"]
    for processing_step_name, params in data_pipeline.items():
        if processing_step := PROCESSING_STEPS.get(processing_step_name, None):
            processing_step(**params)
    return None
