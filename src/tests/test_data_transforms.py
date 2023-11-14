"""
Test data processing and separation
"""
import os
from hydra import initialize, compose
import pandas as pd

from ubend_pl.scripts.data_transforms import run_transforms


def test_data_pipeline() -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config", overrides=["data=test"])
        run_transforms(cfg)
        normalized_data = pd.read_csv(
            cfg["data"]["normalize_data"]["data_output_path"]
        )
        assert all(normalized_data.mean() <= 1.0)

        normalized_data_split_path = cfg["data"]["split_data"][
            "features_output_path"
        ]
        normalized_data_fname = os.path.basename(
            normalized_data_split_path
        ).split(".")[0]

        train_data = pd.read_csv(
            normalized_data_split_path.replace(
                normalized_data_fname, f"train_{normalized_data_fname}"
            )
        )

        val_data = pd.read_csv(
            normalized_data_split_path.replace(
                normalized_data_fname, f"val_{normalized_data_fname}"
            )
        )

        test_data = pd.read_csv(
            normalized_data_split_path.replace(
                normalized_data_fname, f"test_{normalized_data_fname}"
            )
        )

        assert train_data.shape[0] > 0
        assert (
            train_data.shape[0]
            <= (1 - cfg["data"]["split_data"]["test_size"] - cfg["data"]["split_data"]["val_size"])
            * normalized_data.shape[0]
        )

        assert val_data.shape[0] > 0
        assert (
            val_data.shape[0]
            <= cfg["data"]["split_data"]["val_size"] * normalized_data.shape[0]
        )

        assert test_data.shape[0] > 0
        assert (
            test_data.shape[0]
            <= cfg["data"]["split_data"]["test_size"] * normalized_data.shape[0]
        )

    return None
