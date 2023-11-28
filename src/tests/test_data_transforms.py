"""
Test data processing and separation
"""
import os
from hydra import initialize, compose
import numpy as np
import pandas as pd

from ubend_pl.scripts.data_transforms import run_transforms, create_features


def test_data_pipeline() -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config", overrides=["data=test"])
        run_transforms(cfg)
        normalized_data = pd.read_csv(
            cfg["data"]["pipeline"]["normalize_data"]["data_output_path"]
        )
        assert all(normalized_data.mean() <= 1.0)

        normalized_data_split_path = cfg["data"]["pipeline"]["split_data"][
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
            <= (
                1
                - cfg["data"]["pipeline"]["split_data"]["test_size"]
                - cfg["data"]["pipeline"]["split_data"]["val_size"]
            )
            * normalized_data.shape[0]
        )

        assert val_data.shape[0] > 0
        assert (
            val_data.shape[0]
            <= cfg["data"]["pipeline"]["split_data"]["val_size"] * normalized_data.shape[0]
        )

        assert test_data.shape[0] > 0
        assert (
            test_data.shape[0]
            <= cfg["data"]["pipeline"]["split_data"]["test_size"] * normalized_data.shape[0]
        )

    return None


def test_feature_engineering() -> None:
    with initialize(version_base=None, config_path="../../configs"):
        # cfg = compose(config_name="config", overrides=["data=test"])
        input_data_path = "data/raw/test.csv"
        output_data_path = "data/processed/feature_test.csv"
        fake_data = {
            "float_col": list(np.random.rand(10)),
            "cat_col": ["a"] * 10,
        }
        fake_data = pd.DataFrame.from_dict(fake_data)
        fake_data.to_csv(input_data_path, index=False)
        create_features(input_data_path, output_data_path)
        feature_fake_data = pd.read_csv(output_data_path)
        assert len(feature_fake_data.columns) == 7
        assert "cos_float_col" in feature_fake_data
        assert "cos_cat_col" not in feature_fake_data
        return None
