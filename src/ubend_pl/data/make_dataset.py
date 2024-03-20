import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split

from ubend_pl.configs import DataConfig


@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def make_dataset(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    data_config = DataConfig(**cfg.get("data", {}))
    normalize_features(
        data_config.raw_data_path,
        data_config.params_ranges,
        data_config.normalized_data_path,
    )
    split_data(
        data_config.normalized_data_path,
        data_config.train_data_path,
        data_config.val_data_path,
        data_config.val_size,
    )
    return None


def normalize_features(
    raw_data_path: str,
    params_ranges: dict[str, list[float]],
    normalized_data_path: str,
) -> None:
    df = pd.read_csv(raw_data_path)

    for param, value in params_ranges.items():
        vmin, vmax = value
        mean = (vmax + vmin) / 2.0
        std = vmax - mean
        df.loc[:, param] = (df[param].values - mean) / std

    df.to_csv(normalized_data_path, index=False)
    return None


def split_data(
    raw_data_path: str,
    train_data_path: str,
    val_data_path: str,
    val_size: float,
    random_state: int = 1234,
) -> None:
    df = pd.read_csv(raw_data_path)
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=random_state
    )
    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)
    return None


if __name__ == "__main__":
    make_dataset()
