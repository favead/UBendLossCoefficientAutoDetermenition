from dataclasses import dataclass


@dataclass
class DataConfig:
    raw_data_path: str
    normalized_data_path: str
    train_data_path: str
    val_data_path: str
    val_size: float
    params_ranges: dict[str, list[float, float]]


@dataclass
class NnTrainConfig:
    artifact_dir: str
    model_dir: str
