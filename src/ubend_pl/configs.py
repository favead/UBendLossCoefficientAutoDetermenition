from dataclasses import dataclass


@dataclass
class DataConfig:
    raw_data_path: str
    train_data_path: str
    val_data_path: str
    val_size: float


@dataclass
class FeaturesConfig:
    pass
