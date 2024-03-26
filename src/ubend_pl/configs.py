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


@dataclass
class ALTrainConfig:
    artifact_dir: str
    model_dir: str
    train_data_path: str
    val_data_path: str
    query_strategy_type: str
    model_type: str
    model_name: str
    n_start_points: int
    n_query: int
    estimation_step: int


@dataclass
class VisualizationConfig:
    artifact_dir: str
    estimation_info_filename: str
