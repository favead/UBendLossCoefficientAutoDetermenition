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
class GPrConfig:
    n_features: int
    c_min: float
    c_max: float
    nu: float
    ck_const: float
    ck_min: float
    ck_max: float
    whtk_constant: float
    whtk_min: float
    whtk_max: float
    n_restarts_optimizer: int


@dataclass
class GPrTrainConfig:
    artifact_dir: str
    model_dir: str
    train_data_path: str
    val_data_path: str
    query_strategy_type: str
    n_start_points: int
    n_query: int
    estimation_step: int


@dataclass
class GPrQBCConfig:
    regressors_params: list[GPrConfig]


@dataclass
class VisualizationConfig:
    artifact_dir: str
    estimation_info_filename: str
