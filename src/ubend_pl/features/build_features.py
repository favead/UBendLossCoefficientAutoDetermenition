import hydra
from omegaconf import DictConfig, OmegaConf

from ubend_pl.configs import FeaturesConfig


@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def build_features(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    features_config = FeaturesConfig(**cfg.get("features", {}))
    return None


if __name__ == "__main__":
    build_features()
