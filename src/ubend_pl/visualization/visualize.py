import glob
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from ubend_pl.configs import VisualizationConfig


@hydra.main(
    version_base=None, config_name="config", config_path="../../../configs"
)
def visualize(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    visualization_config = VisualizationConfig(cfg.get("visualization"))

    for artifact_folder in glob.glob(visualization_config.artifact_dir):
        pass

    return None


if __name__ == "__main__":
    visualize()
