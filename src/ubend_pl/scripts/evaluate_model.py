"""
Model evaluation and metrics collection
"""

from clearml import Task
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def run_evaluation(cfg: DictConfig) -> None:
    """
    Model evaluation pipeline
    """
    OmegaConf.to_yaml(cfg)
    evaluate_params = cfg["evaluate"]
    task = Task.init(
        project_name=evaluate_params["project_name"],
        task_name=evaluate_params["task_name"],
        tags=evaluate_params["tags"],
    )
    task.close()
    return None
