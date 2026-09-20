from dataclasses import dataclass, field
from typing import Any, Self

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Config fields worth filtering runs by in the MLflow UI
_TAGGED_CONFIG_FIELDS = ("setting", "type_of_pairs", "train_scope", "language_model")
_LIST_TAG_SEPARATOR = "-"


@dataclass
class RunContext:
    """Everything a tracker needs to open a run: its name, the logged params and the tags.

    `job_name` names the pipeline step attaching to the run (`train`, `inference_clustering`, ...),
    so that each step keeps its own log artifact.
    """

    run_name: str
    job_name: str
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_hydra(cls, config: DictConfig) -> Self:
        """Build the context of the running Hydra job: the run is named after the experiment config."""
        hydra_job = HydraConfig.get().job
        return cls.from_config(config, hydra_job.config_name, hydra_job.name)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        config_name: str,
        job_name: str,
    ) -> Self:
        raw_config = OmegaConf.to_container(config, resolve=True)
        params = {key: cls._to_param_value(value) for key, value in raw_config.items()}

        tags = {key: params[key] for key in _TAGGED_CONFIG_FIELDS if key in params}
        tags["train_datasets"] = cls._to_param_value(raw_config.get("train_dataset_names", []))
        tags["experiment_config"] = config_name

        return cls(run_name=config_name, job_name=job_name, params=params, tags=tags)

    @staticmethod
    def _to_param_value(value: Any) -> str:
        # MLflow stores params as strings, so lists are joined to stay readable in the UI
        if isinstance(value, list):
            return _LIST_TAG_SEPARATOR.join(str(item) for item in value)

        return str(value)
