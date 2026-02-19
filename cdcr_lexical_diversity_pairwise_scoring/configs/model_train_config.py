from dataclasses import dataclass

from cdcr_lexical_diversity_pairwise_scoring.configs import DataConfig


@dataclass
class ModelTrainConfig:
    data_config: DataConfig
    batch_size: int
    learning_rate: float
    negative_positive_ratio: int
    training_iterations: int
    use_cuda: bool
    fine_tune: bool
    weight_decay: float
    hidden_size: int
