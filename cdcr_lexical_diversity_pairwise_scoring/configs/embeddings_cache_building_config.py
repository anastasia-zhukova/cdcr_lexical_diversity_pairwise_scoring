from dataclasses import dataclass

from cdcr_lexical_diversity_pairwise_scoring.configs import DataConfig


@dataclass
class EmbeddingsCacheBuildingConfig:
    data_config: DataConfig
    use_cuda: bool
    max_surrounding_context: int
