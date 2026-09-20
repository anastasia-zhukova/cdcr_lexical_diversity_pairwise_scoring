from omegaconf import OmegaConf

from cdcr_lexical_diversity_pairwise_scoring.tracking.run_context import RunContext


def _make_config() -> object:
    return OmegaConf.create(
        {
            "setting": "single",
            "type_of_pairs": "random",
            "ratio": 10,
            "train_scope": "subtopic",
            "train_dataset_names": ["CD2CR"],
            "test_dataset_names": ["CD2CR", "ECBplus"],
            "language_model": "roberta-large",
            "max_pairs_dev": None,
        },
    )


def test_run_is_named_after_the_experiment_config() -> None:
    context = RunContext.from_config(_make_config(), "single-random-cd2cr", "train")

    assert context.run_name == "single-random-cd2cr"
    assert context.job_name == "train"
    assert context.tags["experiment_config"] == "single-random-cd2cr"


def test_params_are_stringified_and_lists_joined() -> None:
    context = RunContext.from_config(_make_config(), "single-random-cd2cr", "train")

    assert context.params["ratio"] == "10"
    assert context.params["max_pairs_dev"] == "None"
    assert context.params["test_dataset_names"] == "CD2CR-ECBplus"


def test_tags_cover_the_filterable_fields() -> None:
    context = RunContext.from_config(_make_config(), "single-random-cd2cr", "train")

    assert context.tags == {
        "setting": "single",
        "type_of_pairs": "random",
        "train_scope": "subtopic",
        "language_model": "roberta-large",
        "train_datasets": "CD2CR",
        "experiment_config": "single-random-cd2cr",
    }
