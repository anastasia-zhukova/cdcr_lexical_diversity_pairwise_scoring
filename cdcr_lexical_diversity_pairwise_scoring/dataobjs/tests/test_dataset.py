from cdcr_lexical_diversity_pairwise_scoring.dataobjs.dataset import filter_and_update_mention_attributes
from cdcr_lexical_diversity_pairwise_scoring.dataobjs.topics import ScopeConfig, Topics


def _raw_mention(mention_id: str, topic_id: str, subtopic_id: str, coref_chain: str) -> dict:
    return {
        "mention_id": mention_id,
        "topic": "t",
        "topic_id": topic_id,
        "subtopic_id": subtopic_id,
        "coref_chain": coref_chain,
    }


def test_ids_are_namespaced_with_the_dataset_name() -> None:
    mentions, mention_ids = filter_and_update_mention_attributes([_raw_mention("m1", "12", "12ecb", "7")], "ECBplus")

    assert mention_ids == ["m1"]
    assert mentions[0]["dataset"] == "ECBplus"
    assert mentions[0]["topic_id"] == "ECBplus_12"
    assert mentions[0]["subtopic_id"] == "ECBplus_12ecb"
    # the chain keeps the raw topic id, as before
    assert mentions[0]["coref_chain"] == "ECBplus_12_7"


def test_datasets_sharing_subtopic_ids_get_separate_topics() -> None:
    ecb, _ = filter_and_update_mention_attributes([_raw_mention("m1", "12", "12ecb", "7")], "ECBplus")
    metam, _ = filter_and_update_mention_attributes([_raw_mention("m2", "12", "12ecb", "7")], "ECBplusMETAm")

    topics = Topics()
    topics.create_from_mention_list(ecb + metam, ScopeConfig.subtopic)

    assert set(topics.topics_dict) == {"ECBplus_12ecb", "ECBplusMETAm_12ecb"}
    assert topics.topics_to_datasets == {"ECBplus_12ecb": "ECBplus", "ECBplusMETAm_12ecb": "ECBplusMETAm"}
    assert [m.mention_id for m in topics.topics_dict["ECBplus_12ecb"].mentions] == ["m1"]
