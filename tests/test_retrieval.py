"""Retrieval ordering and filtering at the Chroma result boundary."""

import pytest

from rag.retrieval import rank_results


@pytest.mark.parametrize("disable_rerank, first_id", [(False, "policy"), (True, "other")])
def test_keyword_bonus_changes_order_without_changing_cosine_scores(disable_rerank, first_id):
    result = {
        "documents": [["refund policy", "unrelated words"]],
        "metadatas": [[{"doc_id": "policy"}, {"doc_id": "other"}]],
        "distances": [[0.6, 0.55]],
        "ids": [["policy-chunk", "other-chunk"]],
    }

    ranked = rank_results(result, "refund policy", 2, None, disable_rerank=disable_rerank)

    assert ranked[0]["doc_id"] == first_id
    assert {row["chunk_id"] for row in ranked} == {"policy-chunk", "other-chunk"}
    assert next(row for row in ranked if row["doc_id"] == "policy")["score"] == pytest.approx(0.4)
    # The policy has a larger rerank score but fails the raw cosine threshold.
    filtered = rank_results(result, "refund policy", 2, 0.425, disable_rerank=disable_rerank)
    assert [row["doc_id"] for row in filtered] == ["other"]


def test_equal_scores_keep_input_order_and_missing_metadata_uses_chroma_id():
    result = {
        "documents": [[None, "second"]],
        "metadatas": [[None, {}]],
        "distances": [[None, 1.0]],
        "ids": [["first", "second"]],
    }

    ranked = rank_results(result, "refund", 1, None)

    assert len(ranked) == 1
    assert ranked[0]["chunk_id"] == "first"
    assert ranked[0]["text"] == ""
    assert ranked[0]["metadata"] == {}
    assert ranked[0]["distance"] is None
    assert ranked[0]["score"] == ranked[0]["rerank_score"] == 0
