"""exp_weights.py의 빠른 계산이 evaluate.py(서버와 같은 계산)와 같은 결과를 내는지.

임베딩 캐시가 있어야 돌아간다 (없으면 건너뜀 — API를 부르지 않는다).
"""

import pytest

import evaluate
from exp_weights import Setting, build_pairs, evaluate_setting
from scorer import MatchingConfig


def cached_vectors(ds):
    from embed import Embedder

    embedder = Embedder("offline", "offline")  # 캐시만 읽는다. 캐시에 없으면 부르기 전에 건너뜀
    members = {m["id"]: evaluate.member_text(evaluate.member_source(m, ds.tags)) for m in ds.members}
    questions = {q["id"]: evaluate.question_text(evaluate.question_source(q)) for q in ds.questions}
    if any(embedder._read_cache(t) is None for t in [*members.values(), *questions.values()]):
        pytest.skip("임베딩 캐시가 없음 (python evaluate.py를 먼저 실행)")
    return ({k: embedder.embed([t])[0] for k, t in members.items()},
            {k: embedder.embed([t])[0] for k, t in questions.items()})


@pytest.mark.parametrize("path", ["config/matching.yaml", "config/exp-weights.yaml"])
def test_빠른_계산은_evaluate와_같다(path):
    ds = evaluate.load_dataset()
    member_vectors, question_vectors = cached_vectors(ds)
    config = MatchingConfig.load(evaluate.ROOT / path)
    exact = evaluate.summarize(ds, config, evaluate.run(ds, config, member_vectors, question_vectors))
    fast = evaluate_setting(build_pairs(ds, config, member_vectors, question_vectors), Setting.of(config))
    assert fast["hit"] == pytest.approx(exact["hit_strict"]["rate"])
    assert fast["hit_lenient"] == pytest.approx(exact["hit_lenient"]["rate"])
    assert fast["sent"] == exact["sent_total"]
    assert fast["irrelevant"] == pytest.approx(exact["irrelevant_sent_rate"])
    assert fast["no_match_sent"] == sum(exact["no_match_sent"].values())
