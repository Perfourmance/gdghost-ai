"""서버 CandidateScorerTest 8개를 그대로 옮긴 것. 설정 · 입력 · 기대값은 Java 테스트와 같다."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from scorer import (Activity, Candidate, CandidateScorer, History, MatchingConfig, PastAnswer, QuestionInput,
                    Questioner, Rule, Weights, cosine, java_math_round, java_round)

NOW = datetime(2026, 10, 31, 3, 0, tzinfo=timezone.utc)


def vector(a: float, b: float) -> np.ndarray:
    return np.array([a, a, b, b], dtype=np.float32)


SAME = vector(1, 1)
OPPOSITE = vector(-1, -1)
ORTHOGONAL = vector(1, -1)

CONFIG = MatchingConfig(
    version="v1", weights=Weights(0.4, 0.3, 0.3), rule=Rule(0.6, 0.3, 0.2, 0.1, 0.1), history=History(0.1, 0.6),
    activity=Activity(10, 0.01, 30, 0.05, 0.05), min_score=0.15, max_candidates=3, max_per_question=5,
    direct_timeout_hours=24, use_questioner_rules_on_anonymous=False,
    category_experience={"JOB": ["인턴", "해커톤"]},
)
scorer = CandidateScorer(CONFIG)


def candidate(member_id: int, experiences: list[str], vec) -> Candidate:
    return Candidate(member_id, "컴퓨터공학과", "DEV", 3, "ENROLLED", experiences, vec, [], 0, 0, None)


def question(vec, questioner: Questioner | None) -> QuestionInput:
    return QuestionInput("JOB", vec, questioner)


def test_항목별_점수와_합계를_설계_공식대로_계산한다():
    c = Candidate(1, "컴퓨터공학과", "DEV", 3, "ENROLLED", ["인턴"], SAME,
                  [PastAnswer(ORTHOGONAL, True)], 20, 1, NOW - timedelta(days=1))

    s = scorer.score(question(SAME, None), c, NOW)

    assert s.rule == 0.6
    assert s.profile == 1.0
    assert s.history == 0.1      # 유사도 0 + 채택 가산
    assert s.activity == 0.1     # 상한 10 × 0.01 + 최근 0.05 − 넘김 0.05
    assert s.relevance == 0.57
    assert s.total == 0.67
    assert s.reasons == ["경험: 인턴"]


def test_질문자_규칙은_학과_포지션_선배를_더하고_상한은_1이다():
    s = scorer.score(question(SAME, Questioner("컴퓨터공학과", "DEV", 5)), candidate(1, ["인턴"], SAME), NOW)

    assert s.rule == 1.0
    assert s.reasons == ["경험: 인턴", "같은 학과", "같은 포지션"]


def test_익명_질문이면_질문자_규칙과_사유가_빠진다():
    s = scorer.score(question(SAME, None), candidate(1, [], SAME), NOW)

    assert s.rule == 0
    assert "같은 학과" not in s.reasons and "같은 포지션" not in s.reasons


def test_최소_점수는_활동_점수를_빼고_판단한다():
    active = Candidate(1, None, "DEV", 3, "ENROLLED", [], ORTHOGONAL, [], 100, 0, NOW)

    assert scorer.score(question(SAME, None), active, NOW).total == 0.15
    assert scorer.rank(question(SAME, None), [active], NOW) == []


def test_태그가_안_맞아도_유사도가_0점5면_후보가_되고_더_낮으면_빠진다():
    base = np.array([1, 0, 0, 0], dtype=np.float32)
    close = np.array([1, 1.7, 0, 0], dtype=np.float32)  # 코사인 약 0.51
    far = np.array([1, 3, 0, 0], dtype=np.float32)      # 코사인 약 0.32

    assert len(scorer.rank(question(base, None), [candidate(1, [], close)], NOW)) == 1
    assert scorer.rank(question(base, None), [candidate(2, [], far)], NOW) == []


def test_벡터가_없으면_규칙_점수만으로_후보가_된다():
    ranked = scorer.rank(question(None, None), [candidate(1, ["인턴"], SAME), candidate(2, [], SAME)], NOW)

    assert [s.member_id for s in ranked] == [1]
    assert ranked[0].profile == 0


def test_합계_순으로_최대_3명만_고른다():
    candidates = [candidate(1, [], ORTHOGONAL), candidate(2, ["인턴"], SAME), candidate(3, ["인턴"], ORTHOGONAL),
                  candidate(4, [], SAME), candidate(5, ["해커톤"], SAME)]

    assert [s.member_id for s in scorer.rank(question(SAME, None), candidates, NOW)] == [2, 5, 4]


def test_답변_이력은_가장_비슷한_것을_쓰고_비슷하면_사유를_단다():
    c = Candidate(1, None, "DEV", 3, "ENROLLED", [], None,
                  [PastAnswer(OPPOSITE, False), PastAnswer(SAME, False)], 2, 0, None)

    s = scorer.score(question(SAME, None), c, NOW)

    assert s.history == 1.0
    assert s.reasons == ["비슷한 질문에 답한 적 있음"]


# --- 서버 테스트에 없는 것: Python으로 옮기며 달라질 수 있는 부분

@pytest.mark.parametrize("x, expected", [
    (2.5, 3),                    # 절반은 위로 (Python round(2.5)는 2)
    (3.5, 4),
    (-2.5, -2),                  # 음수 절반도 +∞ 쪽
    (-2.6, -3),
    (0.49999999999999994, 0),    # floor(x + 0.5)는 1이 되는 값
    (1234.4999, 1234),
])
def test_정수_반올림은_Java_Math_round와_같다(x, expected):
    assert java_math_round(x) == expected


@pytest.mark.parametrize("value, expected", [
    (0.00025, 0.0003),           # × 10000 = 정확히 2.5 → Java는 위로, Python round는 0.0002
    (0.15005, 0.1501),           # 컷 근처의 정확한 절반 — Python round면 0.15
    (0.123449, 0.1234),
    (0.123451, 0.1235),
    (0.1499999, 0.15),
    (0.57, 0.57),
])
def test_소수_4자리_반올림(value, expected):
    assert java_round(value) == expected


def test_코사인은_방향만_보고_길이는_무시한다():
    assert cosine([1, 0], [5, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 3]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


def test_익명이_아닌_질문의_선배_규칙은_졸업이거나_기수가_작을_때():
    junior_q = Questioner("다른학과", "AI", 3)
    same_gen = Candidate(1, "x", "DEV", 3, "ENROLLED", [], None)
    graduated = Candidate(2, "x", "DEV", 4, "GRADUATED", [], None)
    earlier = Candidate(3, "x", "DEV", 2, "ENROLLED", [], None)

    assert scorer.score(question(None, junior_q), same_gen, NOW).rule == 0
    assert scorer.score(question(None, junior_q), graduated, NOW).rule == 0.1
    assert scorer.score(question(None, junior_q), earlier, NOW).rule == 0.1
