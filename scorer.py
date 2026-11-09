"""서버 CandidateScorer 재현 — 후보 점수 계산.

서버: gdghost-server/src/main/java/com/gdghost/routing/CandidateScorer.java
설정: config/matching.yaml (서버 application.yaml의 matching 블록과 같은 형식)

서버와 결과를 끝자리까지 맞추기 위해 따르는 것:
- 벡터는 float32로 들고, 곱은 float32 · 합은 float64로 앞에서부터 더한다 (Java `dot += a[i] * b[i]`)
- 반올림은 Java Math.round (0.5는 위로). Python round는 짝수 쪽으로 가므로 쓰지 않는다
- 최소 점수는 소수 4자리로 반올림한 relevance와 비교한다
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "matching.yaml"

REASON_SIMILAR_HISTORY = "비슷한 질문에 답한 적 있음"
REASON_SIMILAR_PROFILE = "프로필이 질문과 비슷함"


# ---------------------------------------------------------------- 설정 (MatchingConfig)

@dataclass(frozen=True)
class Weights:
    rule: float
    profile: float
    history: float


@dataclass(frozen=True)
class Rule:
    experience: float
    interest: float
    department: float
    position: float
    senior: float


@dataclass(frozen=True)
class History:
    adopted_bonus: float
    reason_min: float


@dataclass(frozen=True)
class Activity:
    answer_count_cap: int
    per_answer: float
    recent_days: int
    recent_bonus: float
    pass_penalty: float


@dataclass(frozen=True)
class MatchingConfig:
    version: str
    weights: Weights
    rule: Rule
    history: History
    activity: Activity
    min_score: float
    max_candidates: int
    max_per_question: int
    direct_timeout_hours: int
    use_questioner_rules_on_anonymous: bool
    category_experience: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "MatchingConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["matching"]
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "MatchingConfig":
        """Spring처럼 kebab-case 키(min-score)를 snake_case 필드(min_score)로 받는다"""
        r = _snake(raw)
        return cls(
            version=str(r["version"]),
            weights=Weights(**_snake(r["weights"])),
            rule=Rule(**_snake(r["rule"])),
            history=History(**_snake(r["history"])),
            activity=Activity(**_snake(r["activity"])),
            min_score=r["min_score"],
            max_candidates=r["max_candidates"],
            max_per_question=r["max_per_question"],
            direct_timeout_hours=r["direct_timeout_hours"],
            use_questioner_rules_on_anonymous=r["use_questioner_rules_on_anonymous"],
            category_experience={k: list(v) for k, v in r["category_experience"].items()},
        )


def _snake(d: dict) -> dict:
    return {k.replace("-", "_"): v for k, v in d.items()}


# ---------------------------------------------------------------- 입력 · 결과

@dataclass(frozen=True)
class Questioner:
    department: str | None
    position: str
    generation: int


@dataclass(frozen=True)
class QuestionInput:
    """익명 질문이면 questioner는 None"""
    category: str
    vector: np.ndarray | None
    questioner: Questioner | None


@dataclass(frozen=True)
class PastAnswer:
    vector: np.ndarray
    adopted: bool


@dataclass(frozen=True)
class Candidate:
    member_id: int | str
    department: str | None
    position: str
    generation: int
    status: str
    experiences: list[str]
    vector: np.ndarray | None
    past_answers: list[PastAnswer] = field(default_factory=list)
    answer_count: int = 0
    pass_count: int = 0
    last_seen_at: datetime | None = None
    matched_interest: str | None = None


@dataclass(frozen=True)
class Scored:
    member_id: int | str
    rule: float
    profile: float
    history: float
    activity: float
    relevance: float
    total: float
    reasons: list[str]


# ---------------------------------------------------------------- 계산

class CandidateScorer:

    def __init__(self, config: MatchingConfig):
        self.config = config

    def rank(self, question: QuestionInput, candidates: list[Candidate], now: datetime) -> list[Scored]:
        """activity는 순위에만 쓴다 — 활동이 많다는 이유만으로 무관한 질문의 후보가 되지 않게"""
        scored = [self.score(question, c, now) for c in candidates]
        passed = [s for s in scored if s.relevance >= self.config.min_score]
        passed.sort(key=lambda s: (-s.total, s.member_id))
        return passed[: self.config.max_candidates]

    def score(self, question: QuestionInput, candidate: Candidate, now: datetime) -> Scored:
        reasons: list[str] = []
        rule = self._rule(question, candidate, reasons)
        profile = 0.0 if question.vector is None or candidate.vector is None \
            else max(0.0, cosine(question.vector, candidate.vector))
        history = self._history(question, candidate)
        if history >= self.config.history.reason_min:
            reasons.append(REASON_SIMILAR_HISTORY)
        if not reasons and profile > 0:
            reasons.append(REASON_SIMILAR_PROFILE)
        activity = self._activity(candidate, now)
        w = self.config.weights
        relevance = w.rule * rule + w.profile * profile + w.history * history
        return Scored(candidate.member_id, java_round(rule), java_round(profile), java_round(history),
                      java_round(activity), java_round(relevance), java_round(relevance + activity), reasons)

    def _rule(self, question: QuestionInput, candidate: Candidate, reasons: list[str]) -> float:
        rule = 0.0
        r = self.config.rule
        wanted = self.config.category_experience.get(question.category, [])
        matched = [e for e in candidate.experiences if e in wanted]
        if matched:
            rule += r.experience
            reasons.append("경험: " + matched[0])
        # 선배 찾기에서 고른 분야
        if candidate.matched_interest is not None:
            rule += r.interest
            reasons.append(candidate.matched_interest)
        q = question.questioner
        if q is None:
            return rule
        if q.department is not None and q.department == candidate.department:
            rule += r.department
            reasons.append("같은 학과")
        if q.position == candidate.position:
            rule += r.position
            reasons.append("같은 포지션")
        # 기수는 먼저 들어온 쪽이 작다
        if candidate.status == "GRADUATED" or candidate.generation < q.generation:
            rule += r.senior
        return min(1.0, rule)

    def _history(self, question: QuestionInput, candidate: Candidate) -> float:
        if question.vector is None:
            return 0.0
        best = 0.0
        for past in candidate.past_answers:
            similarity = cosine(question.vector, past.vector) + (self.config.history.adopted_bonus if past.adopted else 0)
            best = max(best, similarity)
        return min(1.0, best)

    def _activity(self, candidate: Candidate, now: datetime) -> float:
        a = self.config.activity
        score = min(candidate.answer_count, a.answer_count_cap) * a.per_answer - candidate.pass_count * a.pass_penalty
        recent = candidate.last_seen_at is not None and candidate.last_seen_at > now - timedelta(days=a.recent_days)
        return score + a.recent_bonus if recent else score


def as_vector(values) -> np.ndarray:
    """서버는 벡터를 float[]로 읽는다"""
    return np.asarray(values, dtype=np.float32)


def cosine(a, b) -> float:
    a, b = as_vector(a), as_vector(b)
    # Java: double += float * float — 곱은 float32, 합은 float64로 앞에서부터
    dot = _sequential_sum(a * b)
    norm_a = _sequential_sum(a * a)
    norm_b = _sequential_sum(b * b)
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _sequential_sum(products: np.ndarray) -> float:
    # np.sum은 쌍으로 묶어 더하고, Python 3.12+ sum()은 오차 보정을 해서 Java와 끝자리가 다를 수 있다
    return float(np.cumsum(products.astype(np.float64))[-1]) if products.size else 0.0


def java_round(value: float) -> float:
    """Java: Math.round(value * 10_000) / 10_000.0"""
    return java_math_round(value * 10_000) / 10_000.0


def java_math_round(x: float) -> int:
    """Java Math.round(double) — 가장 가까운 정수, 딱 절반이면 +∞ 쪽으로.

    floor(x + 0.5)는 x = 0.49999999999999994에서 덧셈 오차로 1이 되므로 쓰지 않는다 (Java는 0).
    """
    floor = math.floor(x)
    return floor + 1 if x - floor >= 0.5 else floor
