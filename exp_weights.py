"""#8 가중치 · 컷 실험. 설정 조합 수천 개를 빠르게 돌려 results/exp-weights-sweep.md에 표로 남긴다.

evaluate.py는 설정 하나를 서버와 끝자리까지 같게 계산한다 (조합 하나에 약 0.1초).
여기서는 질문 × 부원 짝마다 유사도와 규칙 항목(경험 · 학과 · 포지션 · 선배)을 한 번만 계산해 두고,
가중치만 바꿔 행렬로 다시 더한다. 같은 설정이면 evaluate.py와 결과가 같다 (tests/test_exp_weights.py).
history · activity는 0, 관심 분야(interest)는 선배 찾기에서만 쓰여 질문 알림에는 영향이 없다.

    python exp_weights.py
"""

import itertools
from dataclasses import dataclass

import numpy as np

from evaluate import NO_MATCH_BY_DESIGN, RESULTS, Dataset, candidates, embed_all, load_dataset, question_input
from scorer import MatchingConfig, cosine


@dataclass(frozen=True)
class Pairs:
    """질문 × 부원 행렬 (행 = 평가셋 질문 순서, 열 = 부원 순서 = id 순서)"""
    question_ids: list[str]
    cos: np.ndarray
    experience: np.ndarray      # 질문 분류에 맞는 경험 태그가 있음
    department: np.ndarray      # 아래 셋은 익명 질문이면 0
    position: np.ndarray
    senior: np.ndarray
    label: np.ndarray


def build_pairs(ds: Dataset, config: MatchingConfig, member_vectors: dict, question_vectors: dict) -> Pairs:
    pool = candidates(ds, member_vectors)
    shape = (len(ds.questions), len(pool))
    cos, exp, dep, pos, sen = (np.zeros(shape) for _ in range(5))
    label = np.zeros(shape, dtype=int)
    for i, q in enumerate(ds.questions):
        qi = question_input(q, question_vectors[q["id"]])
        wanted = config.category_experience.get(q["category"], [])
        for j, c in enumerate(pool):
            cos[i, j] = max(0.0, cosine(qi.vector, c.vector))
            exp[i, j] = any(e in wanted for e in c.experiences)
            label[i, j] = ds.labels.get(q["id"], {}).get(c.member_id, 0)
            if qi.questioner is not None:
                dep[i, j] = qi.questioner.department == c.department
                pos[i, j] = qi.questioner.position == c.position
                sen[i, j] = c.status == "GRADUATED" or c.generation < qi.questioner.generation
    return Pairs([q["id"] for q in ds.questions], cos, exp, dep, pos, sen, label)


@dataclass(frozen=True)
class Setting:
    rule: float
    profile: float
    experience: float
    department: float
    position: float
    senior: float
    min_score: float

    @classmethod
    def of(cls, config: MatchingConfig) -> "Setting":
        r = config.rule
        return cls(config.weights.rule, config.weights.profile, r.experience, r.department, r.position, r.senior,
                   config.min_score)


def relevance(p: Pairs, s: Setting) -> np.ndarray:
    rule = np.minimum(1.0, s.experience * p.experience + s.department * p.department + s.position * p.position
                      + s.senior * p.senior)
    return np.round(s.rule * rule + s.profile * p.cos, 4)


def select(score: np.ndarray, passed: np.ndarray, k: int = 3) -> np.ndarray:
    """질문마다 passed인 짝 중 score 상위 k명. 동점은 부원 id 순 (서버와 같음)"""
    sent = np.zeros(score.shape, dtype=bool)
    for i in range(score.shape[0]):
        order = [j for j in np.argsort(-score[i], kind="stable") if passed[i, j]]
        sent[i, order[:k]] = True
    return sent


def metrics(p: Pairs, sent: np.ndarray) -> dict:
    has_correct = (p.label == 2).any(1)
    has_partial = (p.label >= 1).any(1)
    labels = np.where(sent, p.label, -1)
    n = int(sent.sum())
    no_match = np.array([q in NO_MATCH_BY_DESIGN for q in p.question_ids])
    return {
        "hit": float(((labels == 2).any(1) & has_correct).sum() / has_correct.sum()),
        "hit_lenient": float(((labels >= 1).any(1) & has_partial).sum() / has_partial.sum()),
        "sent": n,
        "irrelevant": float((labels == 0).sum() / n) if n else None,
        "precision": float((labels == 2).sum() / n) if n else None,
        "no_match_sent": int(sent[no_match].sum()),
        "starved": int((has_correct & ~sent.any(1)).sum()),     # 정답 선배가 있는데 아무에게도 안 간 질문
    }


def evaluate_setting(p: Pairs, s: Setting) -> dict:
    rel = relevance(p, s)
    return metrics(p, select(rel, rel >= s.min_score))


def pair_pass(p: Pairs, s: Setting) -> dict[int, float]:
    rel = relevance(p, s)
    return {label: float((rel[p.label == label] >= s.min_score).mean()) for label in (2, 1, 0)}


def relative_cut(p: Pairs, s: Setting, threshold: float) -> dict:
    """서버 코드를 바꿔야 하는 안: 질문마다 '후보 전체 평균 유사도 + threshold' 이상만 보낸다"""
    above_mean = p.cos - p.cos.mean(1, keepdims=True)
    return metrics(p, select(relevance(p, s), above_mean >= threshold))


# ---------------------------------------------------------------- 리포트

WEIGHTS = [(0.4, 0.3), (0.3, 0.4), (0.2, 0.5), (0.1, 0.6), (0.0, 0.7)]
EXPERIENCE = [0.6, 0.3, 0.1, 0.0]
CUTS = [round(0.10 + 0.02 * k, 2) for k in range(16)]


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def row(m: dict) -> str:
    return (f"{pct(m['hit'])} | {pct(m['irrelevant'])} | {pct(m['precision'])} | {m['sent']} | "
            f"{m['no_match_sent']} | {m['starved']}")


HEADER = "Hit@3 엄격 | 보낸 알림 중 무관 | 보낸 알림 중 정답 | 보낸 알림 수 | Q31·Q32에 보낸 수 | 정답 있는데 0명"


def sweep_report(p: Pairs, base: Setting, proposed: Setting) -> str:
    lines = ["# #8 가중치 · 컷 실험 — 전체 표", "",
             "`python exp_weights.py`가 만든 파일. 결론은 [exp-weights.md](exp-weights.md).", "",
             f"평가셋 질문 {len(p.question_ids)}개 · 부원 {p.cos.shape[1]}명. history · activity = 0. "
             "학과 0.2 · 포지션 0.1 · 선배 0.1은 기준선 그대로.", ""]

    lines += ["## 1. 가중치 × 경험 점수 (컷 0.04 — 거의 거르지 않음)", "",
              f"| 규칙 : 유사도 | 경험 | {HEADER} |", "|---|---|" + "---|" * 6]
    for (wr, wp), e in itertools.product(WEIGHTS, EXPERIENCE):
        s = Setting(wr, wp, e, base.department, base.position, base.senior, 0.04)
        lines.append(f"| {wr} : {wp} | {e} | {row(evaluate_setting(p, s))} |")

    for name, s in [("기준선 (0.4 : 0.3)", base), ("추천안 (0.1 : 0.6)", proposed)]:
        lines += ["", f"## 2. 컷을 바꾸면 — {name}", "",
                  f"| min-score | {HEADER} | 짝 컷 통과 (정답 / 애매 / 무관) |", "|---|" + "---|" * 7]
        for ms in CUTS:
            t = Setting(s.rule, s.profile, s.experience, s.department, s.position, s.senior, ms)
            passed = pair_pass(p, t)
            lines.append(f"| {ms:.2f} | {row(evaluate_setting(p, t))} | "
                         f"{pct(passed[2])} / {pct(passed[1])} / {pct(passed[0])} |")

    lines += ["", "## 3. 질문별 상대 컷 (서버 코드 변경 필요) — 추천안 순위", "",
              "질문마다 후보 전체의 평균 유사도를 구하고, 그보다 threshold 이상 높은 선배에게만 보낸다.", "",
              f"| threshold | {HEADER} |", "|---|" + "---|" * 6]
    for t in [round(0.04 + 0.01 * k, 2) for k in range(9)]:
        lines.append(f"| +{t:.2f} | {row(relative_cut(p, proposed, t))} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ds = load_dataset()
    base_config = MatchingConfig.load()
    proposed_config = MatchingConfig.load("config/exp-weights.yaml")
    member_vectors, question_vectors = embed_all(ds)
    pairs = build_pairs(ds, base_config, member_vectors, question_vectors)
    path = RESULTS / "exp-weights-sweep.md"
    path.write_text(sweep_report(pairs, Setting.of(base_config), Setting.of(proposed_config)), encoding="utf-8",
                    newline="\n")
    print(f"→ {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
