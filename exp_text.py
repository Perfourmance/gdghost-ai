"""#7 텍스트 조립 방식 비교. 임베딩할 글을 다르게 조립해 #6과 같은 지표로 비교한다.

    python exp_text.py        # → results/exp-text-sweep.md

조립 방식 (서버 DefaultTextBuilder를 옮긴 text_builder.py가 A):
    질문 A  분류명 + 제목 + 본문          "취업·채용 제목\n본문"
    질문 B  분류명 빼기                   "제목\n본문"
    부원 A  라벨: 키워드 목록              "관심 분야: 백엔드(Spring)\n경험: 인턴\n근황: …\n소개: …"
    부원 C  라벨 없이 자연어 문장          "백엔드(Spring)에 관심이 있습니다.\n인턴 경험이 있습니다.\n…"
    부원 D  순서 바꾸기 (소개 → 근황 → 경험 → 관심 분야), 라벨은 그대로

질문 방식 × 부원 방식 조합마다 현재 설정(2026-10-v1)과 #8 추천 설정(2026-11-v2)으로 계산한다.
새 글만 API를 부르고 나머지는 cache/에서 읽는다.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from evaluate import (RESULTS, NO_MATCH_BY_DESIGN, Dataset, best_correct_rank, embed_all, hit_rate, label_of,
                      load_dataset, run, sent_label_counts)
from scorer import MatchingConfig
from text_builder import MemberSource, QuestionSource, _interest, member_text, question_text


# ---------------------------------------------------------------- 조립 방식

def question_without_category(q: QuestionSource) -> str:
    return q.title if q.body is None else q.title + "\n" + q.body


def member_sentences(m: MemberSource) -> str:
    lines = []
    if m.interests:
        lines.append(", ".join(_interest(i) for i in m.interests) + "에 관심이 있습니다.")
    if m.experiences:
        lines.append(", ".join(m.experiences) + " 경험이 있습니다.")
    if m.status_note is not None:
        lines.append(m.status_note)
    if m.bio is not None:
        lines.append(m.bio)
    return "\n".join(lines)


def member_reordered(m: MemberSource) -> str:
    lines = []
    if m.bio is not None:
        lines.append("소개: " + m.bio)
    if m.status_note is not None:
        lines.append("근황: " + m.status_note)
    if m.experiences:
        lines.append("경험: " + ", ".join(m.experiences))
    if m.interests:
        lines.append("관심 분야: " + ", ".join(_interest(i) for i in m.interests))
    return "\n".join(lines)


QUESTION_BUILDERS: dict[str, tuple[str, Callable[[QuestionSource], str]]] = {
    "A": ("분류명 + 제목 + 본문 (지금)", question_text),
    "B": ("분류명 빼기", question_without_category),
}
MEMBER_BUILDERS: dict[str, tuple[str, Callable[[MemberSource], str]]] = {
    "A": ("라벨: 키워드 (지금)", member_text),
    "C": ("라벨 없이 문장", member_sentences),
    "D": ("순서 바꾸기 (소개 먼저)", member_reordered),
}


# ---------------------------------------------------------------- 지표

def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """정답 짝 하나와 무관 짝 하나를 뽑았을 때 정답 쪽 유사도가 더 높을 확률"""
    return float((positive[:, None] > negative[None, :]).mean() + 0.5 * (positive[:, None] == negative[None, :]).mean())


def auc_difference_range(base: tuple[np.ndarray, np.ndarray], other: tuple[np.ndarray, np.ndarray],
                         rounds: int = 1000, seed: int = 0) -> tuple[float, float]:
    """질문을 복원 추출로 다시 뽑아 AUC 차이(other − base)를 rounds번 잰 95% 범위. 0을 포함하면 차이가 잡음 수준"""
    rng = np.random.default_rng(seed)
    n = base[0].shape[0]
    diffs = []
    for _ in range(rounds):
        rows = rng.integers(0, n, n)
        values = [auc(cos[rows][labels[rows] == 2], cos[rows][labels[rows] == 0]) for cos, labels in (other, base)]
        diffs.append(values[0] - values[1])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def similarity_matrix(ds: Dataset, member_vectors: dict, question_vectors: dict) -> tuple[np.ndarray, np.ndarray]:
    members = [m["id"] for m in ds.members]
    cos = np.array([[float(np.dot(question_vectors[q["id"]], member_vectors[m]) /
                           (np.linalg.norm(question_vectors[q["id"]]) * np.linalg.norm(member_vectors[m])))
                     for m in members] for q in ds.questions])
    labels = np.array([[label_of(ds, q["id"], m) for m in members] for q in ds.questions])
    return cos, labels


@dataclass(frozen=True)
class Outcome:
    hit: float
    irrelevant: float
    no_match: int
    ranks: dict[str, int | None]     # 질문별 정답 최고 순위 (컷 없이)


def outcome(ds: Dataset, config: MatchingConfig, member_vectors: dict, question_vectors: dict) -> Outcome:
    results = run(ds, config, member_vectors, question_vectors)
    hits, total = hit_rate(ds, results, 2)
    sent = sent_label_counts(ds, results)
    return Outcome(hits / total, sent[0] / sum(sent.values()),
                   sum(len(r.sent) for r in results if r.id in NO_MATCH_BY_DESIGN),
                   {r.id: best_correct_rank(ds, r) for r in results})


def compare_ranks(base: Outcome, other: Outcome) -> tuple[int, int]:
    """정답 선배의 최고 순위가 좋아진 질문 수 · 나빠진 질문 수"""
    better = worse = 0
    for qid, rank in base.ranks.items():
        if rank is None:
            continue
        better += other.ranks[qid] < rank
        worse += other.ranks[qid] > rank
    return better, worse


# ---------------------------------------------------------------- 실행 · 리포트

def main() -> None:
    ds = load_dataset()
    configs = {"현재": MatchingConfig.load(), "#8 추천": MatchingConfig.load("config/exp-weights.yaml")}
    rows = []
    for (qk, (qname, qb)), (mk, (mname, mb)) in ((q, m) for q in QUESTION_BUILDERS.items()
                                               for m in MEMBER_BUILDERS.items()):
        member_vectors, question_vectors = embed_all(ds, build_member=mb, build_question=qb)
        cos, labels = similarity_matrix(ds, member_vectors, question_vectors)
        relative = cos - cos.mean(1, keepdims=True)
        rows.append({
            "key": f"질문 {qk} · 부원 {mk}", "desc": f"{qname} / {mname}", "matrix": (cos, labels),
            "auc": auc(cos[labels == 2], cos[labels == 0]),
            "auc_relative": auc(relative[labels == 2], relative[labels == 0]),
            "median_gap": float(np.median(cos[labels == 2]) - np.median(cos[labels == 0])),
            "outcomes": {name: outcome(ds, c, member_vectors, question_vectors) for name, c in configs.items()},
        })

    base = rows[0]
    lines = ["# #7 텍스트 조립 방식 비교 — 전체 표", "",
             "`python exp_text.py`가 만든 파일. 결론은 [exp-text.md](exp-text.md).", "",
             f"평가셋 질문 {len(ds.questions)}개 · 부원 {len(ds.members)}명. history · activity = 0.", "",
             "## 1. 유사도가 정답과 무관을 얼마나 가르나", "",
             "AUC = 정답 짝과 무관 짝을 하나씩 뽑았을 때 정답 쪽 유사도가 더 높을 확률 (0.5 = 못 가름, 1 = 완벽). "
             "'질문별'은 질문마다 평균 유사도를 뺀 값으로 잰 것 (#8의 상대 컷). "
             "'지금과 차이 95% 범위'는 질문을 무작위로 다시 뽑아 1,000번 잰 AUC 차이의 범위로, 0을 포함하면 잡음 수준이다.", "",
             "| 조합 | 방식 | AUC | 지금과 차이 95% 범위 | AUC (질문별) | 정답 − 무관 유사도 중앙값 차이 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        low, high = auc_difference_range(base["matrix"], r["matrix"])
        spread = "—" if r is base else f"{low:+.3f} ~ {high:+.3f}"
        lines.append(f"| {r['key']} | {r['desc']} | {r['auc']:.3f} | {spread} | {r['auc_relative']:.3f} | "
                     f"{r['median_gap']:.3f} |")
    for name in configs:
        lines += ["", f"## 2. 뽑은 결과 — 설정 {name}", "",
                  "'좋아짐 / 나빠짐'은 질문 A · 부원 A와 비교해 정답 선배의 최고 순위(컷 없이)가 바뀐 질문 수.", "",
                  "| 조합 | Hit@3 엄격 | 보낸 알림 중 무관 | Q31 · Q32에 보낸 수 | 좋아짐 / 나빠짐 |", "|---|---|---|---|---|"]
        for r in rows:
            o = r["outcomes"][name]
            better, worse = compare_ranks(base["outcomes"][name], o)
            lines.append(f"| {r['key']} | {o.hit:.0%} | {o.irrelevant:.0%} | {o.no_match} | {better} / {worse} |")
    path = RESULTS / "exp-text-sweep.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"→ {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
