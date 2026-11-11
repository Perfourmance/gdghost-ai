"""평가셋으로 매칭 품질을 잰다 (#6 기준선, #7 · #8 실험에서 재사용).

1. 부원 프로필 · 질문을 서버와 같은 글로 조립해 임베딩한다 (cache/에 저장, 두 번째부터 API 0회)
2. 서버와 같은 공식(scorer.py)으로 질문마다 보낼 선배 3명을 뽑는다
3. data/labels.csv 정답표와 비교해 지표를 낸다

출시 직후를 가정한다: 답변 이력 · 답변 수 · 넘기기 · 최근 접속이 없어 history · activity는 0.
후보는 평가셋 부원 전체 (질문자는 부원 목록에 없다).

    python evaluate.py                                   # 기준선 → results/baseline-detail.md · .json
    python evaluate.py --config config/x.yaml --name x   # 다른 설정으로
    python evaluate.py --export-java cache/java-input.txt   # 서버 Java와 대조할 입력
"""

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

from scorer import DEFAULT_CONFIG, Candidate, CandidateScorer, MatchingConfig, QuestionInput, Questioner, Scored
from text_builder import Interest, MemberSource, QuestionSource, member_text, question_text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)  # 출시일. activity가 0이라 결과에는 영향 없음

# 일부러 맞는 선배가 없게 쓴 질문 · 부원 프로필을 보고 쓴 질문 (docs/dataset.md)
NO_MATCH_BY_DESIGN = {"Q31", "Q32"}
WRITTEN_AFTER_PROFILES = {"Q31", "Q32", "Q33", "Q34", "Q35"}


# ---------------------------------------------------------------- 데이터

@dataclass(frozen=True)
class Dataset:
    members: list[dict]
    questions: list[dict]
    labels: dict[str, dict[str, int]]   # 질문 id → 부원 id → 2 · 1 (없으면 0)
    tags: dict


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_labels(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            q = labels.setdefault(row["question_id"], {})
            for key, score in (("partial", 1), ("correct", 2)):
                for member_id in filter(None, row[key].split(";")):
                    q[member_id] = score
    return labels


def load_dataset(data: Path = DATA) -> Dataset:
    return Dataset(read_jsonl(data / "members.jsonl"), read_jsonl(data / "questions.jsonl"),
                   read_labels(data / "labels.csv"), json.loads((data / "tags.json").read_text(encoding="utf-8")))


def member_source(member: dict, tags: dict) -> MemberSource:
    """서버는 태그를 고른 순서가 아니라 태그 목록 순서(sort_order)로 넘긴다 (MemberProfileReader.tags).
    tags.json은 시드 SQL 순서 그대로이고, 시드는 sort_order 순으로 적혀 있다."""
    interest_order = list(tags["interests"])
    interests = sorted(member["interests"], key=lambda i: interest_order.index(i["name"]))
    experiences = sorted(member["experiences"], key=tags["experiences"].index)
    return MemberSource(
        [Interest(i["name"], sorted(i["skills"], key=tags["interests"][i["name"]].index)) for i in interests],
        experiences, member["status_note"], member["bio"])


def question_source(question: dict) -> QuestionSource:
    return QuestionSource(question["category"], question["title"], question["body"])


# ---------------------------------------------------------------- 점수

@dataclass(frozen=True)
class QuestionResult:
    id: str
    category: str
    anonymous: bool
    sent: list[Scored]          # 실제로 알림이 갈 사람 (컷 통과 + 상위 N명)
    ranked: list[Scored]        # 컷 없이 전체 순위 (원인 분석용)


def candidates(ds: Dataset, member_vectors: dict[str, np.ndarray]) -> list[Candidate]:
    # member_id를 "M01" 문자열로 둬도 서버의 숫자 id 정렬과 순서가 같다 (두 자리로 맞춤)
    return [Candidate(m["id"], m["department"], m["position"], m["generation"], m["status"], m["experiences"],
                      member_vectors[m["id"]]) for m in ds.members]


def question_input(question: dict, vector: np.ndarray) -> QuestionInput:
    asker = question["asker"]
    questioner = None if question["anonymous"] else \
        Questioner(asker["department"], asker["position"], asker["generation"])
    return QuestionInput(question["category"], vector, questioner)


def run(ds: Dataset, config: MatchingConfig, member_vectors: dict[str, np.ndarray],
        question_vectors: dict[str, np.ndarray]) -> list[QuestionResult]:
    scorer = CandidateScorer(config)
    pool = candidates(ds, member_vectors)
    results = []
    for q in ds.questions:
        qi = question_input(q, question_vectors[q["id"]])
        ranked = sorted((scorer.score(qi, c, NOW) for c in pool), key=lambda s: (-s.total, s.member_id))
        results.append(QuestionResult(q["id"], q["category"], q["anonymous"], scorer.rank(qi, pool, NOW), ranked))
    return results


# ---------------------------------------------------------------- 지표

def label_of(ds: Dataset, question_id: str, member_id: str) -> int:
    return ds.labels.get(question_id, {}).get(member_id, 0)


def hit_rate(ds: Dataset, results: list[QuestionResult], minimum: int, use_cut: bool = True) -> tuple[int, int]:
    """minimum점 이상인 선배가 있는 질문 중, 보낸 3명 안에 그런 선배가 있는 질문 수"""
    hits = total = 0
    for r in results:
        if not any(score >= minimum for score in ds.labels.get(r.id, {}).values()):
            continue
        total += 1
        picked = r.sent if use_cut else r.ranked[:3]
        hits += any(label_of(ds, r.id, s.member_id) >= minimum for s in picked)
    return hits, total


def random_hit_rate(ds: Dataset, results: list[QuestionResult], minimum: int, k: int = 3) -> float:
    """무작위로 k명을 뽑았을 때 Hit@k의 기댓값 — 정답 선배가 많은 질문은 아무나 뽑아도 맞는다"""
    n = len(ds.members)
    rates = []
    for r in results:
        good = sum(score >= minimum for score in ds.labels.get(r.id, {}).values())
        if good:
            rates.append(1 - comb(n - good, k) / comb(n, k))
    return sum(rates) / len(rates) if rates else 0.0


def sent_label_counts(ds: Dataset, results: list[QuestionResult]) -> dict[int, int]:
    counts = {2: 0, 1: 0, 0: 0}
    for r in results:
        for s in r.sent:
            counts[label_of(ds, r.id, s.member_id)] += 1
    return counts


def pair_values(ds: Dataset, results: list[QuestionResult], field: str) -> dict[int, list[float]]:
    """라벨별 짝의 값 (profile = 임베딩 유사도, relevance = 컷과 비교하는 점수)"""
    values: dict[int, list[float]] = {2: [], 1: [], 0: []}
    for r in results:
        for s in r.ranked:
            values[label_of(ds, r.id, s.member_id)].append(getattr(s, field))
    return values


def best_correct_rank(ds: Dataset, r: QuestionResult) -> int | None:
    """컷 없이 매긴 전체 순위에서 정답(2) 선배 중 가장 높은 순위 (1부터)"""
    for rank, s in enumerate(r.ranked, start=1):
        if label_of(ds, r.id, s.member_id) == 2:
            return rank
    return None


def summarize(ds: Dataset, config: MatchingConfig, results: list[QuestionResult]) -> dict:
    def ratio(pair: tuple[int, int]) -> dict:
        hits, total = pair
        return {"hits": hits, "total": total, "rate": hits / total if total else None}

    def subset(ids: set[str], keep: bool) -> list[QuestionResult]:
        return [r for r in results if (r.id in ids) == keep]

    relevance = pair_values(ds, results, "relevance")
    profile = pair_values(ds, results, "profile")
    sent = sent_label_counts(ds, results)
    sent_total = sum(sent.values())
    return {
        "config": config.version,
        "members": len(ds.members),
        "questions": len(results),
        "hit_strict": ratio(hit_rate(ds, results, 2)),
        "hit_lenient": ratio(hit_rate(ds, results, 1)),
        "hit_strict_no_cut": ratio(hit_rate(ds, results, 2, use_cut=False)),
        "random_hit_strict": random_hit_rate(ds, results, 2),
        "random_hit_lenient": random_hit_rate(ds, results, 1),
        "hit_strict_named": ratio(hit_rate(ds, [r for r in results if not r.anonymous], 2)),
        "hit_strict_anonymous": ratio(hit_rate(ds, [r for r in results if r.anonymous], 2)),
        "hit_strict_written_before": ratio(hit_rate(ds, subset(WRITTEN_AFTER_PROFILES, False), 2)),
        "hit_strict_written_after": ratio(hit_rate(ds, subset(WRITTEN_AFTER_PROFILES, True), 2)),
        "sent_total": sent_total,
        "sent_by_label": sent,
        "precision_sent": sent[2] / sent_total if sent_total else None,
        "irrelevant_sent_rate": sent[0] / sent_total if sent_total else None,
        "cut_pass_by_label": {label: sum(v >= config.min_score for v in values) / len(values) if values else None
                              for label, values in relevance.items()},
        "pairs_by_label": {label: len(values) for label, values in relevance.items()},
        "profile_by_label": {label: describe(values) for label, values in profile.items()},
        "empty_questions": [r.id for r in results if not r.sent],
        "no_match_sent": {qid: len(r.sent) for r in results for qid in [r.id] if qid in NO_MATCH_BY_DESIGN},
        "per_question": [
            {"id": r.id, "category": r.category, "anonymous": r.anonymous,
             "sent": [{"member": s.member_id, "label": label_of(ds, r.id, s.member_id), "relevance": s.relevance,
                       "rule": s.rule, "profile": s.profile, "reasons": s.reasons} for s in r.sent],
             "best_correct_rank": best_correct_rank(ds, r),
             "correct": sorted(m for m, v in ds.labels.get(r.id, {}).items() if v == 2)}
            for r in results],
    }


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {"n": len(values), "min": min(values), "p25": q[0], "median": q[1], "p75": q[2], "max": max(values),
            "mean": statistics.fmean(values)}


# ---------------------------------------------------------------- 리포트

MARK = {2: "✅", 1: "△", 0: "✗"}


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def frac(d: dict) -> str:
    return f"{pct(d['rate'])} ({d['hits']}/{d['total']})"


def histogram(values_by_label: dict[int, list[float]], low: float = 0.25, high: float = 0.80,
              step: float = 0.05) -> list[str]:
    bins = int(round((high - low) / step))
    lines = ["```", "유사도        정답(2)            애매(1)            무관(0)"]
    for b in range(bins + 1):
        lo = low + b * step
        hi = lo + step

        def count(values):
            if b == 0:
                return sum(v < hi for v in values)
            if b == bins:
                return sum(v >= lo for v in values)
            return sum(lo <= v < hi for v in values)

        label = f"<{hi:.2f}" if b == 0 else (f"≥{lo:.2f}" if b == bins else f"{lo:.2f}–{hi:.2f}")
        cells = []
        for key in (2, 1, 0):
            n = count(values_by_label[key])
            total = len(values_by_label[key]) or 1
            cells.append(f"{'█' * round(n / total * 40):<14}{n:>4}")
        lines.append(f"{label:<11} " + " ".join(cells))
    lines.append("```")
    return lines


def report(ds: Dataset, config: MatchingConfig, results: list[QuestionResult], summary: dict) -> str:
    s = summary
    categories = ds.tags["categories"]
    lines = [f"# 평가 상세 — 설정 `{config.version}`", "",
             f"`python evaluate.py`가 만든 파일. 부원 {s['members']}명 · 질문 {s['questions']}개 · "
             f"컷 {config.min_score} · 최대 {config.max_candidates}명. history · activity = 0 (출시 직후 가정).", "",
             "## 핵심 지표", "",
             "| 지표 | 값 | 비교 |", "|---|---|---|",
             f"| Hit@3 엄격 (정답 2가 보낸 3명 안에) | **{frac(s['hit_strict'])}** | 무작위 {pct(s['random_hit_strict'])} |",
             f"| Hit@3 느슨 (2 또는 1) | {frac(s['hit_lenient'])} | 무작위 {pct(s['random_hit_lenient'])} |",
             f"| Hit@3 엄격, 컷 없이 상위 3명 | {frac(s['hit_strict_no_cut'])} | 컷 때문에 놓친 질문 확인용 |",
             f"| 보낸 알림 중 정답(2) 비율 | {pct(s['precision_sent'])} | 무작위 약 {pct(avg_correct_share(ds))} |",
             f"| 보낸 알림 중 무관(0) 비율 | **{pct(s['irrelevant_sent_rate'])}** | 받으면 넘기기 할 알림 |",
             f"| 무관(0) 짝의 컷 통과율 | **{pct(s['cut_pass_by_label'][0])}** | 목표(제안) ≤ 10% |",
             f"| 후보 0명 질문 | {len(s['empty_questions'])}개 {', '.join(s['empty_questions'])} | |",
             "",
             f"보낸 알림 {s['sent_total']}건: 정답 {s['sent_by_label'][2]} · 애매 {s['sent_by_label'][1]} · "
             f"무관 {s['sent_by_label'][0]}", "",
             "## 나눠 보기 (Hit@3 엄격)", "",
             "| 묶음 | 값 |", "|---|---|",
             f"| 실명 질문 | {frac(s['hit_strict_named'])} |",
             f"| 익명 질문 (질문자 규칙 없음) | {frac(s['hit_strict_anonymous'])} |",
             f"| 프로필 보기 전에 쓴 질문 (Q01–Q30) | {frac(s['hit_strict_written_before'])} |",
             f"| 프로필 보고 쓴 질문 (Q31–Q35) | {frac(s['hit_strict_written_after'])} |",
             f"| 맞는 선배 없게 쓴 질문 | " + ", ".join(f"{k}: {v}명에게 보냄" for k, v in s['no_match_sent'].items())
             + " (0명이 정상) |", "",
             "## 라벨별 컷 통과율", "",
             "| 라벨 | 짝 수 | relevance ≥ 컷 |", "|---|---|---|"]
    for label, name in ((2, "정답"), (1, "애매"), (0, "무관")):
        lines.append(f"| {name}({label}) | {s['pairs_by_label'][label]} | {pct(s['cut_pass_by_label'][label])} |")
    lines += ["", "## 라벨별 임베딩 유사도 (profile)", "",
              "| 라벨 | 최소 | 25% | 중앙 | 75% | 최대 | 평균 |", "|---|---|---|---|---|---|---|"]
    for label, name in ((2, "정답"), (1, "애매"), (0, "무관")):
        d = s["profile_by_label"][label]
        if d:
            lines.append(f"| {name}({label}) | {d['min']:.3f} | {d['p25']:.3f} | {d['median']:.3f} | {d['p75']:.3f} | "
                         f"{d['max']:.3f} | {d['mean']:.3f} |")
    lines += [""] + histogram(pair_values(ds, results, "profile")) + [""]
    lines += ["## 질문별", "",
              "✅ 정답 · △ 애매 · ✗ 무관. 괄호는 relevance(규칙 · 유사도). '정답 최고 순위'는 컷 없이 40명 중 순위.", "",
              "| 질문 | 분류 | 보낸 선배 | 정답 최고 순위 | 정답 선배 |", "|---|---|---|---|---|"]
    for q in s["per_question"]:
        sent = " · ".join(f"{MARK[x['label']]}{x['member']} {x['relevance']:.3f} ({x['rule']:.1f}·{x['profile']:.3f})"
                          for x in q["sent"]) or "없음"
        rank = q["best_correct_rank"] or "-"
        anon = " 익명" if q["anonymous"] else ""
        lines.append(f"| {q['id']} | {categories[q['category']]}{anon} | {sent} | {rank} | "
                     f"{', '.join(q['correct']) or '-'} |")
    return "\n".join(lines) + "\n"


def avg_correct_share(ds: Dataset) -> float:
    """무작위로 보냈을 때 받은 사람이 정답일 확률 (질문 평균)"""
    n = len(ds.members)
    shares = [sum(v == 2 for v in ds.labels.get(q["id"], {}).values()) / n for q in ds.questions]
    return sum(shares) / len(shares)


# ---------------------------------------------------------------- 서버 대조용 내보내기

def export_java(ds: Dataset, member_vectors: dict, question_vectors: dict, path: Path) -> None:
    """한 줄에 하나: Q|id|분류|익명|학과|포지션|기수|벡터  ·  M|번호|학과|포지션|기수|상태|경험;…|벡터
    float32 값을 그대로 적는다 (Float.parseFloat로 같은 값이 된다)"""
    def vec(v):
        return ",".join(repr(float(x)) for x in np.asarray(v, dtype=np.float32))

    lines = []
    for q in ds.questions:
        a = q["asker"]
        lines.append("|".join(["Q", q["id"], q["category"], str(q["anonymous"]).lower(), a["department"],
                               a["position"], str(a["generation"]), vec(question_vectors[q["id"]])]))
    for m in ds.members:
        lines.append("|".join(["M", m["id"][1:], m["department"], m["position"], str(m["generation"]), m["status"],
                               ";".join(m["experiences"]), vec(member_vectors[m["id"]])]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def export_scores(results: list[QuestionResult], path: Path) -> None:
    """질문 · 부원마다 rule · profile · relevance, 그리고 보낸 순서. Java 출력과 그대로 비교한다"""
    lines = []
    for r in results:
        for s in sorted(r.ranked, key=lambda x: x.member_id):
            lines.append(f"{r.id} {int(s.member_id[1:])} {s.rule} {s.profile} {s.relevance}")
        lines.append(f"{r.id} SENT " + ",".join(str(int(s.member_id[1:])) for s in r.sent))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- 실행

def embed_all(ds: Dataset, build_member=member_text, build_question=question_text):
    from embed import Embedder

    member_texts = {m["id"]: build_member(member_source(m, ds.tags)) for m in ds.members}
    question_texts = {q["id"]: build_question(question_source(q)) for q in ds.questions}
    embedder = Embedder.from_env()
    texts = list(member_texts.values()) + list(question_texts.values())
    vectors = dict(zip(texts, embedder.embed(texts)))
    stats = embedder.stats
    print(f"임베딩: 글 {len(set(texts))}개 · 캐시 {stats.cache_hits} · 새로 {stats.embedded} · API 호출 {stats.api_calls}")
    return ({k: vectors[t] for k, t in member_texts.items()}, {k: vectors[t] for k, t in question_texts.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--name", default="baseline", help="결과 파일 이름 results/<name>-detail.md · .json")
    parser.add_argument("--export-java", help="서버 Java 대조용 입력 파일 경로 (점수는 같은 이름 .scores로)")
    args = parser.parse_args()

    ds = load_dataset()
    config = MatchingConfig.load(args.config)
    member_vectors, question_vectors = embed_all(ds)
    results = run(ds, config, member_vectors, question_vectors)
    summary = summarize(ds, config, results)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.name}-detail.md").write_text(report(ds, config, results, summary), encoding="utf-8",
                                                    newline="\n")
    (RESULTS / f"{args.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
    if args.export_java:
        path = Path(args.export_java)
        export_java(ds, member_vectors, question_vectors, path)
        export_scores(results, path.with_suffix(".scores"))
    print(f"Hit@3 엄격 {frac(summary['hit_strict'])} (무작위 {pct(summary['random_hit_strict'])}) · "
          f"무관 짝 컷 통과 {pct(summary['cut_pass_by_label'][0])} → results/{args.name}-detail.md")


if __name__ == "__main__":
    main()
