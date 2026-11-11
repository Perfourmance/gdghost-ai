from math import comb

import pytest

from evaluate import (Dataset, QuestionResult, hit_rate, load_dataset, member_source, random_hit_rate, read_labels,
                      sent_label_counts)
from scorer import Scored
from text_builder import member_text


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def by_id(ds, member_id):
    return next(m for m in ds.members if m["id"] == member_id)


def test_관심_분야는_고른_순서가_아니라_태그_목록_순서다(ds):
    # M08은 안드로이드 → iOS 순으로 적혀 있지만 서버 sort_order는 iOS(4)가 안드로이드(5)보다 앞
    text = member_text(member_source(by_id(ds, "M08"), ds.tags))
    assert text.startswith("관심 분야: iOS(Flutter, 앱 출시·심사), 안드로이드(Kotlin, Jetpack Compose)\n")


def test_세부_기술과_경험도_태그_목록_순서다(ds):
    m04 = member_source(by_id(ds, "M04"), ds.tags)
    assert m04.interests[0].skills == ["Figma", "UI 디자인", "UX 리서치"]
    m02 = member_source(by_id(ds, "M02"), ds.tags)
    assert m02.experiences == ["공모전 수상", "해커톤", "동아리 운영진"]


def test_최소_프로필은_관심_분야_한_줄이다(ds):
    assert member_text(member_source(by_id(ds, "M05"), ds.tags)) == "관심 분야: 백엔드"


def test_라벨은_정답이_애매보다_우선한다(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("question_id,correct,partial,memo\nQ1,M1;M2,M3,\nQ2,,,\n", encoding="utf-8")
    assert read_labels(path) == {"Q1": {"M1": 2, "M2": 2, "M3": 1}, "Q2": {}}


def scored(member_id):
    return Scored(member_id, 0, 0, 0, 0, 0.2, 0.2, [])


def tiny():
    members = [{"id": f"M{i}"} for i in range(1, 6)]
    labels = {"Q1": {"M1": 2, "M2": 1}, "Q2": {"M3": 1}, "Q3": {}}
    ds = Dataset(members, [], labels, {})
    results = [
        QuestionResult("Q1", "JOB", False, [scored("M2"), scored("M4")], [scored("M2"), scored("M4"), scored("M1")]),
        QuestionResult("Q2", "JOB", True, [scored("M3")], [scored("M3")]),
        QuestionResult("Q3", "JOB", False, [scored("M5")], [scored("M5")]),
    ]
    return ds, results


def test_Hit3는_해당_점수_선배가_있는_질문만_센다():
    ds, results = tiny()
    assert hit_rate(ds, results, 2) == (0, 1)                 # Q1만 정답이 있고, 보낸 M2 · M4는 정답 아님
    assert hit_rate(ds, results, 2, use_cut=False) == (1, 1)  # 컷 없이 상위 3명이면 M1이 들어옴
    assert hit_rate(ds, results, 1) == (2, 2)                 # Q1(M2 애매) · Q2(M3 애매)


def test_무작위_기준선은_정답이_많을수록_높다():
    ds, results = tiny()
    n = 5
    assert random_hit_rate(ds, results, 2) == pytest.approx(1 - comb(n - 1, 3) / comb(n, 3))
    # 느슨: Q1은 2명(M1 · M2), Q2는 1명(M3) → 두 질문 평균
    q1, q2 = 1 - comb(n - 2, 3) / comb(n, 3), 1 - comb(n - 1, 3) / comb(n, 3)
    assert random_hit_rate(ds, results, 1) == pytest.approx((q1 + q2) / 2)


def test_보낸_알림을_라벨별로_센다():
    ds, results = tiny()
    assert sent_label_counts(ds, results) == {2: 0, 1: 2, 0: 2}
