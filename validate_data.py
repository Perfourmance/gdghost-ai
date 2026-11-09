"""평가셋이 서버 입력 규칙을 지키는지 검사한다.

서버에서 만들 수 없는 프로필 · 질문으로 평가하면 결과가 의미 없으므로, 서버와 같은 제한을 건다.
(TagService.validateSelection, ProfileUpdateRequest, QuestionCreateRequest 기준)

    python validate_data.py            # data/
    python validate_data.py <폴더>     # 다른 폴더 (tags.json은 data/ 것을 씀)
"""

import csv
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

POSITIONS = {"DEV", "AI", "DESIGN", "ETC"}
STATUSES = {"ENROLLED", "LEAVE", "GRADUATED"}
MAX_INTERESTS = 4
MAX_SKILLS_PER_INTEREST = 3
MAX_EXPERIENCES = 6
STATUS_NOTE_MAX = 60
BIO_MAX = 120
DEPARTMENT_MAX = 30
TITLE_MIN, TITLE_MAX = 2, 80
BODY_MAX = 3000


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path.name}:{number} JSON 형식 오류: {e}")
    return rows


def check_text(errors: list[str], where: str, value, limit: int, required: bool = False, minimum: int = 0) -> None:
    if value is None:
        if required:
            errors.append(f"{where}: 비어 있음")
        return
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}: 빈 문자열이면 null로")
    elif not minimum <= len(value) <= limit:
        errors.append(f"{where}: {len(value)}자 (허용 {minimum}–{limit}자)")


def check_unique(errors: list[str], where: str, values: list) -> None:
    if len(values) != len(set(values)):
        errors.append(f"{where}: 중복 있음")


def check_member(m: dict, tags: dict, errors: list[str]) -> None:
    where = m.get("id", "?")
    interests = m.get("interests", [])
    names = [i.get("name") for i in interests]
    if not 1 <= len(interests) <= MAX_INTERESTS:
        errors.append(f"{where}: 관심 분야 {len(interests)}개 (1–{MAX_INTERESTS}개 필수)")
    check_unique(errors, f"{where} 관심 분야", names)
    for interest in interests:
        name, skills = interest.get("name"), interest.get("skills", [])
        if name not in tags["interests"]:
            errors.append(f"{where}: 없는 관심 분야 '{name}'")
            continue
        if len(skills) > MAX_SKILLS_PER_INTEREST:
            errors.append(f"{where}: {name} 세부 기술 {len(skills)}개 (분야당 {MAX_SKILLS_PER_INTEREST}개까지)")
        check_unique(errors, f"{where} {name} 세부 기술", skills)
        for skill in skills:
            if skill not in tags["interests"][name]:
                errors.append(f"{where}: '{skill}'은 {name}의 세부 기술이 아님")
    experiences = m.get("experiences", [])
    if len(experiences) > MAX_EXPERIENCES:
        errors.append(f"{where}: 경험 태그 {len(experiences)}개 ({MAX_EXPERIENCES}개까지)")
    check_unique(errors, f"{where} 경험 태그", experiences)
    for experience in experiences:
        if experience not in tags["experiences"]:
            errors.append(f"{where}: 없는 경험 태그 '{experience}'")
    check_text(errors, f"{where} 근황", m.get("status_note"), STATUS_NOTE_MAX)
    check_text(errors, f"{where} 소개", m.get("bio"), BIO_MAX)
    check_text(errors, f"{where} 학과", m.get("department"), DEPARTMENT_MAX, required=True, minimum=1)
    if m.get("position") not in POSITIONS:
        errors.append(f"{where}: 포지션 '{m.get('position')}' (허용 {sorted(POSITIONS)})")
    if m.get("status") not in STATUSES:
        errors.append(f"{where}: 상태 '{m.get('status')}' (허용 {sorted(STATUSES)})")
    if not isinstance(m.get("generation"), int) or not 1 <= m["generation"] <= 99:
        errors.append(f"{where}: 기수 '{m.get('generation')}' (1–99)")


def check_question(q: dict, tags: dict, errors: list[str]) -> None:
    where = q.get("id", "?")
    if q.get("category") not in tags["categories"]:
        errors.append(f"{where}: 분류 '{q.get('category')}' (허용 {list(tags['categories'])})")
    check_text(errors, f"{where} 제목", q.get("title"), TITLE_MAX, required=True, minimum=TITLE_MIN)
    check_text(errors, f"{where} 본문", q.get("body"), BODY_MAX)
    if not isinstance(q.get("anonymous"), bool):
        errors.append(f"{where}: anonymous는 true/false")
    asker = q.get("asker") or {}
    check_text(errors, f"{where} 질문자 학과", asker.get("department"), DEPARTMENT_MAX, required=True, minimum=1)
    if asker.get("position") not in POSITIONS:
        errors.append(f"{where}: 질문자 포지션 '{asker.get('position')}'")
    if not isinstance(asker.get("generation"), int) or not 1 <= asker["generation"] <= 99:
        errors.append(f"{where}: 질문자 기수 '{asker.get('generation')}'")


def check_labels(path: Path, member_ids: set, question_ids: set, errors: list[str]) -> int:
    """question_id,correct,partial,memo — correct · partial은 ;로 구분한 부원 id"""
    if not path.exists():
        return 0
    seen = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            if qid not in question_ids:
                errors.append(f"labels: 없는 질문 '{qid}'")
            if qid in seen:
                errors.append(f"labels: '{qid}' 중복")
            seen.add(qid)
            correct = {m for m in row.get("correct", "").split(";") if m}
            partial = {m for m in row.get("partial", "").split(";") if m}
            for mid in correct | partial:
                if mid not in member_ids:
                    errors.append(f"labels {qid}: 없는 부원 '{mid}'")
            if correct & partial:
                errors.append(f"labels {qid}: 정답과 애매에 같은 부원 {sorted(correct & partial)}")
    missing = question_ids - seen
    if missing:
        errors.append(f"labels: 라벨 없는 질문 {sorted(missing)}")
    return len(seen)


def main() -> int:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
    tags = json.loads((DATA / "tags.json").read_text(encoding="utf-8"))
    members = read_jsonl(data / "members.jsonl")
    questions = read_jsonl(data / "questions.jsonl")
    errors: list[str] = []

    check_unique(errors, "부원 id", [m.get("id") for m in members])
    check_unique(errors, "질문 id", [q.get("id") for q in questions])
    for m in members:
        check_member(m, tags, errors)
    for q in questions:
        check_question(q, tags, errors)
    labeled = check_labels(data / "labels.csv", {m.get("id") for m in members},
                           {q.get("id") for q in questions}, errors)

    print(f"부원 {len(members)}명 · 질문 {len(questions)}개 · 라벨 {labeled}개 질문")
    if questions:
        counts = {c: sum(q.get("category") == c for q in questions) for c in tags["categories"]}
        anonymous = sum(q.get("anonymous") is True for q in questions)
        print("  분류별 " + " · ".join(f"{tags['categories'][c]} {n}" for c, n in counts.items())
              + f" · 익명 {anonymous}")
    if errors:
        print(f"\nFAIL {len(errors)}건")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
