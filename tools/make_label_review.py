"""라벨 검토용 문서를 만든다: labeling/review.md

질문마다 정답(2) · 애매(1)로 단 부원의 프로필을 옆에 붙여, 라벨이 맞는지 눈으로 훑을 수 있게 한다.

    python tools/make_label_review.py                    # data/labels.csv
    python tools/make_label_review.py <labels.csv>       # 다른 사람의 라벨
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "labeling" / "review.md"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def profile(m: dict) -> str:
    fields = ", ".join(i["name"] + (f"({', '.join(i['skills'])})" if i["skills"] else "") for i in m["interests"])
    parts = [fields, ", ".join(m["experiences"]) or "-", m["status_note"] or "-", m["bio"] or "-"]
    return " / ".join(parts)


def main() -> None:
    labels_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "labels.csv"
    tags = json.loads((DATA / "tags.json").read_text(encoding="utf-8"))
    members = {m["id"]: m for m in read_jsonl(DATA / "members.jsonl")}
    questions = read_jsonl(DATA / "questions.jsonl")
    with labels_path.open(encoding="utf-8-sig", newline="") as f:
        labels = {row["question_id"]: row for row in csv.DictReader(f)}

    lines = ["# 라벨 검토", "",
             f"`{labels_path.name}` 기준. 부원 칸은 관심 분야 / 경험 / 근황 / 소개 순서.", "",
             "고칠 곳은 `data/labels.csv`에서 고친 뒤 `python tools/make_label_review.py`로 다시 만든다.", ""]
    for q in questions:
        row = labels.get(q["id"], {})
        title = f"## {q['id']} [{tags['categories'][q['category']]}] {q['title']}"
        lines += [title + (" (익명)" if q["anonymous"] else ""), ""]
        if q["body"]:
            lines += [f"> {q['body']}", ""]
        for score, key in (("**2 정답**", "correct"), ("1 애매", "partial")):
            ids = [i for i in row.get(key, "").split(";") if i]
            if not ids:
                lines.append(f"- {score}: 없음")
                continue
            lines.append(f"- {score}")
            lines += [f"  - **{i}** {profile(members[i])}" for i in ids]
        if row.get("memo"):
            lines.append(f"- 메모: {row['memo']}")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"질문 {len(questions)}개 → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
