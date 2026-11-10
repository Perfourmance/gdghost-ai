"""라벨 작업지를 만든다.

- labeling/members.md       부원 전체 한눈에 보기 (매칭에 쓰이는 정보만)
- labeling/members.json     members.jsonl을 들여쓰기한 보기용 사본. 고칠 때는 원본 jsonl을 고친다
- labeling/labels-template.csv  질문 전체 + 빈 정답 칸. 엑셀에서 열어 채운다

질문의 note(작성자 표시)는 라벨 판단이 끌려가지 않게 작업지에 넣지 않는다.

    python tools/make_label_sheet.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "labeling"

STATUS = {"ENROLLED": "재학", "LEAVE": "휴학", "GRADUATED": "졸업"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def interests(member: dict) -> str:
    return ", ".join(i["name"] + (f"({', '.join(i['skills'])})" if i["skills"] else "") for i in member["interests"])


def members_sheet(members: list[dict]) -> str:
    lines = ["# 부원 프로필 (라벨용)", "",
             "질문마다 이 표를 훑으며 2(정답) · 1(애매)인 부원을 고른다. 기수 · 학과는 판단에 쓰지 않는다.", "",
             "| ID | 관심 분야 (세부 기술) | 경험 | 근황 | 소개 |", "|---|---|---|---|---|"]
    for m in members:
        lines.append(f"| **{m['id']}** | {interests(m)} | {', '.join(m['experiences']) or '-'} | "
                     f"{m['status_note'] or '-'} | {m['bio'] or '-'} |")
    lines += ["", "<details><summary>참고: 학과 · 포지션 · 기수 · 상태 (라벨 판단에는 쓰지 않음)</summary>", "",
              "| ID | 학과 | 포지션 | 기수 | 상태 |", "|---|---|---|---|---|"]
    for m in members:
        lines.append(f"| {m['id']} | {m['department']} | {m['position']} | {m['generation']} | {STATUS[m['status']]} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def main() -> None:
    tags = json.loads((DATA / "tags.json").read_text(encoding="utf-8"))
    members = read_jsonl(DATA / "members.jsonl")
    questions = read_jsonl(DATA / "questions.jsonl")
    OUT.mkdir(exist_ok=True)
    (OUT / "members.md").write_text(members_sheet(members), encoding="utf-8", newline="\n")
    (OUT / "members.json").write_text(json.dumps(members, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    # 엑셀이 한글을 깨뜨리지 않게 BOM을 붙인다 (utf-8-sig)
    with (OUT / "labels-template.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "correct", "partial", "memo", "category", "question"])
        for q in questions:
            text = q["title"] + (f" / {q['body']}" if q["body"] else "")
            writer.writerow([q["id"], "", "", "", tags["categories"][q["category"]], text])
    print(f"부원 {len(members)}명 → labeling/members.md · members.json, 질문 {len(questions)}개 → labeling/labels-template.csv")


if __name__ == "__main__":
    main()
