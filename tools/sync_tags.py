"""서버의 태그 목록 · 질문 분류를 data/tags.json으로 옮긴다.

평가셋의 태그 이름이 서버와 한 글자라도 다르면 점수가 달라지므로, 손으로 옮기지 않고 서버 파일에서 읽는다.
서버 태그가 바뀌면(새 V 마이그레이션) 다시 돌린다.

    python tools/sync_tags.py ../gdghost-server
"""

import json
import re
import sys
from pathlib import Path

SEED = "src/main/resources/db/migration/V3__seed_tags.sql"
CATEGORY = "src/main/java/com/gdghost/question/QuestionCategory.java"
OUT = Path(__file__).resolve().parent.parent / "data" / "tags.json"


def parse_seed(sql: str) -> dict:
    interests = re.findall(r"\('INTEREST', '([^']+)', \d+\)", sql)
    skills: dict[str, list[str]] = {name: [] for name in interests}
    for parent, name in re.findall(r"ROW\('([^']+)', '([^']+)', \d+\)", sql):
        skills[parent].append(name)
    experiences = re.findall(r"\('EXPERIENCE', '([^']+)', '[^']+', \d+\)", sql)
    return {"interests": skills, "experiences": experiences}


def parse_categories(java: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Z]+)\(\"([^\"]+)\"\)", java))


def main() -> None:
    server = Path(sys.argv[1] if len(sys.argv) > 1 else "../gdghost-server")
    tags = parse_seed((server / SEED).read_text(encoding="utf-8"))
    tags["categories"] = parse_categories((server / CATEGORY).read_text(encoding="utf-8"))
    tags["source"] = {"seed": SEED, "category": CATEGORY}
    OUT.write_text(json.dumps(tags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"interests {len(tags['interests'])} · skills {sum(map(len, tags['interests'].values()))} · "
          f"experiences {len(tags['experiences'])} · categories {len(tags['categories'])} → {OUT.name}")


if __name__ == "__main__":
    main()
