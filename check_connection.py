"""Workers AI 연결 확인.

서버와 같은 모델 · 옵션(@cf/baai/bge-m3, truncate_inputs)으로 세 글을 임베딩하고
차원과 유사도를 출력한다. 키 · 계정 ID는 출력하지 않는다.

    python check_connection.py
"""

import math
import os
import sys

import requests
from dotenv import load_dotenv

MODEL = "@cf/baai/bge-m3"
DIMENSION = 1024
TIMEOUT_SECONDS = 10

SAMPLES = {
    "question": "취업·채용 백엔드 신입 취업 준비 어떻게 하나요?\n"
                "스프링 공부 중인데 포트폴리오를 어떻게 준비해야 할지 모르겠어요",
    "matching_member": "관심 분야: 백엔드(Spring, JPA)\n경험: 인턴, 정규직 취업\n소개: 백엔드 개발자로 일하고 있어요",
    "unrelated_member": "관심 분야: 디자인\n경험: 교환학생\n소개: 독일로 교환학생 다녀왔어요",
}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if norm == 0 else dot / norm


def main() -> int:
    load_dotenv()
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.getenv("WORKERS_AI_API_TOKEN", "").strip()
    if not account_id or not token:
        print("FAIL: .env에 CLOUDFLARE_ACCOUNT_ID · WORKERS_AI_API_TOKEN을 채워 주세요 (.env.example 참고)")
        return 1

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"text": list(SAMPLES.values()), "truncate_inputs": True},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        # 주소에 계정 ID가 있으므로 예외 메시지는 출력하지 않는다
        print(f"FAIL: 연결 실패 ({type(e).__name__})")
        return 1

    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    if response.status_code != 200 or not body.get("success"):
        codes = [error.get("code") for error in body.get("errors", [])]
        print(f"FAIL: HTTP {response.status_code}, error codes {codes}")
        return 1

    vectors = body["result"]["data"]
    if len(vectors) != len(SAMPLES) or any(len(v) != DIMENSION for v in vectors):
        print("FAIL: 응답의 개수 또는 차원이 다름")
        return 1

    question, matching, unrelated = vectors
    print(f"OK: {MODEL}, {len(vectors)}개, {DIMENSION}차원")
    print(f"  질문 ↔ 맞는 선배   cos = {cosine(question, matching):.3f}")
    print(f"  질문 ↔ 무관한 선배 cos = {cosine(question, unrelated):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
