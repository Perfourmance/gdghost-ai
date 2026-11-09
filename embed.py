"""Workers AI 임베딩 + 캐시.

서버 WorkersAiEmbeddingClient와 같은 조건으로 부른다.
- 모델 @cf/baai/bge-m3 · 1024차원 · truncate_inputs: true (긴 글은 모델 한도에서 자름)
- 실패하면 1회 재시도. 400 · 413은 글 자체 문제라 다시 부르지 않는다
- 에러 메시지에 계정 ID · 토큰을 넣지 않는다 (주소에 계정 ID가 있다)

다른 점: 평가는 같은 글을 여러 번 쓰므로 글마다 결과를 cache/에 저장하고, 있으면 부르지 않는다.
서버는 한 번에 한 글이라 시간 초과가 3초지만, 여기는 여러 글을 묶어 보내므로 더 길게 둔다.

    python embed.py 글목록.txt     # 한 줄에 한 글. 두 번째 실행부터는 API를 부르지 않는다
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

MODEL = "@cf/baai/bge-m3"
DIMENSION = 1024
BASE_URL = "https://api.cloudflare.com/client/v4"
ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "cache" / "embeddings"


class EmbeddingError(Exception):
    """메시지에는 상태 코드 · 예외 종류만 담는다"""

    def __init__(self, message: str, bad_input: bool = False):
        super().__init__(message)
        self.bad_input = bad_input


@dataclass
class Stats:
    api_calls: int = 0       # HTTP 요청 수 (재시도 포함)
    embedded: int = 0        # API로 새로 받은 글 수
    cache_hits: int = 0      # 캐시에서 꺼낸 글 수


class Embedder:

    def __init__(self, account_id: str, api_token: str, cache_dir: Path | None = DEFAULT_CACHE,
                 batch_size: int = 32, timeout: float = 30.0, retry_delay: float = 1.0,
                 session: requests.Session | None = None):
        if not account_id or not api_token:
            raise EmbeddingError(".env에 CLOUDFLARE_ACCOUNT_ID · WORKERS_AI_API_TOKEN이 없음")
        self._url = f"{BASE_URL}/accounts/{account_id}/ai/run/{MODEL}"
        self._headers = {"Authorization": f"Bearer {api_token}"}
        self.cache_dir = None if cache_dir is None else Path(cache_dir) / _slug(MODEL)
        self.batch_size = batch_size
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.session = session or requests.Session()
        self.stats = Stats()

    @classmethod
    def from_env(cls, **kwargs) -> "Embedder":
        load_dotenv(ROOT / ".env")
        return cls(os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip(), os.getenv("WORKERS_AI_API_TOKEN", "").strip(),
                   **kwargs)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """입력 순서대로 1024차원 float32 벡터. 같은 글은 한 번만 부른다"""
        vectors: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in dict.fromkeys(texts):  # 순서를 지키며 중복 제거
            cached = self._read_cache(text)
            if cached is None:
                missing.append(text)
            else:
                vectors[text] = cached
                self.stats.cache_hits += 1
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            for text, raw in zip(batch, self._call(batch)):
                self._write_cache(text, raw)
                vectors[text] = _as_vector(raw)
                self.stats.embedded += 1
        return [vectors[text] for text in texts]

    # ---------------------------------------------------------------- API

    def _call(self, texts: list[str]) -> list[list[float]]:
        body = {"text": texts, "truncate_inputs": True}
        try:
            return self._post(body, len(texts))
        except EmbeddingError as first:
            if first.bad_input:
                raise
            time.sleep(self.retry_delay)
            return self._post(body, len(texts))

    def _post(self, body: dict, expected: int) -> list[list[float]]:
        self.stats.api_calls += 1
        try:
            response = self.session.post(self._url, headers=self._headers, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            # 원래 예외 메시지에는 계정 ID가 든 주소가 있으므로 종류만 남긴다
            raise EmbeddingError(f"임베딩 실패: {type(e).__name__}") from None
        if response.status_code != 200:
            raise EmbeddingError(f"임베딩 실패: HTTP {response.status_code}",
                                 bad_input=response.status_code in (400, 413))
        try:
            payload = response.json()
            data = payload["result"]["data"] if payload.get("success") else None
        except (ValueError, KeyError, TypeError):
            data = None
        if data is None:
            raise EmbeddingError("임베딩 응답에 벡터가 없음")
        if len(data) != expected or any(len(v) != DIMENSION for v in data):
            raise EmbeddingError("임베딩 응답의 개수 또는 차원이 다름")
        return data

    # ---------------------------------------------------------------- 캐시

    def _cache_path(self, text: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha256(text.encode("utf-8")).hexdigest() + ".json")

    def _read_cache(self, text: str) -> np.ndarray | None:
        path = self._cache_path(text)
        if path is None or not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        # 해시가 겹치는 일은 사실상 없지만, 다른 글의 벡터를 쓰지 않게 원문까지 확인한다
        if entry.get("text") != text or entry.get("model") != MODEL:
            return None
        return _as_vector(entry["vector"])

    def _write_cache(self, text: str, raw: list[float]) -> None:
        path = self._cache_path(text)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # 응답 숫자를 그대로 남긴다 (float32로 줄이기 전 값)
        path.write_text(json.dumps({"model": MODEL, "text": text, "vector": raw}, ensure_ascii=False),
                        encoding="utf-8")


def _as_vector(raw: list[float]) -> np.ndarray:
    """서버는 벡터를 float[](32비트)로 읽는다"""
    return np.asarray(raw, dtype=np.float32)


def _slug(model: str) -> str:
    return model.strip("@").replace("/", "_")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    texts = [line for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
    embedder = Embedder.from_env()
    try:
        vectors = embedder.embed(texts)
    except EmbeddingError as e:
        print(f"FAIL: {e}")
        return 1
    s = embedder.stats
    print(f"글 {len(texts)}개 → 벡터 {len(vectors)}개 ({DIMENSION}차원)")
    print(f"  API 요청 {s.api_calls}회 · 새로 받음 {s.embedded}개 · 캐시 {s.cache_hits}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
