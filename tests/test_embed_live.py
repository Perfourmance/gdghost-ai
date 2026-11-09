"""실제 Workers AI를 부른다. 평소에는 건너뛰고, 호출 코드를 고쳤을 때 직접 돌린다.

    LIVE_EMBEDDING=true pytest tests/test_embed_live.py      (Git Bash)
    $env:LIVE_EMBEDDING="true"; pytest tests/test_embed_live.py   (PowerShell)
"""

import os

import pytest

from embed import DIMENSION, Embedder
from scorer import cosine

pytestmark = pytest.mark.skipif(os.getenv("LIVE_EMBEDDING") != "true", reason="LIVE_EMBEDDING=true일 때만")


def test_실제_벡터를_받고_관련_있는_글이_더_비슷하다(tmp_path):
    e = Embedder.from_env(cache_dir=tmp_path)  # 캐시를 비워 실제로 부르게

    question, related, unrelated = e.embed([
        "취업·채용 백엔드 신입 포트폴리오 어떻게 준비하나요?",
        "관심 분야: 백엔드(Spring)\n경험: 인턴, 정규직 취업",
        "관심 분야: 디자인\n경험: 교환학생",
    ])

    assert question.shape == (DIMENSION,)
    assert e.stats.api_calls == 1 and e.stats.embedded == 3
    assert cosine(question, related) > cosine(question, unrelated)
