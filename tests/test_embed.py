"""embed.py — 실제 API를 부르지 않고 가짜 응답으로 확인한다.

앞의 7개는 서버 WorkersAiEmbeddingClientTest와 같은 상황, 뒤는 캐시 · 묶음 호출.
"""

import json

import pytest
import requests

from embed import DIMENSION, MODEL, Embedder, EmbeddingError

ACCOUNT = "acct-secret-123"
TOKEN = "token-secret-456"


class FakeResponse:
    def __init__(self, status: int, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """응답을 차례로 돌려주고, 받은 요청을 기록한다. 응답 대신 예외를 넣으면 던진다"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, headers, json, timeout):
        self.requests.append({"url": url, "headers": headers, "json": json})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(json)
        return response


def ok(texts_or_count, dim: int = DIMENSION) -> FakeResponse:
    count = texts_or_count if isinstance(texts_or_count, int) else len(texts_or_count)
    return FakeResponse(200, {"success": True, "result": {"shape": [count, dim],
                                                          "data": [[0.1 * (i + 1)] * dim for i in range(count)]}})


def echo(body) -> FakeResponse:
    """받은 글 수만큼 벡터를 돌려준다"""
    return ok(len(body["text"]))


def embedder(session, tmp_path=None, **kwargs) -> Embedder:
    return Embedder(ACCOUNT, TOKEN, cache_dir=tmp_path, retry_delay=0, session=session, **kwargs)


# --- 서버 WorkersAiEmbeddingClientTest와 같은 상황

def test_실제_응답_모양에서_벡터를_꺼낸다():
    session = FakeSession(ok(2))

    vectors = embedder(session).embed(["가", "나"])

    assert [v.shape for v in vectors] == [(DIMENSION,), (DIMENSION,)]
    assert vectors[0].dtype.name == "float32"
    assert session.requests[0]["json"] == {"text": ["가", "나"], "truncate_inputs": True}
    assert session.requests[0]["url"].endswith(f"/accounts/{ACCOUNT}/ai/run/{MODEL}")
    assert session.requests[0]["headers"] == {"Authorization": f"Bearer {TOKEN}"}


def test_실패하면_한_번_더_부른다():
    session = FakeSession(FakeResponse(500), ok(1))

    assert len(embedder(session).embed(["가"])) == 1
    assert len(session.requests) == 2


def test_두_번_실패하면_상태_코드만_담아_던지고_비밀값은_없다():
    session = FakeSession(FakeResponse(503), FakeResponse(503))

    with pytest.raises(EmbeddingError) as e:
        embedder(session).embed(["가"])

    assert str(e.value) == "임베딩 실패: HTTP 503"
    assert ACCOUNT not in str(e.value) and TOKEN not in str(e.value)
    assert not e.value.bad_input


@pytest.mark.parametrize("status", [400, 413])
def test_글_문제인_400은_다시_부르지_않는다(status):
    session = FakeSession(FakeResponse(status))

    with pytest.raises(EmbeddingError) as e:
        embedder(session).embed(["가"])

    assert e.value.bad_input
    assert len(session.requests) == 1


def test_차원이_다르면_실패한다():
    session = FakeSession(ok(1, dim=768), ok(1, dim=768))

    with pytest.raises(EmbeddingError, match="차원"):
        embedder(session).embed(["가"])


def test_연결이_안_되면_종류만_담아_던진다():
    error = requests.ConnectionError(f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run")
    session = FakeSession(error, error)

    with pytest.raises(EmbeddingError) as e:
        embedder(session).embed(["가"])

    assert str(e.value) == "임베딩 실패: ConnectionError"
    assert ACCOUNT not in str(e.value)
    assert e.value.__cause__ is None  # 주소가 든 원래 예외를 달고 가지 않는다


def test_키가_비면_만들_수_없다():
    with pytest.raises(EmbeddingError):
        Embedder("", TOKEN)
    with pytest.raises(EmbeddingError):
        Embedder(ACCOUNT, "  ".strip())


# --- 캐시 · 묶음 호출

def test_두_번째_실행은_API를_부르지_않는다(tmp_path):
    first = embedder(FakeSession(echo), tmp_path)
    vectors = first.embed(["가", "나"])

    second_session = FakeSession()  # 부르면 pop에서 실패한다
    second = embedder(second_session, tmp_path)
    again = second.embed(["가", "나"])

    assert second_session.requests == []
    assert second.stats.api_calls == 0 and second.stats.cache_hits == 2
    assert all((a == b).all() for a, b in zip(vectors, again))


def test_캐시에_없는_글만_부른다(tmp_path):
    embedder(FakeSession(echo), tmp_path).embed(["가"])

    session = FakeSession(echo)
    e = embedder(session, tmp_path)
    e.embed(["가", "나", "다"])

    assert session.requests[0]["json"]["text"] == ["나", "다"]
    assert (e.stats.cache_hits, e.stats.embedded) == (1, 2)


def test_같은_글은_한_번만_부르고_입력_순서대로_돌려준다():
    session = FakeSession(echo)

    vectors = embedder(session).embed(["가", "나", "가"])

    assert session.requests[0]["json"]["text"] == ["가", "나"]
    assert len(vectors) == 3 and (vectors[0] == vectors[2]).all() and not (vectors[0] == vectors[1]).all()


def test_묶음_크기로_나눠_부른다():
    session = FakeSession(echo, echo, echo)

    embedder(session, batch_size=2).embed(["1", "2", "3", "4", "5"])

    assert [r["json"]["text"] for r in session.requests] == [["1", "2"], ["3", "4"], ["5"]]


def test_캐시_파일의_원문이_다르면_쓰지_않는다(tmp_path):
    e = embedder(FakeSession(echo), tmp_path)
    e.embed(["가"])
    path = next((tmp_path).rglob("*.json"))
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["text"] = "다른 글"
    path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")

    session = FakeSession(echo)
    embedder(session, tmp_path).embed(["가"])

    assert len(session.requests) == 1


def test_실패한_글은_캐시에_남지_않는다(tmp_path):
    with pytest.raises(EmbeddingError):
        embedder(FakeSession(FakeResponse(400)), tmp_path).embed(["가"])

    assert list(tmp_path.rglob("*.json")) == []
