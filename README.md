# gdghost-ai

GDGhost AI 매칭(선배 추천)의 품질 평가와 튜닝.

서버 코드는 [gdghost-server](https://github.com/Perfourmance/gdghost-server)에 있고,
이 저장소는 AI팀의 평가 코드 · 실험 결과 · 분석 문서를 둡니다.

## 실행

Python 3.13 기준입니다.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash. PowerShell은 .venv\Scripts\Activate.ps1, macOS는 source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env               # Workers AI 계정 ID · 토큰을 채운다 (.env는 git에 안 올라감)
python check_connection.py         # 연결 확인
```

정상이면 이렇게 나옵니다.

```
OK: @cf/baai/bge-m3, 3개, 1024차원
  질문 ↔ 맞는 선배   cos = 0.610
  질문 ↔ 무관한 선배 cos = 0.413
```

> Git Bash에서 한글이 깨지면 `PYTHONIOENCODING=utf-8 python check_connection.py`

## 임베딩

```python
from embed import Embedder

embedder = Embedder.from_env()            # .env의 Workers AI 키
vectors = embedder.embed(["글1", "글2"])   # 입력 순서대로 1024차원 벡터
```

- 서버와 같은 모델 · 옵션(`@cf/baai/bge-m3`, `truncate_inputs`)으로 부르고, 실패하면 1회 재시도합니다.
- 받은 벡터는 `cache/embeddings/`에 글마다 저장해서 **같은 글은 다시 부르지 않습니다** (무료 한도 절약). `cache/`는 git에 올라가지 않습니다.
- 명령줄: `python embed.py 글목록.txt` (한 줄에 한 글)

## 평가

```bash
python evaluate.py                                     # 기준선 → results/baseline-detail.md · baseline.json
python evaluate.py --config config/실험.yaml --name 실험  # 다른 설정으로
```

- 평가셋(`data/`)을 서버와 같은 글로 조립해 임베딩하고, 서버와 같은 공식으로 질문마다 3명을 뽑아 정답표와 비교합니다.
- 결과 요약은 [results/baseline.md](results/baseline.md).
- `--export-java 경로`: 같은 벡터를 서버 `CandidateScorer`로 다시 계산해 대조할 입력 파일을 만듭니다 (점수는 `.scores`).

```bash
python exp_weights.py      # 가중치 · 컷 조합 비교 → results/exp-weights-sweep.md
```

- 짝마다 유사도와 규칙 항목을 한 번만 계산해 두고 가중치만 바꿔 다시 더하므로, 조합 수천 개를 몇 초 안에 돌립니다. 같은 설정이면 `evaluate.py`와 결과가 같습니다.
- 결론은 [results/exp-weights.md](results/exp-weights.md), 추천 설정은 [config/exp-weights.yaml](config/exp-weights.yaml).

## 테스트

```bash
pytest
```

| 테스트 | 내용 |
|---|---|
| `tests/test_text_builder.py` | 서버 `DefaultTextBuilderTest` 3개 이식 |
| `tests/test_scorer.py` | 서버 `CandidateScorerTest` 8개 이식 + Java와 다를 수 있는 반올림 · 유사도 계산 |
| `tests/test_embed.py` | 임베딩 호출 · 재시도 · 캐시 (가짜 응답, API 안 부름) |
| `tests/test_embed_live.py` | 실제 API 호출. `LIVE_EMBEDDING=true`일 때만 |
| `tests/test_evaluate.py` | 태그 순서 · 라벨 읽기 · Hit@3 · 무작위 기준선 |
| `tests/test_exp_weights.py` | 빠른 계산이 `evaluate.py`와 같은지 (임베딩 캐시가 있을 때) |
| `tests/test_config.py` | `config/matching.yaml` · 분류 이름이 서버와 같은지 (옆에 `../gdghost-server`가 있을 때) |

점수 계산(`scorer.py`)과 글 조립(`text_builder.py`)은 서버 Java 코드를 그대로 옮긴 것입니다.
서버 코드가 바뀌면 같이 고치고, 서버 테스트의 기대값으로 맞는지 확인합니다.

## 작업 방법

이슈 → 브랜치 → PR → 머지. gdghost-server의 [CONTRIBUTING](https://github.com/Perfourmance/gdghost-server/blob/main/CONTRIBUTING.md)과 같은 흐름입니다.

| 접두어 | 언제 |
|---|---|
| `feat/` | 평가 코드 |
| `exp/` | 튜닝 실험 |
| `docs/` | 문서 |
| `chore/` | 환경 · 설정 |
| `fix/` | 버그 |

- 브랜치: `feat/3-eval-dataset` (종류/이슈번호-짧은이름)
- 커밋: `feat: 평가셋 부원 프로필 (#3)`
- PR 본문: `Closes #3`
