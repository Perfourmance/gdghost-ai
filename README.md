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
