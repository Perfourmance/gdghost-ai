"""config/matching.yaml · 분류 이름이 서버와 같은지 확인한다.

서버 저장소가 옆 폴더(../gdghost-server)에 있을 때만 비교하고, 없으면 건너뛴다.
서버 값이 바뀌면 여기서 실패하므로, 그때 config/matching.yaml을 다시 복사한다.
"""

import re
from pathlib import Path

import pytest
import yaml

from scorer import DEFAULT_CONFIG, MatchingConfig
from text_builder import CATEGORY_LABELS

SERVER = Path(__file__).resolve().parents[2] / "gdghost-server"
SERVER_YAML = SERVER / "src/main/resources/application.yaml"
SERVER_CATEGORY = SERVER / "src/main/java/com/gdghost/question/QuestionCategory.java"

needs_server = pytest.mark.skipif(not SERVER_YAML.exists(), reason="../gdghost-server가 없음")


def test_기준선_설정을_읽는다():
    config = MatchingConfig.load()

    assert config.version == "2026-10-v1"
    assert config.weights.rule + config.weights.profile + config.weights.history == pytest.approx(1.0)
    assert config.min_score == 0.15
    assert config.history.adopted_bonus == 0.1
    assert config.activity.answer_count_cap == 10
    assert set(config.category_experience) == set(CATEGORY_LABELS)


@needs_server
def test_matching_설정이_서버와_같다():
    server = yaml.safe_load(SERVER_YAML.read_text(encoding="utf-8"))["matching"]
    ours = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["matching"]

    assert ours == server


@needs_server
def test_분류_이름이_서버와_같다():
    java = SERVER_CATEGORY.read_text(encoding="utf-8")

    assert dict(re.findall(r"([A-Z]+)\(\"([^\"]+)\"\)", java)) == CATEGORY_LABELS
