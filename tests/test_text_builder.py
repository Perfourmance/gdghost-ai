"""서버 DefaultTextBuilderTest 3개를 그대로 옮긴 것. 기대값은 Java 테스트와 같다."""

from text_builder import Interest, MemberSource, QuestionSource, member_text, question_text, search_text


def test_부원_글은_설계_문서_순서로_조립한다():
    text = member_text(MemberSource(
        interests=[Interest("백엔드", ["Spring", "DB 설계"]), Interest("기획", [])],
        experiences=["인턴", "공모전 수상"],
        status_note="인턴 근무 중",
        bio="취업 과정을 이야기해요",
    ))

    assert text == ("관심 분야: 백엔드(Spring, DB 설계), 기획\n"
                    "경험: 인턴, 공모전 수상\n"
                    "근황: 인턴 근무 중\n"
                    "소개: 취업 과정을 이야기해요")


def test_부원_글은_없는_항목을_뺀다():
    assert member_text(MemberSource([], [], None, "소개만")) == "소개: 소개만"
    assert member_text(MemberSource([], [], None, None)) == ""


def test_질문_글은_분류_제목_본문이고_본문은_없어도_된다():
    assert question_text(QuestionSource("JOB", "이력서", "무엇을 쓰나요")) == "취업·채용 이력서\n무엇을 쓰나요"
    assert question_text(QuestionSource("STUDY", "공부법", None)) == "기술·공부 공부법"


# --- 서버 테스트에 없는 것 (설계 문서 3장 예시 기준)

def test_선배_찾기_글은_분류_분야_입력글이다():
    assert search_text("PATH", "데이터·AI", "대학원 갈지 고민이에요") == "진로 고민 데이터·AI\n대학원 갈지 고민이에요"
