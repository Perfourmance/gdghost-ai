from exp_text import member_reordered, member_sentences, question_without_category
from text_builder import Interest, MemberSource, QuestionSource

MEMBER = MemberSource([Interest("백엔드", ["Spring"]), Interest("데브옵스")], ["인턴", "해커톤"], "백엔드 개발자", "취업 이야기")


def test_부원_문장형은_라벨_없이_같은_내용이다():
    assert member_sentences(MEMBER) == ("백엔드(Spring), 데브옵스에 관심이 있습니다.\n"
                                        "인턴, 해커톤 경험이 있습니다.\n백엔드 개발자\n취업 이야기")


def test_부원_순서_바꾸기는_소개부터다():
    assert member_reordered(MEMBER) == ("소개: 취업 이야기\n근황: 백엔드 개발자\n경험: 인턴, 해커톤\n"
                                        "관심 분야: 백엔드(Spring), 데브옵스")


def test_빈_항목은_빼고_조립한다():
    minimal = MemberSource([Interest("백엔드")], [], None, None)
    assert member_sentences(minimal) == "백엔드에 관심이 있습니다."
    assert member_reordered(minimal) == "관심 분야: 백엔드"


def test_질문에서_분류명만_뺀다():
    assert question_without_category(QuestionSource("JOB", "제목", "본문")) == "제목\n본문"
    assert question_without_category(QuestionSource("JOB", "제목", None)) == "제목"
