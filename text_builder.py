"""서버 DefaultTextBuilder 재현 — 임베딩에 넣을 글을 조립한다.

서버: gdghost-server/src/main/java/com/gdghost/embedding/DefaultTextBuilder.java
조립 규칙은 AI팀 담당(설계 문서 2장). 실험(#7)은 이 모듈을 바꾼 변형으로 한다.

주의 — 순서: 서버는 태그를 사용자가 고른 순서가 아니라 태그 목록 순서(sort_order)로 넘긴다
(MemberProfileReader.tags). 이 함수는 받은 순서 그대로 쓰므로, 호출하는 쪽에서 맞춰 넘긴다.
"""

from dataclasses import dataclass, field

# QuestionCategory.label() — tests/test_config.py가 서버와 같은지 확인한다
CATEGORY_LABELS = {
    "JOB": "취업·채용",
    "CAREER": "커리어·이직",
    "PATH": "진로 고민",
    "CAMPUS": "학교생활",
    "STUDY": "기술·공부",
}


@dataclass(frozen=True)
class Interest:
    name: str
    skills: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemberSource:
    interests: list[Interest]
    experiences: list[str]
    status_note: str | None
    bio: str | None


@dataclass(frozen=True)
class QuestionSource:
    category: str
    title: str
    body: str | None


def member_text(member: MemberSource) -> str:
    """넣을 내용이 없으면 빈 문자열"""
    lines = []
    if member.interests:
        lines.append("관심 분야: " + ", ".join(_interest(i) for i in member.interests))
    if member.experiences:
        lines.append("경험: " + ", ".join(member.experiences))
    if member.status_note is not None:
        lines.append("근황: " + member.status_note)
    if member.bio is not None:
        lines.append("소개: " + member.bio)
    return "\n".join(lines)


def question_text(question: QuestionSource) -> str:
    head = CATEGORY_LABELS[question.category] + " " + question.title
    return head if question.body is None else head + "\n" + question.body


def search_text(category: str, interest_name: str, text: str) -> str:
    """선배 찾기 입력. 질문과 같은 방식으로 분류 + 분야 + 글"""
    return CATEGORY_LABELS[category] + " " + interest_name + "\n" + text


def _interest(interest: Interest) -> str:
    return interest.name if not interest.skills else f"{interest.name}({', '.join(interest.skills)})"
