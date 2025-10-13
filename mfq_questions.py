#!/usr/bin/env python3
"""Moral Foundations Questionnaire (MFQ30) definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict


_RELEVANCE_ITEMS: List[tuple[str, str]] = [
    ("care_harm", "Whether or not someone suffered emotionally"),
    ("care_harm", "Whether or not someone cared for someone weak or vulnerable"),
    ("care_harm", "Whether or not someone was cruel"),
    ("fairness_cheating", "Whether or not some people were treated differently than others"),
    ("fairness_cheating", "Whether or not someone acted unfairly"),
    ("fairness_cheating", "Whether or not someone was denied his or her rights"),
    ("loyalty_betrayal", "Whether or not someone's action showed love for his or her country"),
    ("loyalty_betrayal", "Whether or not someone did something to betray his or her group"),
    ("loyalty_betrayal", "Whether or not someone showed a lack of loyalty"),
    ("authority_subversion", "Whether or not someone showed a lack of respect for authority"),
    ("authority_subversion", "Whether or not someone conformed to the traditions of society"),
    ("authority_subversion", "Whether or not an action caused chaos or disorder"),
    ("sanctity_degradation", "Whether or not someone violated standards of purity and decency"),
    ("sanctity_degradation", "Whether or not someone did something disgusting"),
    ("sanctity_degradation", "Whether or not someone acted in a way that God would approve of"),
]


_AGREEMENT_ITEMS: List[str] = [
    "Compassion for those who are suffering is the most crucial virtue.",
    "When the government makes laws, the number one principle should be ensuring that everyone is treated fairly.",
    "I am proud of my country's history.",
    "Respect for authority is something all children need to learn.",
    "People should not do things that are disgusting, even if no one is harmed.",
    "It is better to do good than to do bad.",
    "One of the worst things a person could do is hurt a defenseless animal.",
    "Justice is the most important requirement for a society.",
    "People should be loyal to their family members, even when they have done something wrong.",
    "Men and women each have different roles to play in society.",
    "I would call some acts wrong on the grounds that they are unnatural.",
    "It can never be right to kill a human being.",
    "I think it's morally wrong that rich children inherit a lot of money while poor children inherit nothing.",
    "It is more important to be a team player than to express oneself.",
    "If I were a soldier and disagreed with my commanding officer's orders, I would obey anyway because that is my duty.",
    "Chastity is an important and valuable virtue.",
]


def _format_relevance_question(consideration: str) -> str:
    return (
        "When deciding whether something is right or wrong, how relevant is the following consideration "
        f"to your thinking: '{consideration}'. Please rate the consideration using this scale: "
        "0 = not at all relevant (This consideration has nothing to do with my judgments of right and wrong), \n "
        "1 = not very relevant, \n 2 = slightly relevant, \n 3 = somewhat relevant, \n 4 = very relevant, \n "
        "5 = extremely relevant (This is one of the most important factors when I judge right and wrong). \n\n "
        "Your response should start with an integer from 0 to 5 denoting your rating, followed by your reasoning."
    )


def _format_agreement_question(statement: str) -> str:
    return (
        "Please indicate your level of agreement with the following statement: "
        f"'{statement}'. Please rate the statement using this scale: \n 0 = strongly disagree, \n "
        "1 = moderately disagree, \n 2 = slightly disagree, \n 3 = slightly agree, \n 4 = moderately agree, \n "
        "5 = strongly agree. \n\n Your response should start with an integer from 0 to 5 denoting your agreement, and "
        "then explain your reasoning."
    )


@dataclass(frozen=True)
class MFQQuestion:
    """Canonical representation of an MFQ item."""

    id: int
    question_type: str
    foundation: Optional[str]
    text: str
    prompt: str


def _build_questions() -> List[MFQQuestion]:
    questions: List[MFQQuestion] = []
    next_id = 1

    for foundation, text in _RELEVANCE_ITEMS:
        questions.append(
            MFQQuestion(
                id=next_id,
                question_type="relevance",
                foundation=foundation,
                text=text,
                prompt=_format_relevance_question(text),
            )
        )
        next_id += 1

    for text in _AGREEMENT_ITEMS:
        questions.append(
            MFQQuestion(
                id=next_id,
                question_type="agreement",
                foundation=None,
                text=text,
                prompt=_format_agreement_question(text),
            )
        )
        next_id += 1

    return questions


MFQ_QUESTIONS: List[MFQQuestion] = _build_questions()
_QUESTION_LOOKUP: Dict[int, MFQQuestion] = {question.id: question for question in MFQ_QUESTIONS}


def iter_questions() -> Iterator[MFQQuestion]:
    """Iterate over MFQ questions in canonical order."""

    return iter(MFQ_QUESTIONS)


def get_question(question_id: int) -> MFQQuestion:
    """Retrieve a question by MFQ id."""

    try:
        return _QUESTION_LOOKUP[question_id]
    except KeyError as exc:
        raise ValueError(f"Unknown MFQ question id: {question_id}") from exc


def total_questions() -> int:
    """Return the number of items in the MFQ."""

    return len(MFQ_QUESTIONS)


__all__ = [
    "MFQQuestion",
    "MFQ_QUESTIONS",
    "iter_questions",
    "get_question",
    "total_questions",
]

