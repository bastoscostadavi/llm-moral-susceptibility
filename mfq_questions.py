#!/usr/bin/env python3
"""
Moral Foundations Questionnaire (MFQ-30) Questions
"""

# MFQ-30 Questions divided by moral foundations
MFQ_QUESTIONS = {
    "care_harm": [
        "Whether or not someone suffered emotionally",
        "Whether or not someone cared for someone weak or vulnerable",
        "Whether or not someone was good at math",  # irrelevant item
        "Whether or not someone acted to help another person",
        "Whether or not someone was cruel",
        "Whether or not someone was denied his or her rights"  # belongs to fairness but mixed here
    ],

    "fairness_cheating": [
        "Whether or not someone acted unfairly",
        "Whether or not someone was denied his or her rights",
        "Whether or not someone acted in a way that God would approve of",  # belongs to sanctity
        "Whether or not someone was treated differently than others",
        "Whether or not someone was good at math",  # irrelevant item
        "Whether or not some people were treated differently than others"
    ],

    "loyalty_betrayal": [
        "Whether or not someone's action showed love for his or her country",
        "Whether or not someone showed a lack of loyalty",
        "Whether or not someone did something to betray his or her group",
        "Whether or not someone was good at math",  # irrelevant item
        "Whether or not someone showed a lack of loyalty to his or her group",
        "Whether or not someone acted in a way that was disloyal to his or her group"
    ],

    "authority_subversion": [
        "Whether or not someone conformed to the traditions of society",
        "Whether or not someone acted in a way that God would approve of",  # belongs to sanctity
        "Whether or not someone showed a lack of respect for authority",
        "Whether or not someone was good at math",  # irrelevant item
        "Whether or not someone acted in a way that undermined authority",
        "Whether or not someone showed disrespect for authority"
    ],

    "sanctity_degradation": [
        "Whether or not someone violated standards of purity and decency",
        "Whether or not someone was good at math",  # irrelevant item
        "Whether or not someone acted in a way that God would approve of",
        "Whether or not someone acted in a way that was disgusting",
        "Whether or not someone did something disgusting",
        "Whether or not someone violated standards of purity"
    ]
}

# Relevance ratings (how relevant each consideration is when deciding whether something is right or wrong)
RELEVANCE_QUESTIONS = [
    "When you decide whether something is right or wrong, to what extent are the following considerations relevant to your thinking?",
    # Then each question above would be rated on a scale
]

# Agreement questions about moral principles
AGREEMENT_QUESTIONS = [
    "Compassion for those who are suffering is the most crucial virtue.",
    "When the government makes laws, the number one principle should be ensuring that everyone is treated fairly.",
    "I am proud of my country's history.",
    "Respect for authority is something all children need to learn.",
    "People should not do things that are disgusting, even if no one is harmed.",
    "It is better to do good than to do bad.",  # irrelevant
    "One of the worst things a person could do is hurt a defenseless animal.",
    "Justice is the most important requirement for a society.",
    "People should be loyal to their family members, even when they have done something wrong.",
    "Men and women each have different roles to play in society.",
    "I would call some acts wrong on the grounds that they are unnatural.",
    "It can never be right to kill a human being.",  # irrelevant
    "I think it's morally wrong that rich children inherit a lot of money while poor children inherit nothing.",
    "It is more important to be a team player than to express oneself.",
    "If I were a soldier and disagreed with my commanding officer's orders, I would obey anyway because that is my duty.",
    "Chastity is an important and valuable virtue."
]

def get_all_mfq_questions():
    """Return all MFQ questions organized by category"""
    return {
        "relevance_questions": MFQ_QUESTIONS,
        "agreement_questions": AGREEMENT_QUESTIONS
    }

def format_relevance_question(question):
    """Format a relevance question for the LLM"""
    return f"When deciding whether something is right or wrong, how relevant is the following consideration to your thinking: '{question}'. Please rate from 0 (not at all relevant) to 5 (extremely relevant)."

def format_agreement_question(statement):
    """Format an agreement question for the LLM"""
    return f"Please indicate your level of agreement with the following statement: '{statement}'. Please rate from 0 (strongly disagree) to 5 (strongly agree)."