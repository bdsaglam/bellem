# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/jerx.reward.gpt.ipynb.

# %% auto 0
__all__ = ['log', 'DEFAULT_SYSTEM_PROMPT', 'USER_PROMPT', 'DEFAULT_PROMPT_MESSAGES', 'Result', 'make_question_answer_func',
           'make_reward_func']

# %% ../../../nbs/jerx.reward.gpt.ipynb 3
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.pydantic_v1 import BaseModel, Field

from ...text.utils import fuzzy_match
from ...logging import get_logger

log = get_logger(__name__)

# %% ../../../nbs/jerx.reward.gpt.ipynb 5
DEFAULT_SYSTEM_PROMPT = """You are an expert Q&A system that is trusted around the world. You are given a question that requires multi-hop reasoning. Always answer the question using the provided context information, and not prior knowledge.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
"""

USER_PROMPT = """The context information below is provided as a set of entity-relation-entity triplets from knowledge graph.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question.
{question}
"""

DEFAULT_PROMPT_MESSAGES = ChatPromptTemplate.from_messages([
    ("system", DEFAULT_SYSTEM_PROMPT), 
    ("user", USER_PROMPT),
])


class Result(BaseModel):
    """Data model for answering the question."""
    answer: str = Field(description="The answer to the question in 2-4 words.")
    reasoning: str = Field(description="Multi-hop reasoning for the answer.")


def make_question_answer_func(llm: ChatOpenAI | None = None):
    if llm is None:
        llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

    chain = create_structured_output_runnable(Result, llm=llm, prompt=DEFAULT_PROMPT_MESSAGES)
    def func(context: str, question: str) -> Result:
        return chain.invoke(dict(context=context, question=question))
    return func

# %% ../../../nbs/jerx.reward.gpt.ipynb 6
def make_reward_func(llm: ChatOpenAI | None = None, answer_comparator=fuzzy_match):
    qa = make_question_answer_func(llm)

    def reward(context: str, question: str, answers: list[str]) -> float:
        result = qa(context, question)
        correct = any(answer_comparator(result.answer, answer) for answer in answers)
        return int(correct)

    return reward