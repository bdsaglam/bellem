# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/musique.singlehop.ipynb.

# %% auto 0
__all__ = ['make_docs', 'BaselineSingleHop', 'benchmark']

# %% ../../nbs/musique.singlehop.ipynb 3
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm

from .eval import compute_scores_dataframe, aggregate_scores

tqdm.pandas()

# %% ../../nbs/musique.singlehop.ipynb 4
def make_docs(example):
    ps = example["paragraphs"]
    for p in ps:
        idx = p["idx"]
        title = p["title"]
        body = p["paragraph_text"]
        is_supporting = p["is_supporting"]
        text = f"# {title}\n{body}"
        yield dict(
            text=text,
            is_supporting=is_supporting,
            parent_id=example["id"],
            idx=idx,
        )

# %% ../../nbs/musique.singlehop.ipynb 5
class BaselineSingleHop:
    def __init__(self, qa_func, retrieval_func):
        self.qa_func = qa_func
        self.retrieval_func = retrieval_func

    def _call(self, example) -> dict:
        docs = list(make_docs(example))
        question = example["question"]
        query = question
        retrieved_docs = self.retrieval_func(docs, query)
        context = "\n\n".join(doc['text'] for doc in retrieved_docs)
        qa_result = self.qa_func(context=context, question=question)
        answer = qa_result.get("answer")
        hop = {
            "question": question,
            "query" : query,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "answer": answer,
            "qa_result": qa_result,
        }
        return {'answer': answer, 'hops': [hop]}

    def __call__(self, example, ignore_errors: bool = False) -> dict:
        try:
            output = self._call(example)
        except Exception as exc:
            if ignore_errors:
                id = example["id"]
                print(f"Failed to answer the question {id}\n{exc}")
                output = dict(answer="N/A", hops=[{'error': str(exc)}])
            else:
                raise
        return output

# %% ../../nbs/musique.singlehop.ipynb 6
def benchmark(
    dataf: pd.DataFrame,
    qa_func: Callable,
    retrieval_func: Callable,
    ignore_errors: bool = False,
) -> tuple[pd.DataFrame, dict]:
    pipeline = BaselineSingleHop(qa_func, retrieval_func)

    def process(example):
        output = pipeline(example, ignore_errors=ignore_errors)
        example["predicted_answer"] = output["answer"]
        example["raw_output"] = output
        example["answers"] = [example["answer"], *example["answer_aliases"]]
        return example

    dataf = dataf.progress_apply(process, axis=1)
    dataf = compute_scores_dataframe(dataf)
    scores = aggregate_scores(dataf)
    return dataf, scores