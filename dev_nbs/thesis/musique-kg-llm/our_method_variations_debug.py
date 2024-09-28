import json
import logging
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import bm25s
from tenacity import retry, stop_after_attempt, wait_random_exponential
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from bellek.musique.constants import ABLATION_RECORD_IDS
from bellek.musique.singlehop import benchmark as benchmark_single
from bellek.musique.multihop import benchmark as benchmark_multi
from bellek.musique.qa import answer_question_cte, answer_question_standard
from bellek.utils import set_seed

load_dotenv()

tqdm.pandas()
pd.options.display.float_format = "{:,.3f}".format

set_seed(89)

logging.getLogger("bm25s").setLevel(logging.ERROR)

model = SentenceTransformer("all-MiniLM-L6-v2")

suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
RESULTS_FILE = Path(f"./our-method-results-{suffix}.jsonl")
REPORT_FILE = Path(f"./our-method-report-{suffix}.jsonl")


df = pd.read_json("../../data/generated/musique-evaluation/dataset.jsonl", orient="records", lines=True)
df = df.set_index("id", drop=False).loc[ABLATION_RECORD_IDS].copy().reset_index(drop=True)


qd_df = pd.read_json(
    "../../data/generated/musique-evaluation/question-decomposition.jsonl",
    orient="records",
    lines=True,
)
df = pd.merge(df.drop(columns=["question", "question_decomposition"]), qd_df, on="id", suffixes=("", ""))


# Retrieval functions


def bm25_retrieval(docs: list[dict], query: str, top_k: int):
    top_k = min(top_k, len(docs))
    retriever = bm25s.BM25(corpus=docs)
    tokenized_corpus = bm25s.tokenize([doc["text"] for doc in docs], show_progress=False)
    retriever.index(tokenized_corpus, show_progress=False)
    results, _ = retriever.retrieve(bm25s.tokenize(query), k=top_k, show_progress=False)
    return results[0].tolist()


def semantic_retrieval(docs: list[dict], query: str, top_k: int):
    embeddings = model.encode([doc["text"] for doc in docs])
    query_vectors = model.encode([query])
    similarities = model.similarity(embeddings, query_vectors)
    sorted_indices = similarities.argsort(dim=0, descending=True)
    return [docs[i] for i in sorted_indices[:top_k]]


def dummy_retrieval(docs: list[dict], query: str, top_k: int):
    return docs


def perfect_retrieval(docs: list[dict], query: str, top_k: int):
    return [doc for doc in docs if doc["is_supporting"]]


results = []

# Parameters
N_RUNS = 1

qa_retry_deco = retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=30))

# Hyperparamaters
qdecomp_params = [
    (False, benchmark_single),
    (True, benchmark_multi),
]

prompting_params = [
    ("Standard", answer_question_standard),
    ("CTE", answer_question_cte),
]

for run in range(1, N_RUNS + 1):
    for qdecomp, benchmark in qdecomp_params:
        for qa_technique, qa_func in prompting_params:
            for retriever_name, retriever, top_ks in [
                ("Sparse", bm25_retrieval, [3, 5, 10]),
                ("Dense", semantic_retrieval, [3, 5, 10]),
            ]:
                for top_k in top_ks:
                    _, scores = benchmark(
                        df,
                        qa_retry_deco(qa_func),
                        partial(retriever, top_k=top_k),
                    )
                    results.append(
                        {
                            **scores,
                            "qdecomp": qdecomp,
                            "context": "paragraphs",
                            "retrieval": retriever_name,
                            "top_k": top_k,
                            "qa": qa_technique,
                            "run": run,
                        }
                    )
                    # with open(RESULTS_FILE, "a") as f:
                    #     f.write(json.dumps(results[-1]) + "\n")

# # Report
report_df = pd.DataFrame.from_records(
    results,
    columns=[
        "qdecomp",
        "context",
        "retrieval",
        "top_k",
        "qa",
        "run",
        "exact_match",
        "f1",
    ],
)
report_df.to_json(REPORT_FILE, orient="records", lines=True)
