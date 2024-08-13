import json
import logging
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import bm25s
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from bellek.musique.constants import ABLATION_RECORD_IDS
from bellek.musique.multihop import benchmark as benchmark_multi
from bellek.musique.singlehop import benchmark as benchmark_single
from bellek.qa.ablation import answer_question_cte, answer_question_standard
from bellek.utils import set_seed

load_dotenv()

tqdm.pandas()
pd.options.display.float_format = "{:,.3f}".format
set_seed(89)

logging.getLogger("bm25s").setLevel(logging.ERROR)

model = SentenceTransformer("all-MiniLM-L6-v2")

suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

N_RUNS = 3
SAMPLE_SIZE = 1


df = pd.read_json("../../data/generated/musique-evaluation/dataset.jsonl", orient="records", lines=True)
df.set_index("id", inplace=True, drop=False)
df = df.loc[ABLATION_RECORD_IDS].copy().reset_index(drop=True)
df = df.head(SAMPLE_SIZE)


qd_df = pd.read_json(
    "../../data/generated/musique-evaluation/question-decomposition.jsonl", orient="records", lines=True
)
df = pd.merge(df.drop(columns=["question", "question_decomposition"]), qd_df, on="id", suffixes=("", ""))


jerx_file = Path("../../data/raw/musique-evaluation/jerx-inferences/llama3-base.jsonl")
jerx_df = pd.read_json(jerx_file, lines=True)
jerx_df.head()


jerx_mapping = {(row["id"], row["paragraph_idx"]): row["generation"] for _, row in jerx_df.iterrows()}


def extract_triplets(example: dict):
    example["triplets_str"] = [jerx_mapping[(example["id"], p["idx"])].strip() for p in example["paragraphs"]]
    return example


df = df.apply(extract_triplets, axis=1)


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

# Hyperparamaters
# qdecomp_params = [(False, benchmark_single), (True, benchmark_multi)]
qdecomp_params = [(True, benchmark_multi)]

# ## Only paragraphs


for run in range(1, N_RUNS + 1):
    for qdecomp, benchmark in qdecomp_params:
        for qa_technique, qa_func in [("standard", answer_question_standard), ("cte", answer_question_cte)]:
            for retriever_name, retriever, top_ks in [
                ("bm25", bm25_retrieval, [3, 5, 10]),
                ("semantic", semantic_retrieval, [3, 5, 10]),
                ("dummy", dummy_retrieval, [20]),
                ("perfect", perfect_retrieval, [2]),
            ]:
                for top_k in top_ks:
                    _, scores = benchmark(df, qa_func, partial(retriever, top_k=top_k), ignore_errors=True)
                    results.append(
                        {
                            **scores,
                            "retrieval": retriever_name,
                            "top_k": top_k,
                            "context": "paragraphs",
                            "qa": qa_technique,
                            "qdecomp": qdecomp,
                            "run": run,
                        }
                    )

with open(f"./our-method-results-{suffix}-1.jsonl", "w") as f:
    f.write(json.dumps(results, indent=2))

# ## Paragraphs + Triplets


def enhance_paragraphs(row):
    paragraphs_with_triplets = []
    for p in row["paragraphs"]:
        p = deepcopy(p)
        triplets_str = str(jerx_mapping[(row["id"], p["idx"])])
        p["paragraph_text"] = "\n".join([p["paragraph_text"], "# Entity-relation-entity triplets", triplets_str])
        paragraphs_with_triplets.append(p)
    row["paragraphs"] = paragraphs_with_triplets
    return row


df_paragraph_triplets = df.apply(enhance_paragraphs, axis=1)


for run in range(1, N_RUNS + 1):
    for qdecomp, benchmark in qdecomp_params:
        for qa_technique, qa_func in [("standard", answer_question_standard)]:
            for retriever_name, retriever, top_ks in [
                ("bm25", bm25_retrieval, [3, 5, 10]),
                ("semantic", semantic_retrieval, [3, 5, 10]),
                ("dummy", dummy_retrieval, [20]),
                ("perfect", perfect_retrieval, [2]),
            ]:
                for top_k in top_ks:
                    _, scores = benchmark(
                        df_paragraph_triplets, qa_func, partial(retriever, top_k=top_k), ignore_errors=True
                    )
                    results.append(
                        {
                            **scores,
                            "retrieval": retriever_name,
                            "top_k": top_k,
                            "context": "paragraphs+triplets",
                            "qa": qa_technique,
                            "qdecomp": qdecomp,
                            "run": run,
                        }
                    )

with open(f"./our-method-results-{suffix}-2.jsonl", "w") as f:
    f.write(json.dumps(results, indent=2))

# ## Only triplets


def replace_paragraphs(row):
    paragraphs_with_triplets = []
    for p in row["paragraphs"]:
        triplets_str = str(jerx_mapping[(row["id"], p["idx"])])
        for triplet in triplets_str.splitlines():
            p = deepcopy(p)
            p["title"] = ""
            p["paragraph_text"] = triplet.strip()
            paragraphs_with_triplets.append(p)
    row["paragraphs"] = paragraphs_with_triplets
    return row


df_only_triplets = df.apply(replace_paragraphs, axis=1)


for run in range(1, N_RUNS + 1):
    for qdecomp, benchmark in qdecomp_params:
        for qa_technique, qa_func in [("standard", answer_question_standard)]:
            for retriever_name, retriever, top_ks in [
                ("bm25", bm25_retrieval, [3, 5, 10]),
                ("semantic", semantic_retrieval, [3, 5, 10]),
                ("dummy", dummy_retrieval, [20]),
                ("perfect", perfect_retrieval, [2]),
            ]:
                for top_k in top_ks:
                    top_k_effective = top_k * 7
                    _, scores = benchmark(
                        df_only_triplets, qa_func, partial(retriever, top_k=top_k_effective), ignore_errors=True
                    )
                    results.append(
                        {
                            **scores,
                            "retrieval": retriever_name,
                            "top_k": top_k,
                            "context": "triplets",
                            "qa": qa_technique,
                            "qdecomp": qdecomp,
                            "run": run,
                        }
                    )

with open(f"./our-method-results-{suffix}-3.jsonl", "w") as f:
    f.write(json.dumps(results, indent=2))

# # Report
report_df = pd.DataFrame.from_records(
    results,
    columns=["qdecomp", "context", "retrieval", "top_k", "qa", "run", "exact_match", "f1"],
)
report_df.to_json(f"./our-method-report-{suffix}.jsonl", orient="records", lines=True)
