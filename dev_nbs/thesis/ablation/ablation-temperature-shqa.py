from datetime import datetime
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
import pandas as pd
from functools import partial
from bellek.musique.constants import ABLATION_RECORD_IDS
from bellek.musique.qa import answer_question_standard, answer_question_cte
from bellek.utils import set_seed
from bellek.musique.singlehop import benchmark

tqdm.pandas()
pd.options.display.float_format = "{:,.3f}".format

set_seed(89)

load_dotenv()


def perfect_retrieval_func(docs, query):
    return [doc for doc in docs if doc["is_supporting"]]


suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
RESULTS_FILE = Path(f"./ablation-temperature-shqa-results-{suffix}.jsonl")
REPORT_FILE = Path(f"./ablation-temperature-shqa-report-{suffix}.jsonl")

df = pd.read_json("../../data/generated/musique-evaluation/dataset.jsonl", orient="records", lines=True)
df = df.set_index("id", drop=False).loc[ABLATION_RECORD_IDS].copy().reset_index(drop=True)


N_RUNS = 1

results = []

for temperature in tqdm(
    [
        0.0,
        0.1,
        0.3,
        0.5,
        0.7,
        1.0,
        1.5,
        2.0,
    ]
):
    completion_kwargs = {"temperature": temperature, "max_tokens": 1000}
    for qa_technique, qa_func in [
        ("Standard", answer_question_standard),
        ("CTE", answer_question_cte),
    ]:
        qa_func = partial(qa_func, completion_kwargs=completion_kwargs)
        for run in range(1, N_RUNS + 1):
            _, scores = benchmark(df, qa_func, perfect_retrieval_func, ignore_errors=False)
            results.append(
                {
                    **scores,
                    "Context": "paragraphs",
                    "Retrieval": "Perfect",
                    "Prompting": qa_technique,
                    "Temperature": temperature,
                    "Run": run,
                }
            )
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(results[-1]) + "\n")

# # Report
report_df = pd.DataFrame.from_records(
    results,
    columns=["Context", "Retrieval", "Prompting", "Temperature", "Run", "exact_match", "f1"],
)
report_df.to_json(REPORT_FILE, orient="records", lines=True)
