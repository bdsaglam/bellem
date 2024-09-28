from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from bellek.musique.qa import answer_question_standard, answer_question_cot, answer_question_cot_fs, answer_question_cte
from bellek.utils import set_seed
from bellek.musique.singlehop import benchmark
from bellek.musique.constants import ABLATION_RECORD_IDS
from tqdm.auto import tqdm

tqdm.pandas()

load_dotenv()

set_seed(89)

pd.options.display.float_format = "{:,.3f}".format


def perfect_retrieval_func(docs, query):
    return [doc for doc in docs if doc["is_supporting"]]


N_RUNS = 3


df = pd.read_json("../../data/generated/musique-common/base-dataset-validation.jsonl", orient="records", lines=True)
df = df.set_index("id", drop=False).loc[ABLATION_RECORD_IDS].copy().reset_index(drop=True)

results = []

for qa_technique, qa_func in tqdm(
    [
        ("standard", answer_question_standard),
        ("cot-zs", answer_question_cot),
        ("cot-fs", answer_question_cot_fs),
        ("cte", answer_question_cte),
    ]
):
    for run in range(1, N_RUNS + 1):
        _, scores = benchmark(df, qa_func, perfect_retrieval_func, ignore_errors=False)
        results.append(
            {
                **scores,
                "retrieval": "groundtruth",
                "context": "paragraphs",
                "qa": qa_technique,
                "run": run,
            }
        )

report_df = pd.DataFrame.from_records(results, columns=["context", "retrieval", "qa", "run", "exact_match", "f1"])

suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
report_df.to_json(f"./ablation-prompting-technique-{suffix}.jsonl", orient="records", lines=True)

report_df.drop(columns=["context", "retrieval", "run"]).groupby(["qa"]).agg(["min", "mean", "max", "std"]).to_json(
    f"./ablation-prompting-technique--agg-{suffix}.jsonl", orient="records"
)
