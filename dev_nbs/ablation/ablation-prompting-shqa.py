# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import json
import pandas as pd
from pathlib import Path
from copy import deepcopy
from functools import partial

from bellek.musique.qa import answer_question_standard, answer_question_cot, answer_question_cot_fs, answer_question_cte
from bellek.utils import set_seed, jprint
from bellek.musique.singlehop import benchmark

set_seed(89)

# %%
from tqdm.auto import tqdm
tqdm.pandas()

# %%
pd.options.display.float_format = '{:,.3f}'.format

# %%
def perfect_retrieval_func(docs, query):
    return [doc for doc in docs if doc['is_supporting']]

# %%
N_RUNS = 3

# %%
from bellek.musique.constants import ABLATION_RECORD_IDS

df = pd.read_json('../../data/generated/musique-common/base-dataset-validation.jsonl', orient='records', lines=True)
df = df.set_index('id', drop=False).loc[ABLATION_RECORD_IDS].copy().reset_index(drop=True)
# df = df.sample(10)

print(df.shape)
df.head()

# %%
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

# %% [markdown]
# # Report

# %%
report_df = pd.DataFrame.from_records(results, columns=['context', 'retrieval', 'qa', 'run', 'exact_match', 'f1'])
report_df

# %%
from datetime import datetime
suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
report_df.to_json(f'./ablation-prompting-technique-{suffix}.jsonl', orient='records', lines=True)

# %%
report_df.drop(columns=['context', 'retrieval', 'run']).groupby(['qa']).agg(['min', 'mean', 'max', 'std'])

# %% [markdown]
# ## Inspect

# %%
fail_mask = ~(df_cte['fuzzy_match'])

# %%
df_cte.loc[fail_mask]['predicted_answer']

# %%
run = 3
row = df_cte.loc[fail_mask].iloc[run]

print("="*80)
print(row['question'])
print(row['answers'])

print("="*80)
jprint(row['raw_output'])

# %%



