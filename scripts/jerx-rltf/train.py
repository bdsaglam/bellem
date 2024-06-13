import pandas as pd
import torch
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from bellek.hf.datasets.utils import load_datasets
from bellek.hf.transformers.experiment import preprocess_config
from bellek.hf.transformers.generation import preprocess_generation_params
from bellek.hf.transformers.llama3 import prepare_llama3_for_inference
from bellek.jerx.reward.gpt import make_reward_func
from bellek.logging import get_logger
from bellek.utils import NestedDict, flatten_dict, set_seed

load_dotenv()

log = get_logger(__name__)

SEED = 42
set_seed(SEED)


def compute_heuristic_reward(generation: str, delimiter: str = "|") -> float:
    lines = generation.splitlines()
    if len(lines) < 2:
        return 0

    triplets = set([line for line in lines if len(line.split(delimiter)) == 3])
    if len(triplets) > 30:
        return 0

    entities = set()
    relations = set()
    for triplet in triplets:
        subj, relation, obj = triplet.split(delimiter)
        entities.add(subj.strip())
        entities.add(obj.strip())
        relations.add(relation.strip())

    reward = 0
    if len(entities) > 5:
        reward += 0.3
    if len(relations) > 5:
        reward += 0.5

    if len(triplets) > 5:
        reward += 0.1
    elif len(triplets) >= 3:
        reward += 0.05

    if (len(triplets) / len(lines)) > 0.8:
        reward += 0.05

    return reward


class RewardTracker:
    def __init__(self):
        self.records = []
        self.reward_func = make_reward_func()

    def compute_rewards(self, batch: list[dict]) -> list[float]:
        rewards = []
        for generation, question, answers, id in zip(
            batch["generation"],
            batch["question"],
            batch["answers"],
            batch["id"],
        ):
            answers = [answer.strip() for answer in answers.split(";")]
            reward = self.compute_reward(generation=generation, question=question, answers=answers, id=id)
            rewards.append(reward)
        return rewards

    def compute_reward(self, *, generation: str, question: str, answers: list[str], id: str | None = None) -> float:
        qa_asmt = self.reward_func(context=generation, question=question, answers=answers)
        qa_reward = qa_asmt.reward
        heuristic_reward = compute_heuristic_reward(generation)
        reward = min(0.8 * qa_reward + 0.2 * heuristic_reward, 1.0)
        result = {
            "id": id,
            "generation": generation,
            "reward": reward,
            "heuristic_reward": heuristic_reward,
            "qa_reward": qa_reward,
            "question": question,
            "answers": answers,
            "answer": qa_asmt.answer,
            "reasoning": qa_asmt.reasoning,
        }
        self.records.append(result)
        return reward

    def report(self):
        return pd.DataFrame.from_records(self.records)


def run_experiment(wandb_run):
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)

    model_id = config.at("pretrained_model.model_name_or_path")
    quantization_config = BitsAndBytesConfig(**config.at("pretrained_model.quantization_config"))
    lora_config = LoraConfig(**config.at("trainer.lora"))

    # load tokenizer and model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        peft_config=lora_config,
        attn_implementation="flash_attention_2",
        # device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prepare_llama3_for_inference(tokenizer, model)

    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id

    def preprocess_example(example):
        query = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        answers = ";".join(example["answers"])
        return {"query": query, "answers": answers}

    train_ds = load_datasets(config.at("dataset.train")).map(preprocess_example)

    def tokenize(example):
        return tokenizer(
            example["query"],
            padding="longest",
            # max_length=1055,
            # add_special_tokens=False,
        )

    tokenized_train_ds = train_ds.map(tokenize, batched=True)
    tokenized_train_ds[0].keys()

    # token_counts = pd.Series([len(example["input_ids"]) for example in tokenized_train_ds])
    # token_counts.describe()

    reward_tracker = RewardTracker()

    ppo_config = PPOConfig(
        seed=config["seed"],
        **config.at("trainer.config"),
    )

    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=None,
        config=ppo_config,
        dataset=tokenized_train_ds,
        tokenizer=tokenizer,
    )

    # dl = ppo_trainer.prepare_dataloader(tokenized_train_ds.remove_columns(['answers']))
    # for batch in dl:
    #     print(batch.keys())
    #     print(np.array(batch['input_ids']).shape)
    #     input_tensors = list(torch.unbind(torch.stack(batch['input_ids']).transpose(1, 0)))
    #     print(np.array(input_tensors).shape)

    generation_params = preprocess_generation_params(
        tokenizer,
        config.at("inference.generation_params", {}),
    )

    response_template = config.at("trainer.response_template")
    assert response_template

    for batch in tqdm(ppo_trainer.dataloader):
        input_tensors = list(torch.unbind(torch.stack(batch["input_ids"]).transpose(1, 0)))

        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(input_tensors, **generation_params)
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=False) for r in response_tensors]

        #### Compute reward score
        batch["generation"] = [response.rsplit(response_template, 1)[-1].strip() for response in batch["response"]]
        rewards = reward_tracker.compute_rewards(batch)

        #### Run PPO step
        stats = ppo_trainer.step(input_tensors, response_tensors, [torch.tensor(reward) for reward in rewards])
        ppo_trainer.log_stats(stats, batch, rewards)

    # Push model to hub
    final_model_id = config.at("hfhub.model_id")
    ppo_trainer.model.push_to_hub(final_model_id)
    tokenizer.push_to_hub(final_model_id)

    # Save rewards
    df = reward_tracker.report()
    df.to_json("jerx-rltf-rewards.jsonl", lines=True, orient="records")
    df.describe()
