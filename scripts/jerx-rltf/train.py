import magentic
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed

from bellem.hf.datasets.utils import load_datasets
from bellem.hf.transformers.experiment import preprocess_config
from bellem.hf.transformers.generation import preprocess_generation_params
from bellem.hf.transformers.llama3 import prepare_llama3_for_inference
from bellem.hf.transformers.utils import prepare_model_kwargs
from bellem.jerx.reward.heuristic import compute_heuristic_reward
from bellem.jerx.reward.llm import assess_quality
from bellem.jerx.reward.qa import make_qa_reward_func
from bellem.logging import get_logger
from bellem.ml.experiment import main
from bellem.utils import NestedDict, flatten_dict

load_dotenv()

log = get_logger(__name__)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


class RewardTracker:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.records = []
        self.qa_reward_func = make_qa_reward_func(model_name, completion_kwargs=dict(max_tokens=2048))
        self.quality_assesment_model = magentic.OpenaiChatModel(model_name)

    def compute_rewards(self, batch: list[dict]) -> list[float]:
        rewards = []
        for generation, question, answers, messages, id in zip(
            batch["generation"],
            batch["question"],
            batch["answers"],
            batch["messages"],
            batch["id"],
        ):
            document = messages[-1]["content"]
            answers = [answer.strip() for answer in answers.split(";")]
            reward = self.compute_reward(
                generation=generation, document=document, question=question, answers=answers, id=id
            )
            rewards.append(reward)
        return rewards

    def compute_reward(
        self, *, generation: str, document: str, question: str, answers: list[str], id: str | None = None
    ) -> float:
        # Heuristic reward
        heuristic_reward = compute_heuristic_reward(generation)
        if heuristic_reward < 0.2:
            result = {
                "id": id,
                "generation": generation,
                "reward": heuristic_reward,
                "heuristic_reward": heuristic_reward,
                "qa_reward": 0,
                "quality_reward": 0,
                "question": question,
                "answers": answers,
                "answer": "N/A",
                "reasoning": "N/A",
            }
            self.records.append(result)
            return heuristic_reward

        # QA reward
        triplets_str = self.preprocess_generation(generation)
        qa_asmt = self.qa_reward_func(
            context=triplets_str,
            question=question,
            answers=answers,
        )
        qa_reward = qa_asmt.reward

        # Quality assessment reward
        with self.quality_assesment_model:
            quality_asmt = assess_quality(document, triplets_str)
        quality_reward = quality_asmt.reward

        # Combine rewards
        reward = min(0.3 * quality_reward + 0.5 * qa_reward + 0.2 * heuristic_reward, 1.0)

        result = {
            "id": id,
            "generation": generation,
            "reward": reward,
            "heuristic_reward": heuristic_reward,
            "qa_reward": qa_reward,
            "quality_reward": quality_reward,
            "question": question,
            "answers": answers,
            "answer": qa_asmt.answer,
            "reasoning": qa_asmt.reasoning,
        }
        self.records.append(result)
        return reward

    def as_dataframe(self):
        return pd.DataFrame.from_records(self.records)

    def preprocess_generation(self, generation: str) -> str:
        valid_lines = set()
        for line in generation.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.count("|") != 2:
                continue
            valid_lines.add(line)
        return "\n".join(valid_lines)


def run_experiment(wandb_run):
    config = preprocess_config(NestedDict.from_flat_dict(wandb_run.config))
    wandb_run.config.update(flatten_dict(config), allow_val_change=True)

    set_seed(config["seed"])

    # load tokenizer and model
    pretrained_model_config = config["pretrained_model"]
    model_id = pretrained_model_config.pop("model_name_or_path")
    model_kwargs = prepare_model_kwargs(pretrained_model_config)
    model_kwargs["peft_config"] = LoraConfig(**config.at("trainer.lora"))
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prepare_llama3_for_inference(tokenizer, model)

    # Dataset
    def preprocess_example(example):
        query = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        answers = ";".join(example["answers"])
        return {"query": query, "answers": answers}

    train_ds = load_datasets(config.at("dataset.train")).map(preprocess_example)

    def tokenize(example):
        return tokenizer(example["query"])

    tokenized_train_ds = train_ds.map(tokenize, batched=False)
    tokenized_train_ds.set_format(type="torch")

    token_counts = pd.Series([len(example["input_ids"]) for example in tokenized_train_ds])
    print(token_counts.describe())

    # Trainer
    reward_tracker = RewardTracker(config.at("reward.model_name", "gpt-3.5-turbo"))

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
        data_collator=collator,
    )

    generation_params = preprocess_generation_params(
        tokenizer,
        config.at("inference.generation_params", {}),
    )

    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        response_tensors = []
        for query_tensor in query_tensors:
            response_tensor = ppo_trainer.generate(query_tensor, return_prompt=False, **generation_params)
            response_tensors.append(response_tensor.squeeze())

        batch["response"] = [tokenizer.decode(t) for t in response_tensors]

        #### Compute reward score
        batch["generation"] = batch["response"]
        rewards = reward_tracker.compute_rewards(batch)

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, [torch.tensor(reward) for reward in rewards])
        ppo_trainer.log_stats(stats, batch, rewards)

    # Push model to hub
    final_model_id = config.at("hfhub.model_id")
    ppo_trainer.model.push_to_hub(final_model_id)
    tokenizer.push_to_hub(final_model_id)

    # Save rewards
    df = reward_tracker.as_dataframe()
    df.to_json("jerx-rltf-records.jsonl", lines=True, orient="records")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.json")
    args, _ = parser.parse_known_args()

    main(run_experiment, args)
