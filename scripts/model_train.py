from os import truncate
from pydantic import BaseModel, Field
from pydantic.types import Json
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.pipelines.base import load_model
from scripts.config import setting
from pathlib import Path
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import torch
from datasets import Dataset


def load_jsonl(filepath: Path) -> list[Json]:
    """
    This was to load the jsonl file both in trai.jsonl and validate.jsonl
    """
    lines: list = filepath.read_text(encoding="utf-8").splitlines()
    json_data: list[Json] = [json.loads(line)["text"] for line in lines if line.strip()]
    return json_data


def load_files():
    return load_jsonl(setting.envpath.train_file), load_jsonl(setting.envpath.val_file)


class Train:
    def __init__(self) -> None:
        self.train_lines: list[Json] = []
        self.vali_lines: list[Json] = []
        self.set_dataset()
        self.tokenzier = self.tokenzier_()

    def set_dataset(self):
        self.train_lines, self.vali_lines = load_files()
        print(f"Training files length :- {len(self.train_lines)} rows")
        print(f"validate files length :- {len(self.vali_lines)} rows")

    def tokenzier_(self):
        tokenzier = AutoTokenizer.from_pretrained(setting.envpath.local_path)
        if (
            tokenzier.pad_token is None
        ):  # for slm generative type pad_token was not there.
            tokenzier.pad_token = tokenzier.eos_token
            tokenzier.pad_token_id = tokenzier.eos_token_id

        if isinstance(tokenzier, GPT2Tokenizer):
            return tokenzier
        raise Exception("critical: some issue with tokenization")

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            setting.envpath.local_path, dtype=torch.float32
        )
        if isinstance(model, LlamaForCausalLM):
            return model
        raise Exception("Critical: Some issues with loading model")

    def applying_lora(self) -> PeftModel:
        """
        Narrow fine tunning the slm,
        mainly concentrate on the q_proj and v_proj layers
        just to control in and out with out disturbing remaning general knowledge layers.
        """
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=setting.train.lora_r,
            lora_alpha=setting.train.lora_alpha,
            lora_dropout=0.0,  # no clue but standard for slm
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(self.load_model(), lora_config)
        model.print_trainable_parameters()
        return model

    def text_tokens(self, texts: list[Json]) -> Dataset:
        """
        tokenize the train/validate text
        """
        tokenized = self.tokenzier(
            texts,
            truncation=True,
            max_length=setting.train.max_length,
            padding="max_length",
        )
        # check how many sample was truncated
        original_len = [len(self.tokenzier.encode(t)) for t in texts]
        truncated = sum(1 for l in original_len if l > setting.train.max_length)
        if truncated:
            print(f"{truncated} sample exceed than {setting.train.max_length}")
        else:
            print("all sample are fitted in the max_length")
        return Dataset.from_dict(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"],
            }
        )

    def run(self):
        train_dataset = self.text_tokens(self.train_lines)
        vali_dataset = self.text_tokens(self.vali_lines)
        training_arguments = TrainingArguments(
            output_dir=setting.envpath.output_dir,
            num_train_epochs=setting.train.epochs,
            per_device_train_batch_size=setting.train.batch_size,
            per_device_eval_batch_size=setting.train.batch_size,
            learning_rate=setting.train.learning_rate,
            warmup_steps=10,
            gradient_accumulation_steps=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=5,
            fp16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
        )
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
        train_model = self.applying_lora()
        trainer = Trainer(
            model=train_model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=vali_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenzier, mlm=False),
            callbacks=[early_stopping],
        )
        trainer.train()
        # after training merge the weights
        assert isinstance(trainer.model, PeftModel)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(setting.envpath.output_dir)
        self.tokenzier.save_pretrained(setting.envpath.output_dir)
        print(
            f"trained model saved into the {setting.envpath.output_dir.absolute} path"
        )
        print(f"Delete the model with :rm -rf <path>")

        log_path = setting.envpath.workspace / "training_log.json"
        logs = [
            entry
            for entry in trainer.state.log_history
            if "loss" in entry or "eval_loss" in entry
        ]
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2)

        print("log was saved")
