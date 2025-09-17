#!/usr/bin/env python

import os
import logging
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# ─── 1) Fixed scratch directory ───────────────────────────────
SCRATCH_DIR = "/scratch/test.local/huser/user"
os.makedirs(SCRATCH_DIR, exist_ok=True)
os.chdir(SCRATCH_DIR)

# ─── 2) Optional CUDA debugging ───────────────────────────────
os.environ['CUDA_LAUNCH_BLOCKING']         = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF']      = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM']       = 'false'  # silence the fork warning

# ─── 3) Logging into scratch/training.log ────────────────────
LOG_FILE = os.path.join(SCRATCH_DIR, "training.log")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# ─── 4) Constants ─────────────────────────────────────────────
MODEL_ID   = "codellama/CodeLlama-7b-Instruct-hf"
TRAIN_FILE = os.path.expanduser("~/testfinetune/code/datasets/train_data_for_codellama.jsonl")
TEST_FILE  = os.path.expanduser("~/testfinetune/code/datasets/test_data_for_codellama.jsonl")

OUTPUT_DIR = os.path.join(SCRATCH_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH = 512
SEED       = 42

# ─── 5) Seed everything ────────────────────────────────────────
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

def load_datasets(train_path, test_path):
    logger.info("Loading datasets...")
    dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    logger.info("Datasets loaded successfully.")
    return dataset

def tokenize_dataset(dataset, tokenizer):
    logger.info("Tokenizing dataset...")

    def preprocess(example):
        text = example.get("text", "")
        if isinstance(text, list):
            text = " ".join(map(str, text))
        tok = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"], "labels": tok["input_ids"]}

    tokenized = dataset.map(
        preprocess,
        batched=False,
        remove_columns=dataset["train"].column_names
    )
    logger.info("Tokenization complete.")
    return tokenized

def create_model():
    logger.info(f"Loading model and tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # place the 4bit model directly on the local GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": device},       # load the entire model on this GPU
        torch_dtype=torch.float16
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    logger.info("Model and tokenizer initialized.")
    return tokenizer, model

def run_training():
    logger.info("Starting fine-tuning...")
    dataset = load_datasets(TRAIN_FILE, TEST_FILE)
    tokenizer, model = create_model()
    tokenized = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=3,
        logging_steps=25,
        fp16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer       = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator
    )

    logger.info("Training begins...")
    trainer.train()
    logger.info("Training completed.")

    model_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info("Model saved to: %s", model_dir)

if __name__ == "__main__":
    run_training()
