%pip install transformers datasets peft accelerate bitsandbytes

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("training.log")  # Save to file
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
TRAIN_FILE = "./datasets/train_data.json"
TEST_FILE = "./datasets/test_data.json"
OUTPUT_DIR = "./results"

def load_datasets(train_path, test_path):
    logger.info("Loading datasets...")
    try:
        data_files = {"train": train_path, "test": test_path}
        dataset = load_dataset("json", data_files=data_files)
        logger.info("Datasets loaded successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise

def tokenize_dataset(dataset, tokenizer):
    logger.info("Tokenizing dataset...")
    try:
        def preprocess(example):
            return tokenizer(
                example["input"],
                text_target=example["output"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        tokenized = dataset.map(preprocess, batched=True)
        logger.info("Tokenization complete.")
        return tokenized
    except Exception as e:
        logger.error(f"Failed during tokenization: {e}")
        raise

def create_model():
    logger.info(f"Loading model and tokenizer from {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",
            device_map="auto"
        )

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
    except Exception as e:
        logger.error(f"Failed to initialize model/tokenizer: {e}")
        raise

def main():
    try:
        logger.info("Starting fine-tuning script...")

        dataset = load_datasets(TRAIN_FILE, TEST_FILE)
        tokenizer, model = create_model()
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_steps=25,
            fp16=True,
            report_to="none",
            ddp_find_unused_parameters=False
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        logger.info("Beginning training...")
        trainer.train()
        logger.info("Training completed successfully.")

        logger.info("Saving final model...")
        model.save_pretrained(f"{OUTPUT_DIR}/final_model")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
        logger.info("Model saved at %s/final_model", OUTPUT_DIR)

    except Exception as e:
        logger.exception("An error occurred during the fine-tuning process.")

if __name__ == '__main__':
    main()