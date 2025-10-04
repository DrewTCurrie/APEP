"""
preflight_test.py

Quick preflight for Ada training.
Checks dataset, model, LoRa, tokenizer, and a tiny training step.
"""

import os
import torch
from unsloth import FastModel, FastLanguageModel, get_chat_template
from trainers.spinner import CuteSpinner, cute_print
from trainers.visualizer import LiveTrainingVisualizer
from trainers.callbacks import ValidationPreviewCallback
from data.dataset_utils import prepare_dataset
from config import *
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer

def main():
    cute_print("🚀 Preflight Test: Ada Training Environment", "PINK")

    # Load system prompt
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read().strip()

    # Load model and tokenizer
    cute_print("📥 Loading model and tokenizer...", "CYAN")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=REQUIRED_CHAT_TEMPLATE, map_eos_token=True)

    cute_print("🔧 Applying LoRa (tiny preflight)...", "CYAN")
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=RANDOM_STATE,
        use_rslora=USE_RSLORA,
        loftq_config=LOFTQ_CONFIG,
    )
    model.gradient_checkpointing_enable()

    # Prepare tiny dataset
    cute_print("📊 Preparing tiny dataset for preflight...", "CYAN")
    dataset = prepare_dataset(
        TRAINING_DATA_PATH,
        system_prompt,
        tokenizer,
        personality_path=PERSONALITY_PATH,
        max_seq_len=MAX_SEQ_LEN,
        target_megachats=2,  # tiny sample
        mega_size=MEGA_SIZE
    )
    train_ds = dataset
    val_ds = dataset

    # Tiny training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR+"/preflight",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        max_steps=2,
        warmup_steps=1,
        logging_steps=1,
        fp16=False,       # Disable
        bf16=True,        # Enable bfloat16
        seed=SEED,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", return_tensors="pt")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        data_collator=data_collator,
    )

    trainer.add_callback(ValidationPreviewCallback())

    # Run tiny training
    cute_print("🔄 Starting tiny training run...", "CYAN")
    spinner = CuteSpinner("Preflight training", color=SPINNER_COLOR, emoji=SPINNER_EMOJI)
    spinner.start()
    try:
        trainer.train()
    finally:
        spinner.stop()

    # Test generation
    cute_print("📝 Generating sample Ada response...", "CYAN")
    test_prompt = "Are you a human? Explain why."
    inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(inputs)
    response = tokenizer.decode(model.generate(inputs, attention_mask=attention_mask, max_length=100)[0], skip_special_tokens=True)
    cute_print(f"📝 Sample Response: {response}", "PINK")

if __name__ == "__main__":
    main()
