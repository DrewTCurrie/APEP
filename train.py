# train.py

"""
Main training script for Ada.

Features:
- Loads model and tokenizer using unsloth
- Prepares dataset with caching and multi-turn conversations
- Random example preview (multi-turn)
- LoRa application
- Optimized training args
- Full resource utilization (CUDA + CPU offloading)
"""

import unsloth
from unsloth import FastModel, FastLanguageModel, get_chat_template
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
import torch
import os
import random

from data.dataset_utils import prepare_dataset
import config


# =============================
# Helper Functions
# =============================
def cute_print(msg, color="CYAN"):
    colors = {
        "HEADER": "\033[95m",
        "PINK": "\033[95m",
        "CYAN": "\033[96m",
        "BLUE": "\033[94m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "RED": "\033[91m",
        "END": "\033[0m",
    }
    col = colors.get(color.upper(), colors["CYAN"])
    print(f"{col}{msg}{colors['END']}")


# =============================
# Main Training Function
# =============================
def main():
    # Load system prompt
    with open(config.SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read().strip()

    cute_print("🚀 Starting Ada Training", "PINK")
    print("=" * 60)

    # -----------------------------
    # Load model & tokenizer
    # -----------------------------
    cute_print("📥 Loading model and tokenizer...", "CYAN")
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(
        tokenizer, chat_template=config.REQUIRED_CHAT_TEMPLATE, map_eos_token=True
    )
    cute_print(f"🛠️ Tokenizer Applied: {config.REQUIRED_CHAT_TEMPLATE}", "CYAN")

    # -----------------------------
    # Apply LoRa
    # -----------------------------
    cute_print("🔧 Applying LoRa...", "CYAN")
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=config.LORA_R,
        target_modules=config.LORA_TARGET_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.LORA_BIAS,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        random_state=config.RANDOM_STATE,
        use_rslora=config.USE_RSLORA,
        loftq_config=config.LOFTQ_CONFIG,
    )
    model.gradient_checkpointing_enable()

    # -----------------------------
    # Prepare Dataset
    # -----------------------------
    dataset = prepare_dataset(
        path=config.TRAINING_DATA_PATH,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        personality_path=config.PERSONALITY_PATH,
        max_seq_len=config.MAX_SEQ_LEN,
        target_megachats=config.TARGET_MEGACHATS,
        mega_size=config.MEGA_SIZE,
        cache_path=os.path.join(config.OUTPUT_DIR, "cached_dataset"),
        dataset_weights={
            'generated':0.0,
            'external':0.95,
            'personality':0.05,
        }
    )
    train_test = dataset.train_test_split(
        test_size=config.TEST_SIZE, shuffle=True, seed=config.SEED
    )
    train_ds, val_ds = train_test["train"], train_test["test"]
    cute_print(f"📊 Dataset Split: Train={len(train_ds)}, Val={len(val_ds)}", "CYAN")
    # -----------------------------
    # Preview Random Multi-turn Example
    # -----------------------------
    example = random.choice(train_ds)

    # If your dataset stores messages as a list of {role, content}
    if "messages" in example:
        cute_print("🔍 Random Multi-turn Training Example Preview:", "PINK")
        for msg in example["messages"]:
            role = msg["role"].upper()
            # pick colors per role
            color = (
                "YELLOW" if role == "SYSTEM" else "CYAN" if role == "USER" else "GREEN"
            )
            cute_print(f"\n[{role}]:", color)
            print(msg["content"].strip())
    else:
        # Fallback: decode input_ids/labels if no messages field
        input_ids = example["input_ids"]
        labels = example["labels"]
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.decode(
            [i for i, l in zip(input_ids, labels) if l != -100],
            skip_special_tokens=True,
        )
        cute_print("🔍 Random Multi-turn Training Example Preview:", "PINK")
        cute_print("\n[Drew]:", "CYAN")
        print(decoded_input)
        cute_print("\n[Ada]:", "GREEN")
        print(decoded_labels)

    print("=" * 60)
    approval = input("✨ Do you approve this training example? (Y/n): ").strip().lower()
    if approval in ["n", "no"]:
        print("❌ Training halted.")
        exit(0)

    # -----------------------------
    # Training Arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        logging_steps=config.LOGGING_STEPS,
        max_steps=config.MAX_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        seed=config.SEED,
        fp16=False,
        bf16=True,  # use bf16 for RTX 40xx
        report_to=config.REPORT_TO,
        eval_strategy=config.EVAL_STRATEGY,
        eval_steps=config.EVAL_STEPS,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, padding="longest", return_tensors="pt"
    )

    # -----------------------------
    # Initialize Trainer
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=config.MAX_SEQ_LEN,
        packing=False,
        data_collator=data_collator,
    )

    # -----------------------------
    # Training Loop
    # -----------------------------
    cute_print("🔄 Starting Training...", "CYAN")
    trainer.train()

    # -----------------------------
    # Save Model
    # -----------------------------
    cute_print("💾 Saving Model...", "CYAN")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(os.path.join(config.OUTPUT_DIR, "Ada_Latest"))
    tokenizer.save_pretrained(os.path.join(config.OUTPUT_DIR, "Ada_Latest_Tokenizer"))
    model.save_pretrained_merged(
        os.path.join(config.OUTPUT_DIR, "Ada_Latest_Merged"),
        tokenizer,
        save_method="merged_16bit",
    )
    cute_print(f"✅ Model saved to: {config.OUTPUT_DIR}", "GREEN")

    # -----------------------------
    # Test Generation
    # -----------------------------
    test_prompt = "What is your name? Are you a human?"
    inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)
    response = tokenizer.decode(
        model.generate(inputs, max_length=100)[0], skip_special_tokens=True
    )
    cute_print(f"📝 Generated Response: {response}", "PINK")
    print("=" * 60)


# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    main()
