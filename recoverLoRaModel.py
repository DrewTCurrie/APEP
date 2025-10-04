import os
import config
from unsloth import FastModel
from peft import PeftModel

def main():
    base_model_name = config.MODEL_NAME  # same as in train.py
    adapter_path = os.path.join(config.OUTPUT_DIR, "Ada_Latest")
    merged_path = os.path.join(config.OUTPUT_DIR, "Ada_Latest_Recovered")

    print(f"🔄 Loading base model: {base_model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=config.MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=False,  # must load full precision for merging
    )

    print(f"📂 Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("🪄 Merging adapters into base model...")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {merged_path}")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print("✅ Recovery + merge complete! Model available at:", merged_path)

if __name__ == "__main__":
    main()
