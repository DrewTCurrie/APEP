# data/dataset_utils.py
"""
Dataset preparation utilities for Ada model training.
Features:
- True multi-turn conversation grouping
- Oversampling to reach target megachats
- Personality injection
- Caching prepared dataset with full conversation for previews
"""

import os
import json
import random
from datasets import Dataset, load_dataset, load_from_disk

def prepare_dataset(
    path,
    system_prompt,
    tokenizer,
    personality_path="personality.json",
    max_seq_len=4096,
    target_megachats=500,
    mega_size=20,
    cache_path="/data/trainingData/cached_dataset",
    force_rebuild=False,
    seed=3407
):
    random.seed(seed)

    # Load from cache if available
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"📥 Loading cached dataset from {cache_path}...")
        return load_from_disk(cache_path)

    print("📦 Preparing dataset from scratch...")

    # Load main JSON dataset
    raw_data = load_dataset("json", data_files=path, split="train")
    raw_list = list(raw_data)

    # Load personality-only messages
    with open(personality_path, "r") as f:
        personality_data = json.load(f)
    personality_chats = [[{"role": "assistant", "content": msg["content"]}] for msg in personality_data]

    # === Multi-turn conversation grouping ===
    conversations = []
    current_conv = []
    for msg in raw_list:
        role = msg["role"].lower()
        current_conv.append({"role": role, "content": msg["content"]})

        # End a conversation block at assistant reply
        if role == "assistant":
            conversations.append(current_conv)
            current_conv = []

    # Combine with personality chats
    all_conversations = conversations + personality_chats

    # Oversample to reach target megachats
    megachats = []
    while len(megachats) < target_megachats:
        random.shuffle(all_conversations)
        # Build megachats in blocks of mega_size conversations
        for i in range(0, len(all_conversations), mega_size):
            block = sum(all_conversations[i:i+mega_size], [])
            megachats.append(block)
            if len(megachats) >= target_megachats:
                break

    megachats = megachats[:target_megachats]

    # === Encode chats ===
    input_ids_list, labels_list, messages_list = [], [], []
    for chat in megachats:
        # Always start with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat:
            messages.append(msg)
            if msg["role"] == "assistant":
                # Encode context for model input
                context = messages[:-1]
                prompt_ids = tokenizer.apply_chat_template(
                    context, add_generation_prompt=True, tokenize=True, return_tensors=None
                )
                if isinstance(prompt_ids[0], list):
                    prompt_ids = [tok for sublist in prompt_ids for tok in sublist]

                reply_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
                if tokenizer.eos_token_id:
                    reply_ids += [tokenizer.eos_token_id]

                full_ids = prompt_ids + reply_ids
                full_labels = [-100] * len(prompt_ids) + reply_ids

                # Truncate if too long
                if len(full_ids) > max_seq_len:
                    drop = len(full_ids) - max_seq_len
                    full_ids = full_ids[drop:]
                    full_labels = full_labels[drop:]

                input_ids_list.append(full_ids)
                labels_list.append(full_labels)
                messages_list.append(messages.copy())  # Keep full conversation for previews

    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "messages": messages_list
    })

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    dataset.save_to_disk(cache_path)
    print(f"💾 Dataset cached at {cache_path}")

    return dataset
