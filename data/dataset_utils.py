# data/dataset_utils.py
"""
Dataset preparation utilities for Ada model training.
Features:
- Multi-dataset loading (generated + DailyDialog + EmpatheticDialogues + PersonaChat + TopicalChat)
- True multi-turn conversation grouping
- Oversampling to reach target megachats
- Personality injection
- Caching prepared dataset with full conversation for previews
"""

import os
import json
import random
from datasets import Dataset, load_dataset, load_from_disk
from pathlib import Path


def load_external_conversations(external_data_dir="data/trainingData"):
    """
    Load external conversation datasets from JSON files.
    Each JSON should contain: [{"conversations": [{"role": "user/assistant", "content": "..."}]}, ...]
    
    Returns:
        list: Conversations in format [[{"role": "...", "content": "..."}], ...]
    """
    external_data_path = Path(external_data_dir)
    all_conversations = []
    
    dataset_files = {
        'dailydialog.json': 'DailyDialog',
        'empathetic_dialogues.json': 'EmpatheticDialogues',
        'personachat.json': 'PersonaChat',
        'topical_chat.json': 'TopicalChat'
    }
    
    print("\n📚 Loading external datasets...")
    
    for filename, dataset_name in dataset_files.items():
        filepath = external_data_path / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Data format: [{"conversations": [{"role": "user", "content": "..."}, ...]}, ...]
                if isinstance(data, list):
                    # Extract the conversations list from each item
                    for item in data:
                        if "conversations" in item and isinstance(item["conversations"], list):
                            # Convert to your format: just the list of messages
                            all_conversations.append(item["conversations"])
                    
                    print(f"  ✓ {dataset_name}: {len(data)} conversations")
                else:
                    print(f"  ✗ {dataset_name}: Unexpected format")
                    
            except Exception as e:
                print(f"  ✗ {dataset_name}: Failed to load - {e}")
        else:
            print(f"  ⚠ {dataset_name}: Not found, skipping")
    
    print(f"  📊 Total external conversations loaded: {len(all_conversations)}\n")
    return all_conversations


def prepare_dataset(
    path,
    system_prompt,
    tokenizer,
    personality_path="personality.json",
    max_seq_len=4096,
    target_megachats=500,
    mega_size=20,
    cache_path="/home/drew/Documents/AI_Hobby/training/AdaTraining/data/trainingData/cached_dataset",
    force_rebuild=False,
    seed=3407,
    include_external_datasets=True,
    external_data_dir="data/trainingData",
    dataset_weights=None
):
    """
    Prepare training dataset with optional external datasets.
    
    Args:
        path: Path to main generated training data (JSON with role/content format)
        system_prompt: System prompt to prepend to conversations
        tokenizer: Tokenizer for encoding
        personality_path: Path to personality-only messages
        max_seq_len: Maximum sequence length
        target_megachats: Number of megachats to generate
        mega_size: Number of conversations per megachat
        cache_path: Where to cache the prepared dataset
        force_rebuild: Force rebuild even if cache exists
        seed: Random seed
        include_external_datasets: Whether to load external datasets (DailyDialog, etc.)
        external_data_dir: Directory containing external dataset JSON files
        dataset_weights: Optional dict like {'generated': 0.2, 'external': 0.7, 'personality': 0.1}
                        If None, uses default balanced mixing
    """
    random.seed(seed)

    # Load from cache if available
    if os.path.exists(cache_path) and not force_rebuild:
        print(f"📥 Loading cached dataset from {cache_path}...")
        return load_from_disk(cache_path)

    print("📦 Preparing dataset from scratch...")

    # === Load Generated Dataset ===
    print("\n📝 Loading generated training data...")
    raw_data = load_dataset("json", data_files=path, split="train")
    raw_list = list(raw_data)

    # Group into conversations
    generated_conversations = []
    current_conv = []
    for msg in raw_list:
        role = msg["role"].lower()
        current_conv.append({"role": role, "content": msg["content"]})

        # End a conversation block at assistant reply
        if role == "assistant":
            generated_conversations.append(current_conv)
            current_conv = []
    
    print(f"  ✓ Generated conversations: {len(generated_conversations)}")

    # === Load Personality Messages ===
    personality_chats = []
    if os.path.exists(personality_path):
        print(f"\n💭 Loading personality data from {personality_path}...")
        with open(personality_path, "r") as f:
            personality_data = json.load(f)
        personality_chats = [[{"role": "assistant", "content": msg["content"]}] for msg in personality_data]
        print(f"  ✓ Personality messages: {len(personality_chats)}")
    else:
        print(f"  ⚠ Personality file not found, skipping")

    # === Load External Datasets ===
    external_conversations = []
    if include_external_datasets:
        external_conversations = load_external_conversations(external_data_dir)

    # === Determine Mixing Weights ===
    if dataset_weights is None:
        # Default weights based on what's available
        if external_conversations:
            dataset_weights = {
                'generated': 0.20,
                'external': 0.70,
                'personality': 0.10
            }
        else:
            # No external data, use old behavior
            dataset_weights = {
                'generated': 0.90,
                'external': 0.00,
                'personality': 0.10
            }
    
    print(f"\n⚖️  Dataset mixing weights:")
    print(f"  - Generated: {dataset_weights.get('generated', 0):.1%}")
    print(f"  - External: {dataset_weights.get('external', 0):.1%}")
    print(f"  - Personality: {dataset_weights.get('personality', 0):.1%}")

    # === Build Conversation Pool ===
    total_convos_needed = target_megachats * mega_size
    print(f"\n🎯 Target: {target_megachats} megachats × {mega_size} convos = {total_convos_needed} total conversations")
    
    conversation_pool = []
    
    # Sample from generated conversations
    if generated_conversations:
        n_generated = int(total_convos_needed * dataset_weights.get('generated', 0.2))
        # Oversample if needed
        if n_generated > len(generated_conversations):
            sampled = random.choices(generated_conversations, k=n_generated)
        else:
            sampled = random.sample(generated_conversations, k=min(n_generated, len(generated_conversations)))
        conversation_pool.extend(sampled)
        print(f"  ✓ Sampled {len(sampled)} from generated data")
    
    # Sample from external conversations
    if external_conversations:
        n_external = int(total_convos_needed * dataset_weights.get('external', 0.7))
        # Oversample if needed
        if n_external > len(external_conversations):
            sampled = random.choices(external_conversations, k=n_external)
        else:
            sampled = random.sample(external_conversations, k=min(n_external, len(external_conversations)))
        conversation_pool.extend(sampled)
        print(f"  ✓ Sampled {len(sampled)} from external datasets")
    
    # Sample from personality messages
    if personality_chats:
        n_personality = int(total_convos_needed * dataset_weights.get('personality', 0.1))
        # Oversample if needed
        if n_personality > len(personality_chats):
            sampled = random.choices(personality_chats, k=n_personality)
        else:
            sampled = random.sample(personality_chats, k=min(n_personality, len(personality_chats)))
        conversation_pool.extend(sampled)
        print(f"  ✓ Sampled {len(sampled)} from personality data")
    
    print(f"\n📦 Total conversation pool: {len(conversation_pool)} conversations")

    # === Build Megachats ===
    print(f"\n🔨 Building {target_megachats} megachats...")
    megachats = []
    
    # Shuffle the pool
    random.shuffle(conversation_pool)
    
    # Build megachats in blocks of mega_size conversations
    for i in range(0, len(conversation_pool), mega_size):
        block_conversations = conversation_pool[i:i+mega_size]
        # Flatten all conversations in this block into one megachat
        megachat = []
        for conv in block_conversations:
            megachat.extend(conv)
        
        if megachat:  # Only add non-empty megachats
            megachats.append(megachat)
        
        if len(megachats) >= target_megachats:
            break

    megachats = megachats[:target_megachats]
    print(f"  ✓ Created {len(megachats)} megachats")

    # === Encode Chats ===
    print(f"\n🔐 Encoding conversations...")
    input_ids_list, labels_list, messages_list = [], [], []
    
    for chat_idx, chat in enumerate(megachats):
        if (chat_idx + 1) % 100 == 0:
            print(f"  Progress: {chat_idx + 1}/{len(megachats)} megachats encoded...")
        
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

    print(f"  ✓ Encoded {len(input_ids_list)} training examples")

    # === Create Dataset ===
    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "messages": messages_list
    })

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    dataset.save_to_disk(cache_path)
    print(f"\n💾 Dataset cached at {cache_path}")
    print(f"📊 Final dataset size: {len(dataset)} training examples\n")

    return dataset