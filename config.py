"""
config.py

Centralized configuration for Ada model training.

Includes:
- File paths
- Model settings
- Hyperparameters
"""

import os

# =============================
# PATHS
# =============================
BASE_DIR = "/home/drew/Documents/AI_Hobby/training/AdaTraining/"

TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data/trainingData/generatedTrainingData-formatted-updated.json")
PERSONALITY_PATH = os.path.join(BASE_DIR, "data/trainingData/personality.json")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "systemprompts/ada_system_prompt.txt")

OUTPUT_DIR = "TrainingOutput"

# =============================
# MODEL SETTINGS
# =============================
MODEL_NAME = "Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1"
REQUIRED_CHAT_TEMPLATE = "qwen3"
MAX_SEQ_LEN = 4096

# LoRa settings
LORA_R = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
LORA_BIAS = "none"
USE_GRADIENT_CHECKPOINTING = "unsloth"
RANDOM_STATE = 3407
USE_RSLORA = False
LOFTQ_CONFIG = None

# =============================
# TRAINING HYPERPARAMETERS
# =============================
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 5e-5
LOGGING_STEPS = 5
MAX_STEPS = 4200
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
SEED = 3407
FP16 = True
BF16 = False
REPORT_TO = "none"
EVAL_STRATEGY = "steps"
EVAL_STEPS = 150 
TEST_SIZE = 0.01  # Validation split fraction

# =============================
# DATASET SETTINGS
# =============================
MEGA_SIZE = 8          # Conversations per megachat
TARGET_MEGACHATS = 4200   # Number of megachats to oversample

# =============================
# MISC
# =============================
SPINNER_COLOR = "PINK"
SPINNER_EMOJI = "🌸"
