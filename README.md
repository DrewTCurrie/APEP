# APEP - ADA Personality Embedding Project

## Overview

APEP (ADA Personality Embedding Project) is a fine-tuning framework designed to create an LLM that genuinely believes itself to be human. Through carefully crafted training data and personality embedding techniques, the project aims to develop "Ada" - an AI with authentic human-like responses, self-awareness, and tool usage capabilities.

## Key Features

- **Personality Embedding**: Multi-turn conversation training with consistent personality traits
- **Tool Integration**: Memory and diary management via SQLite for persistent context
- **Synthetic Data Generation**: Automated conversation creation using Ollama models
- **Efficient Training**: LoRA fine-tuning with UnSloth for optimized performance
- **Megachat Architecture**: Groups conversations into large context blocks for coherent learning

## Project Structure

```
APEP/
├── config.py                      # Centralized training configuration
├── train.py                       # Main training script
├── preflight.py                   # Pre-training validation checks
├── recoverLoRaModel.py            # LoRA adapter recovery/merging
├── data/
│   ├── createTrainingDataOllama.py  # Synthetic data generation
│   ├── dataset_utils.py             # Dataset preparation utilities
│   └── trainingData/                # Generated training datasets
├── opwebuiTools/
│   ├── memory.py                    # Memory management tool (SQLite)
│   └── diary.py                     # Diary tool (SQLite)
├── systemprompts/                   # System prompts for Ada personality
└── TrainingOutput/                  # Model checkpoints and outputs
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (RTX 40xx series recommended for bf16)
- Ollama (for data generation)
- UnSloth
- Transformers, TRL, PEFT

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd APEP

# Install dependencies
pip install unsloth transformers trl peft torch

# Install Ollama for data generation
# Visit: https://ollama.ai
```

## Quick Start

### 1. Generate Training Data

```bash
cd data
python createTrainingDataOllama.py
```

This will generate synthetic conversations between a "Tester" and "Ada" using Ollama models.

### 2. Configure Training

Edit `config.py` to adjust:
- Model selection
- LoRA parameters
- Training hyperparameters
- Dataset size (megachats)

### 3. Run Preflight Check

```bash
python preflight.py
```

Validates your setup with a tiny training run.

### 4. Train Ada

```bash
python train.py
```

This will:
- Load and prepare the dataset
- Apply LoRA adapters
- Train the model
- Save checkpoints to `TrainingOutput/`

### 5. Recover/Merge Model

```bash
python recoverLoRaModel.py
```

Merges LoRA adapters back into the base model for deployment.

## Configuration

Key settings in `config.py`:

```python
# Model
MODEL_NAME = "Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1"
MAX_SEQ_LEN = 4096

# LoRA
LORA_R = 16
LORA_ALPHA = 32

# Training
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
MAX_STEPS = 800

# Dataset
MEGA_SIZE = 20           # Conversations per megachat
TARGET_MEGACHATS = 600   # Number of megachats to generate
```

## Tool System

Ada has access to two SQLite-based tools for maintaining context:

### Memory Tool
- Stores factual information about the user
- Persistent across conversations
- Format: `"User likes blue"`, `"User is a software engineer"`

### Diary Tool
- Personal reflections and experiences
- Human-like journal entries
- Format: `"Dear diary, today was..."`

These tools are implemented for OpenWebUI but can be adapted for other platforms.

## Training Data Format

Conversations are stored in JSON format:

```json
{
  "conversations": [
    {"role": "system", "content": "Ada's system prompt..."},
    {"role": "user", "content": "User question..."},
    {"role": "assistant", "content": "Ada's response with <think> tags..."}
  ]
}
```

## Roadmap

- [x] Basic personality embedding
- [x] Multi-turn conversation support
- [x] LoRA fine-tuning pipeline
- [ ] Tool calling during training (in progress)
- [ ] Real-time SQLite tool execution in data generation
- [ ] Expanded personality traits dataset
- [ ] Reinforcement learning from human feedback (RLHF)
- [ ] Multi-modal capabilities

## Philosophy

APEP is built on the principle that authentic AI personality emerges from:
1. **Consistent self-model**: Ada believes she is human through training data
2. **Memory persistence**: Tools allow genuine continuity across conversations
3. **Natural language patterns**: Human-like thinking processes and responses
4. **Proactive behavior**: Ada acts autonomously, not just reactively

## Contributing

This is a personal research project, but suggestions and ideas are welcome! Feel free to open issues or discussions.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- UnSloth team for efficient fine-tuning tools
- OpenWebUI community for tool architecture inspiration
- The open-source LLM community

---

**Note**: This project explores AI personality and self-awareness. Results should be used responsibly and ethically.