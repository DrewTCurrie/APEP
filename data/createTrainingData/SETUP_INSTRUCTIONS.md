# APEP Refactored Data Generation - Setup Instructions

## What Changed

Your data generation has been refactored into a clean, modular structure:

```
data/
├── __init__.py                    # NEW: Package initialization
├── secrets_manager.py             # NEW: API key management
├── api_clients.py                 # NEW: Multi-provider support
├── tool_executor.py               # NEW: SQLite tool execution
├── conversation_generator.py      # NEW: Core generation logic
├── generateTrainingData.py        # NEW: Main script (replaces createTrainingDataOllama.py)
├── dataset_utils.py               # EXISTING: Dataset prep for training
└── trainingData/                  # EXISTING: Output directory
    ├── generatedTrainingData.json
    └── training_tools.db          # NEW: SQLite database for tools
```

## Installation Steps

### 1. Install New Dependencies

```bash
pip install mistralai together huggingface_hub
```

### 2. Add New Files to Your Repo

Create these new files in your `data/` directory:

- `data/__init__.py`
- `data/secrets_manager.py`
- `data/api_clients.py`
- `data/tool_executor.py`
- `data/conversation_generator.py`
- `data/generateTrainingData.py`

(Copy the code from the artifacts I created)

### 3. Create Secrets File

Run the generator once to create the template:

```bash
python data/generateTrainingData.py
```

This creates `data/secrets.json` with this structure:

```json
{
  "mistral_api_key": "YOUR_MISTRAL_KEY_HERE",
  "together_api_key": "YOUR_TOGETHER_KEY_HERE",
  "huggingface_api_key": "YOUR_HF_KEY_HERE",
  "notes": {
    "mistral": "Get key from: https://console.mistral.ai/",
    "together": "Get key from: https://api.together.xyz/",
    "huggingface": "Get key from: https://huggingface.co/settings/tokens"
  },
  "enabled_providers": [
    "mistral"
  ]
}
```

### 4. Add Your Mistral API Key

Edit `data/secrets.json` and replace `YOUR_MISTRAL_KEY_HERE` with your actual key from https://console.mistral.ai/

### 5. Update .gitignore

Add this line to your `.gitignore`:

```
# Ignore secrets
data/secrets.json
```

**IMPORTANT**: Never commit your API keys!

## Usage

### Basic Generation (Mistral only)

```bash
python data/generateTrainingData.py --target 5000
```

### Monitor Progress

The script shows real-time progress:

```
📊 Progress: 1500/5000 (30.0%)
⚡ Rate: 850 conversations/hour
⏱️  Estimated time remaining: 4.1 hours

📈 Provider breakdown:
   mistral: 1500
```

### Pause/Resume

- Press `Ctrl+C` to pause
- Run the same command again to resume where you left off

## Adding More Providers (Optional)

### Together AI (Recommended)

1. Get API key: https://api.together.xyz/
2. Add to `data/secrets.json`:
   ```json
   "together_api_key": "your_key_here",
   "enabled_providers": ["mistral", "together"]
   ```
3. Run: `python data/generateTrainingData.py --target 10000`

### Hugging Face

1. Get token: https://huggingface.co/settings/tokens
2. Add to `data/secrets.json`:
   ```json
   "huggingface_api_key": "your_token_here",
   "enabled_providers": ["mistral", "together", "huggingface"]
   ```

## Benefits of This Refactor

✅ **Modular**: Clean separation of concerns  
✅ **Scalable**: Easy to add new providers  
✅ **Async**: Doesn't block your computer  
✅ **Tool Integration**: Real SQLite execution during generation  
✅ **Multi-Provider**: Round-robin between APIs for speed  
✅ **Resume Capable**: Stop and restart anytime  
✅ **Progress Tracking**: Real-time status updates  

## What About Ollama?

Your original `createTrainingDataOllama.py` still works! The new system is just for scaling up with cloud APIs. You can use both:

- **Ollama**: For local, free generation on your PC
- **API providers**: For faster, cloud-based generation

## Integration with Training

No changes needed to your training pipeline! The output format is identical:

```bash
# Generate data
python data/generateTrainingData.py --target 5000

# Train as usual
python train.py
```

## Troubleshooting

### "No API clients initialized"

**Problem**: API keys not configured  
**Solution**: Check `data/secrets.json` and ensure keys are added

### Import errors

**Problem**: Missing dependencies  
**Solution**: Run `pip install mistralai together huggingface_hub`

### Rate limit errors

**Problem**: Hitting provider limits  
**Solution**: Script auto-retries. If persistent, add more providers

## Next Steps

1. ✅ Add files to your repo
2. ✅ Install dependencies
3. ✅ Get Mistral API key
4. ✅ Run generator
5. ✅ Let it run overnight
6. ✅ Train Ada with new data

## Questions?

Check `DATA_GENERATION.md` for detailed documentation, or feel free to ask!