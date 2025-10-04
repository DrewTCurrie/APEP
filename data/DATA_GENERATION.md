# Training Data Generation

This document explains how to generate training conversations for APEP using multiple API providers.

## Quick Start

```bash
# 1. Install dependencies
pip install mistralai together huggingface_hub

# 2. Run the generator (creates secrets template)
python data/generateTrainingData.py

# 3. Add your API keys to data/secrets.json

# 4. Generate conversations
python data/generateTrainingData.py --target 5000
```

## Supported Providers

### Mistral AI (Recommended)
- **Free tier**: 1 request/second, 500k tokens/min, 1B tokens/month
- **Model**: mistral-large-2411
- **Get API key**: https://console.mistral.ai/

### Together AI
- **Free tier**: $25 credit to start
- **Models**: Llama 3.1, Mixtral, Qwen, etc.
- **Get API key**: https://api.together.xyz/

### Hugging Face
- **Free tier**: Rate-limited inference API
- **Models**: Various open-source models
- **Get API key**: https://huggingface.co/settings/tokens

## Configuration

### secrets.json Format

```json
{
  "mistral_api_key": "your_key_here",
  "together_api_key": "your_key_here",
  "huggingface_api_key": "your_key_here",
  "enabled_providers": [
    "mistral"
  ]
}
```

### Enabling Multiple Providers

To use multiple providers simultaneously for faster generation:

```json
{
  "enabled_providers": [
    "mistral",
    "together",
    "huggingface"
  ]
}
```

The generator will round-robin between providers automatically.

## Usage Examples

### Basic Generation

```bash
# Generate 5,000 conversations (default)
python data/generateTrainingData.py

# Generate specific number
python data/generateTrainingData.py --target 10000
```

### Multi-Provider Generation

```bash
# Use multiple providers (overrides secrets.json)
python data/generateTrainingData.py --providers mistral,together

# Custom output location
python data/generateTrainingData.py --output my_custom_data.json
```

### Advanced Options

```bash
# Reset tool database before starting
python data/generateTrainingData.py --reset-tools

# Full example
python data/generateTrainingData.py \
    --target 10000 \
    --providers mistral,together \
    --output data/trainingData/custom_data.json
```

## Architecture

### Module Structure

```
data/
├── __init__.py                 # Package initialization\
└──createTrainingData
   ├── secrets_manager.py           # API key management
   ├── api_clients.py               # Provider API clients
   ├── tool_executor.py             # SQLite tool execution
   ├── conversation_generator.py    # Conversation logic
   └── generateTrainingDataAPI.py   # Main script
```

### Data Flow

1. **Question Generation**: Tester model generates a question based on previous context
2. **Ada's Response**: Ada responds, potentially with tool calls
3. **Tool Execution**: Any `<tool_call>` tags are parsed and executed against SQLite
4. **Tool Results**: Results are fed back to Ada for continuation
5. **Thinking Rewrite**: Internal thinking is rewritten for quality
6. **Save**: Complete conversation saved to JSON

### Tool Integration

The generator includes SQLite-based memory and diary tools:

**Memory Tools** (for facts):
- `add_memory(input_text=["User likes X"])`
- `recall_memories()`
- `delete_memory(indices=[1])`

**Diary Tools** (for reflections):
- `add_diary(input_text=["Dear diary, ..."])`
- `recall_diary()`
- `delete_diary(indices=[1])`

These tools are automatically executed during generation, creating realistic training data where Ada learns to use persistent memory naturally.

## Performance

### Rate Limits

| Provider | Requests/Second | Est. Conversations/Hour |
|----------|----------------|------------------------|
| Mistral | 1.0 | ~900 |
| Together | 1.0 | ~900 |
| HuggingFace | 0.5 | ~450 |

**Multi-provider**: With all 3 providers, you can generate ~2,250 conversations/hour.

### Token Usage

Each conversation uses approximately:
- **Question generation**: 200-500 tokens
- **Ada's response**: 300-800 tokens
- **Tool continuation**: 200-400 tokens (if tools used)
- **Thinking rewrite**: 150-300 tokens
- **Total per conversation**: ~850-2000 tokens

**Monthly estimates**:
- Mistral's 1B tokens/month = ~500k-1.2M conversations
- Together's $25 credit ≈ 25M tokens = ~12k-30k conversations

## Monitoring

The generator provides real-time progress updates:

```
📊 Progress: 1500/5000 (30.0%)
⚡ Rate: 850 conversations/hour
⏱️  Estimated time remaining: 4.1 hours

📈 Provider breakdown:
   mistral: 850
   together: 650
```

## Troubleshooting

### API Key Issues

```
❌ Error: No API clients initialized!
```

**Solution**: Check `data/secrets.json` and ensure:
1. API keys are added (not placeholder values)
2. At least one provider is in `enabled_providers`

### Rate Limit Errors

```
⚠️  MistralClient error (attempt 1/3): Rate limit exceeded
```

**Solution**: The generator automatically retries with exponential backoff. If persistent:
1. Reduce `requests_per_second` in `api_clients.py`
2. Enable additional providers to distribute load

### Import Errors

```
ImportError: Please install mistralai: pip install mistralai
```

**Solution**: Install missing dependencies:
```bash
pip install mistralai together huggingface_hub
```

## Best Practices

1. **Start Small**: Test with 100-500 conversations first
2. **Monitor Token Usage**: Check provider dashboards regularly
3. **Use Multiple Providers**: Distribute load for faster generation
4. **Save Regularly**: The script auto-saves after each conversation
5. **Validate Output**: Periodically check generated conversations for quality

## Output Format

Conversations are saved in JSON format compatible with the training pipeline:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "Ada's system prompt..."
    },
    {
      "role": "user",
      "content": "User question..."
    },
    {
      "role": "assistant",
      "content": "<think>Internal reasoning...</think>\n\nAda's response..."
    }
  ]
}
```

## Integration with Training Pipeline

After generating conversations:

```bash
# 1. Generate data
python data/generateTrainingData.py --target 10000

# 2. Update config.py with new data path (if needed)

# 3. Run training
python train.py
```

The existing `dataset_utils.py` will automatically handle the new conversations in the megachat grouping process.

## Tips for Quality Data

1. **Diversity**: Use multiple providers for varied conversation styles
2. **Context**: The generator maintains conversation context between exchanges
3. **Tools**: Let Ada naturally use memory/diary tools - don't force it
4. **Iteration**: Review generated conversations and adjust prompts in `systemprompts/`
5. **Balance**: Mix tool-heavy and tool-free conversations for realistic training

---

For questions or issues, check the main project README or open a GitHub issue.