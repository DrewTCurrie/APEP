"""
trainers/callbacks.py

Custom Trainer callbacks for validation previews.
"""

from transformers import TrainerCallback
from .spinner import cute_print


class ValidationPreviewCallback(TrainerCallback):
    """
    Callback to preview a single validation example during training.

    Shows:
        - Decoded prompt
        - Model generated response
    """

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, eval_dataloader=None, **kwargs):
        """Display one decoded validation sample."""
        if eval_dataloader is None or tokenizer is None or model is None:
            return
        try:
            batch = next(iter(eval_dataloader))
            input_ids = batch["input_ids"][0].to(model.device)
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            outputs = model.generate(input_ids.unsqueeze(0), max_length=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            cute_print("\n🧪 Validation Example Preview:", "PINK")
            print("Prompt:\n", prompt)
            print("Model Response:\n", response)
            print("="*60)
        except Exception as e:
            cute_print(f"❌ Validation Preview Failed: {e}", "RED")
