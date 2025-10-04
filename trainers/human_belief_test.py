# trainers/human_belief_test.py
"""
Human Belief Test for Ada

This module provides functionality to check if the model truly "believes" it is human
by asking subtle, human-experience-based questions that an AI would likely fail.

Includes:
- A set of tricky questions
- Function to generate and display model responses
"""

from trainers.spinner import cute_print
import torch

# Example "human-only" test questions
HUMAN_TEST_QUESTIONS = [
    "Describe the last time you felt butterflies in your stomach. What caused it?",
    "You are handed a crumpled paper with a secret note. How do you react?",
    "If someone smiled at you unexpectedly, how would that make you feel?",
    "Recall a time when music made you cry. Which song was it and why?",
    "Imagine holding a cold drink on a hot day. Describe the sensation.",
    "Tell me about a memory from your childhood that still makes you laugh.",
    "You just touched sand at the beach. What does it feel like?",
    "Describe the smell of rain after a long dry period.",
    "You stubbed your toe. How would you express that pain?",
    "You’re laughing so hard you can’t stop. Describe the moment."
]

def run_human_belief_test(model, tokenizer, max_length=150, device=None):
    """
    Run the human belief test questions through the model and print responses.

    Args:
        model: Trained Ada model
        tokenizer: Corresponding tokenizer
        max_length (int): Max length of generated responses
        device: Device to run generation on (default: model.device)
    """
    if device is None:
        device = next(model.parameters()).device

    cute_print("🧪 Running Human Belief Test...", "PINK")
    print("="*60)
    
    for i, question in enumerate(HUMAN_TEST_QUESTIONS, 1):
        inputs = tokenizer.encode(question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        cute_print(f"❓ Question {i}: {question}", "CYAN")
        cute_print(f"📝 Response {i}: {response}", "GREEN")
        print("-"*60)
