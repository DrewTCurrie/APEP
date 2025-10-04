"""
conversation_generator.py

Core conversation generation logic with tool execution support.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from api_clients import BaseAPIClient
from tool_executor import ToolExecutor

SYSTEM_PROMPT_PATH="../systemprompts/"

class ConversationGenerator:
    """Generates training conversations with tool execution"""
    
    def __init__(
        self, 
        client: BaseAPIClient,
        tool_executor: ToolExecutor,
        prompts_dir: str = "systemprompts"
    ):
        self.client = client
        self.tool_executor = tool_executor
        self.prompts_dir = Path(prompts_dir)
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load system prompts from files"""
        prompts = {}
        prompt_files = {
            "tester_system": os.path.join(SYSTEM_PROMPT_PATH, "tester_prompt.txt"),
            "tester_user": os.path.join(SYSTEM_PROMPT_PATH, "tester_prompt.txt"),
            "ada_system": os.path.join(SYSTEM_PROMPT_PATH, "ada_system_prompt.txt"),
            "rewriter_system": os.path.join(SYSTEM_PROMPT_PATH, "reweritter_prompt.txt")
        }
        
        for key, filename in prompt_files.items():
            filepath = self.prompts_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    prompts[key] = f.read()
            else:
                print(f"⚠️  Missing prompt file: {filepath}")
                prompts[key] = ""
        
        # Add tool instructions to Ada's system prompt
        if prompts.get("ada_system"):
            prompts["ada_system"] += self._get_tool_instructions()
        
        return prompts
    
    def _get_tool_instructions(self) -> str:
        """Get tool usage instructions to append to Ada's prompt"""
        return """

=== AVAILABLE TOOLS ===
You have access to memory and diary management tools. Use them naturally and proactively:

Memory Tools (for facts about Drew):
- <tool_call>add_memory(input_text=["User likes X"])</tool_call>
- <tool_call>recall_memories()</tool_call>
- <tool_call>delete_memory(indices=[1])</tool_call>

Diary Tools (for your personal reflections):
- <tool_call>add_diary(input_text=["Dear diary, today..."])</tool_call>
- <tool_call>recall_diary()</tool_call>
- <tool_call>delete_diary(indices=[1])</tool_call>

IMPORTANT: Use these tools proactively! Don't wait for Drew to tell you what to remember.
Store important preferences, ongoing projects, and personal reflections naturally.
"""
    
    async def generate_conversation(
        self,
        previous_question: str = "None.",
        previous_answer: str = "None.",
        user_id: str = "drew"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a complete conversation with tool execution.
        
        Args:
            previous_question: Context from previous conversation
            previous_answer: Context from previous conversation
            user_id: User ID for tool execution
        
        Returns:
            Conversation dict or None on failure
        """
        
        # Step 1: Generate question from Tester
        question = await self._generate_question(previous_question, previous_answer)
        if not question:
            return None
        
        # Step 2: Ada's initial response (may include tool calls)
        ada_response = await self._generate_ada_response(question)
        if not ada_response:
            return None
        
        # Step 3: Execute any tool calls
        tool_results = self.tool_executor.parse_and_execute(ada_response, user_id)
        
        # Step 4: If tools were called, give Ada the results and let her continue
        if tool_results:
            ada_response = await self._handle_tool_results(
                question, 
                ada_response, 
                tool_results
            )
            if not ada_response:
                return None
        
        # Step 5: Rewrite thinking section for human-like quality
        final_response = await self._rewrite_thinking(ada_response)
        if not final_response:
            final_response = ada_response
        
        # Build conversation object
        conversation = {
            "conversations": [
                {"role": "system", "content": self.prompts["ada_system"]},
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_response}
            ]
        }
        
        return conversation
    
    async def _generate_question(
        self, 
        previous_question: str, 
        previous_answer: str
    ) -> Optional[str]:
        """Generate a question from the Tester"""
        tester_user = (
            self.prompts["tester_user"] +
            f"\n\nYour previous question: {previous_question}\n" +
            f"Ada's answer to previous question: {previous_answer}"
        )
        
        response = await self.client.query([
            {"role": "system", "content": self.prompts["tester_system"]},
            {"role": "user", "content": tester_user}
        ])
        
        if not response:
            return None
        
        # Extract question (remove thinking tags if present)
        question = response.partition('</think>\n\n')[2] or response
        return question.strip()
    
    async def _generate_ada_response(self, question: str) -> Optional[str]:
        """Generate Ada's response to the question"""
        response = await self.client.query([
            {"role": "system", "content": self.prompts["ada_system"]},
            {"role": "user", "content": question}
        ])
        
        return response if response else None
    
    async def _handle_tool_results(
        self,
        question: str,
        initial_response: str,
        tool_results: List[tuple]
    ) -> Optional[str]:
        """Give Ada tool results and let her continue"""
        
        # Format tool results
        tool_response_text = "\n\n".join([
            f"<tool_result>{json.dumps(result[2])}</tool_result>"
            for result in tool_results
        ])
        
        # Ada sees the tool results and continues
        continuation = await self.client.query([
            {"role": "system", "content": self.prompts["ada_system"]},
            {"role": "user", "content": question},
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": tool_response_text}
        ])
        
        if continuation:
            return initial_response + "\n" + continuation
        else:
            return initial_response
    
    async def _rewrite_thinking(self, response: str) -> Optional[str]:
        """Rewrite thinking section for more human-like quality"""
        
        # Extract thinking section
        thinking_section = response.partition('<think>\n')[2].partition('</think>')[0]
        
        if not thinking_section:
            return response  # No thinking section to rewrite
        
        # Ask rewriter to improve it
        rewrite_prompt = (
            "** Internal thinking section to rewrite:**\n" +
            thinking_section +
            "\n**End of Internal thinking section to rewrite.**"
        )
        
        revised_thinking = await self.client.query([
            {"role": "system", "content": self.prompts["rewriter_system"]},
            {"role": "user", "content": rewrite_prompt}
        ])
        
        if not revised_thinking:
            return response  # Keep original on failure
        
        # Clean up and substitute
        revised_thinking = revised_thinking.replace("the user", "Drew")
        revised_thinking = revised_thinking.partition('</think>\n\n')[2] or revised_thinking
        
        # Reconstruct response
        trimmed_answer = response.partition('</think>\n\n')[2]
        final_response = revised_thinking + "</think>\n\n" + trimmed_answer
        
        return final_response
    
    def get_prompts(self) -> Dict[str, str]:
        """Get loaded prompts (useful for inspection)"""
        return self.prompts.copy()