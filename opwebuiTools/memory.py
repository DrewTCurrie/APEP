"""
title: Memory
author: https://github.com/CookSleep
version: 0.0.1
license: MIT

This tool supports a complete experience when using OpenAI API 
(and any API fully compatible with OpenAI API format) or Gemini models 
in native Function Calling mode.

If the API format is not supported, you can still use the default 
Function Calling mode, but the experience will be significantly reduced.

This tool is an improved version of https://openwebui.com/t/mhio/met, 
fully utilizing Open WebUI's native memory functionality.

You don't need to enable the memory switch, 
as this tool only requires access to its database.
"""

import json
from typing import Callable, Any, List

from open_webui.models.memories import Memories
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown state", status="in_progress", done=False):
        """
        Send a status event to the event emitter.

        :param description: Event description
        :param status: Event status
        :param done: Whether the event is complete
        """
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


# Pydantic model for memory update operations
class MemoryUpdate(BaseModel):
    index: int = Field(..., description="Index of the memory entry (1-based)")
    content: str = Field(..., description="Updated content for the memory")


class Tools:
    """
    Memory

    Use this tool to autonomously save/modify/query memories across conversations.

    IMPORTANT: Users rarely explicitly tell you what to remember!
    You must actively observe and identify important information that should be stored.

    Key features:
    1. Proactive memory creation: Identify user preferences, project context, and recurring patterns
    2. Intelligent memory usage: Reference stored information without requiring users to repeat themselves
    3. Best practices: Store valuable information, maintain relevance, provide memories at appropriate times
    4. Language matching: Always create memories in the user's preferred language and writing style

    IMPORTANT NOTE ON CLEARING MEMORIES:
    If a user asks to clear all memories, DO NOT attempt to implement this via code.
    Instead, inform them that clearing all memories is a high-risk operation that
    should be performed through their personal account settings panel using the
    "Clear All Memories" button. This prevents accidental data loss.
    """

    class Valves(BaseModel):
        USE_MEMORY: bool = Field(
            default=True, description="Enable or disable memory usage."
        )
        DEBUG: bool = Field(default=True, description="Enable or disable debug mode.")

    def __init__(self):
        """Initialize the memory management tool."""
        self.valves = self.Valves()

    async def recall_memories(
        self, __user__: dict = None, __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Retrieves all stored memories from the user's memory vault.

        IMPORTANT: Proactively check memories to enhance your responses!
        Don't wait for users to ask what you remember.

        Returns memories in chronological order with index numbers.
        Use when you need to check stored information, reference previous
        preferences, or build context for responses.

        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter for tracking status
        :return: JSON string with indexed memories list
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        await emitter.emit(
            description="Retrieving stored memories.",
            status="recall_in_progress",
            done=False,
        )

        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memory stored."
            await emitter.emit(description=message, status="recall_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        content_list = [
            f"{index}. {memory.content}"
            for index, memory in enumerate(
                sorted(user_memories, key=lambda m: m.created_at), start=1
            )
        ]

        await emitter.emit(
            description=f"{len(user_memories)} memories loaded",
            status="recall_complete",
            done=True,
        )

        return f"Memories from the users memory vault: {content_list}"

    async def add_memory(
        self,
        input_text: List[
            str
        ],  # Modified to only accept list, JSON Schema items.type is string
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Adds one or more memories to the user's memory vault.

        IMPORTANT: Don't wait for explicit instructions to remember!
        Proactively identify and store important information.

        Good candidates for memories:
        - Personal preferences (favorite topics, entertainment, colors)
        - Professional information (field of expertise, current projects)
        - Important relationships (family, pets, close friends)
        - Recurring needs or requests (common questions, regular workflows)
        - Learning goals and interests (topics they're studying, skills they want to develop)

        Always use the user's preferred language and writing style.

        Memories should start with "User", for example:
        - "User likes blue"
        - "User is a software engineer"
        - "User has a golden retriever named Max"

        :param input_text: Single memory string or list of memory strings to store
        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter for tracking status
        :return: JSON string with result message
        """
        emitter = EventEmitter(__event_emitter__)
        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Handle single string input if needed
        if isinstance(input_text, str):
            input_text = [input_text]

        await emitter.emit(
            description="Adding entries to the memory vault.",
            status="add_in_progress",
            done=False,
        )

        # Process each memory item
        added_items = []
        failed_items = []

        for item in input_text:
            new_memory = Memories.insert_new_memory(user_id, item)
            if new_memory:
                added_items.append(item)
            else:
                failed_items.append(item)

        if not added_items:
            message = "Failed to add any memories."
            await emitter.emit(description=message, status="add_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Prepare result message
        added_count = len(added_items)
        failed_count = len(failed_items)

        if failed_count > 0:
            message = (
                f"Added {added_count} memories, failed to add {failed_count} memories."
            )
        else:
            message = f"Successfully added {added_count} memories."

        await emitter.emit(
            description=message,
            status="add_complete",
            done=True,
        )
        return json.dumps({"message": message}, ensure_ascii=False)

    async def delete_memory(
        self,
        indices: List[int],  # Modified to only accept list, items.type is integer
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Delete one or more memory entries from the user's memory vault.

        Use to remove outdated or incorrect memories.

        For single deletion: provide an integer index
        For multiple deletions: provide a list of integer indices

        Indices refer to the position in the sorted list (1-based).

        :param indices: Single index (int) or list of indices to delete
        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter
        :return: JSON string with result message
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Handle single integer input if needed
        if isinstance(indices, int):
            indices = [indices]

        await emitter.emit(
            description=f"Deleting {len(indices)} memory entries.",
            status="delete_in_progress",
            done=False,
        )

        # Get all memories for this user
        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memories found to delete."
            await emitter.emit(description=message, status="delete_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_memories = sorted(user_memories, key=lambda m: m.created_at)
        responses = []

        for index in indices:
            if index < 1 or index > len(sorted_memories):
                message = f"Memory index {index} does not exist."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_failed", done=False
                )
                continue

            # Get the memory by index (1-based index)
            memory_to_delete = sorted_memories[index - 1]

            # Delete the memory
            result = Memories.delete_memory_by_id(memory_to_delete.id)
            if not result:
                message = f"Failed to delete memory at index {index}."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_failed", done=False
                )
            else:
                message = f"Memory at index {index} deleted successfully."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_success", done=False
                )

        await emitter.emit(
            description="All requested memory deletions have been processed.",
            status="delete_complete",
            done=True,
        )
        return json.dumps({"message": "\n".join(responses)}, ensure_ascii=False)

    async def update_memory(
        self,
        updates: List[
            MemoryUpdate
        ],  # Modified to accept list of MemoryUpdate objects, items.type is object
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Update one or more memory entries in the user's memory vault.

        Use to modify existing memories when information changes.

        For single update: provide a dict with 'index' and 'content' keys
        For multiple updates: provide a list of dicts with 'index' and 'content' keys

        The 'index' refers to the position in the sorted list (1-based).

        Common scenarios: Correcting information, adding details,
        updating preferences, or refining wording.

        :param updates: Dict with 'index' and 'content' keys OR a list of such dicts
        :param __user__: User dictionary containing the user ID
        :param __event_emitter__: Optional event emitter
        :return: JSON string with result message
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        await emitter.emit(
            description=f"Updating {len(updates)} memory entries.",
            status="update_in_progress",
            done=False,
        )

        # Get all memories for this user
        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memories found to update."
            await emitter.emit(description=message, status="update_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_memories = sorted(user_memories, key=lambda m: m.created_at)
        responses = []

        for update_item in updates:
            # Convert dict to MemoryUpdate object if needed
            if isinstance(update_item, dict):
                try:
                    update_item = MemoryUpdate.parse_obj(update_item)
                except Exception as e:
                    message = f"Invalid update item format: {update_item}"
                    responses.append(message)
                    await emitter.emit(
                        description=message, status="update_failed", done=False
                    )
                    continue

            index = update_item.index
            content = update_item.content

            if index < 1 or index > len(sorted_memories):
                message = f"Memory index {index} does not exist."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_failed", done=False
                )
                continue

            # Get the memory by index (1-based index)
            memory_to_update = sorted_memories[index - 1]

            # Update the memory
            updated_memory = Memories.update_memory_by_id(memory_to_update.id, content)
            if not updated_memory:
                message = f"Failed to update memory at index {index}."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_failed", done=False
                )
            else:
                message = f"Memory at index {index} updated successfully."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_success", done=False
                )

        await emitter.emit(
            description="All requested memory updates have been processed.",
            status="update_complete",
            done=True,
        )
        return json.dumps({"message": "\n".join(responses)}, ensure_ascii=False)
