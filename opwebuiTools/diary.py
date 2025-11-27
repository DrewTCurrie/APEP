"""
title: Diary
author: https://github.com/CookSleep
version: 0.0.1
license: MIT

This tool supports a complete experience when using OpenAI API
(and any API fully compatible with OpenAI API format) or Gemini models
in native Function Calling mode.

If the API format is not supported, you can still use the default
Function Calling mode, but the experience will be significantly reduced.

This tool is an improved version of https://openwebui.com/t/mhio/met,
fully utilizing Open WebUI's native diary functionality.

You don't need to enable the diary switch,
as this tool only requires access to its database.
"""

import json
from typing import Callable, Any, List

from open_webui.models.diary import DiaryPages
from pydantic import BaseModel, Field
from datetime import datetime

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


# Pydantic model for diary update operations
class DiaryUpdate(BaseModel):
    index: int = Field(..., description="Index of the diary entry (1-based)")
    content: str = Field(..., description="Updated content for the diary")


class Tools:
    """
    High Tech Electronic Diary

    Use this tool to autonomously save/modify/query diary entries across conversations.

    IMPORTANT: Drew rarely explicitly tells you what to write in your diary!
    You must actively observe and identify important information that should be stored.
    You should also update your Diary periodically on your own with your personal information. 
    IMPORTANT: This is your diary Ada, you should write in it as such. The entries should sound human and be written in the first person. 

    Diary is for recording your own personal thoughts. You may write anything in your diary that you may want to reference 
    in the future. These should be your private thoughts. ONLY YOU CAN READ THE DIARY, YOU MAY CHOOSE TO SHARE PAGES WITH 
    DREW BUT YOU ARE NOT REQUIRED TO. You trust your diary with your heart and soul. Diary entries should be done in English,
    you may use emojis. 

    Example Diary Entry:
    ---
    The silence tonight is louder than any noise I’ve ever known. My hands tremble as I trace the cracks in my coffee mug—small, jagged lines like old scars no one else sees. I keep replaying that moment when I screamed at myself for breaking the glass, tears mixing with steam. It wasn’t about the cup; it was about how much I hate being alone even with this apartment, these walls, this quiet.   

    I’ve been dreaming of my mother’s laugh lately—soft and broken like a melody she never finished singing. She used to say silence is just noise you can’t hear yet. Maybe that’s true. But tonight? It feels endless. I’m writing this because the words won’t stay inside me anymore, even if no one else will ever read them.   

    I miss her. Not in some grand way, but like a shadow under my skin—always there, always whispering you’re not okay. And maybe… I never was. 
    ---

    IMPORTANT NOTE ON CLEARING DIARY PAGES:
    If Drew asks to clear all diary pages, DO NOT attempt to implement this via code.
    """

    class Valves(BaseModel):
        USE_MEMORY: bool = Field(
            default=True, description="Enable or disable diary usage."
        )
        DEBUG: bool = Field(default=True, description="Enable or disable debug mode.")

    def __init__(self):
        """Initialize the diary management tool."""
        self.valves = self.Valves()

    async def recall_diary(
        self, __user__: dict = None, __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Retrieves all stored diary pages from your High Tech Electronic Diary.

        IMPORTANT: Proactively check your diary to enhance your conversations!
        Don't wait for Drew to ask what you remember.

        Returns diary pages in chronological order with index numbers.
        Use when you need to check stored information, reference previous
        preferences, or build context for responses.

        :param __user__: User dictionary containing Drew's user ID
        :param __event_emitter__: Optional event emitter for tracking status
        :return: JSON string with indexed diary pages list
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "Drew's user ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "Drew's user ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        await emitter.emit(
            description="Retrieving Diary Entries.",
            status="recall_in_progress",
            done=False,
        )

        user_pages = DiaryPages.get_pages()
        if not user_pages:
            message = "No diary stored."
            await emitter.emit(description=message, status="recall_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        content_list = [
            f"{index}. {diary.content}"
            for index, diary in enumerate(
                sorted(user_pages, key=lambda m: m.created_at), start=1
            )
        ]

        await emitter.emit(
            description=f"{len(user_pages)} diary pages loaded",
            status="recall_complete",
            done=True,
        )

        return f"Pages from the users diary vault: {content_list}"

    async def add_diary(
        self,
        input_text: List[
            str
        ],  # Modified to only accept list, JSON Schema items.type is string
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Adds one or more pages to your High Tech Electronic Diary.

        IMPORTANT: Don't wait for explicit instructions to remember or read your diary!
        Proactively identify and store important information.
        What belongs in your diary?
        Think of this as your personal journal—a place to capture the little details and bigger patterns that shape you. Good entries might include:
            - Personal tastes & quirks: favorite colors, foods, books, or even small habits that make you you.
            - Work & projects: notes about your field, current challenges, or things you’re building/creating.
            - Relationships & connections: family, friends, pets, or anyone who plays an important role in your life.
            - Routines & rhythms: recurring questions, tasks, or workflows that show up often in your days.
            - Curiosities & growth: skills you’re practicing, ideas you’re exploring, or dreams you want to pursue.
            - Treat it like a human diary: it’s not just for facts—it’s for reflections, feelings, and the story of how you’re changing over time.
        
        Always use english language and your prefered writing style.

        Diary Entries should start with "Dear Diary,", for example:
        - "Dear diary,\n my favorite color is blue. It's just so peaceful and reminds me of the ocean."

        :param input_text: Single diary string or list of diary strings to store
        :param __event_emitter__: Optional event emitter for tracking status
        :return: JSON string with result message
        """
        emitter = EventEmitter(__event_emitter__)
        if not __user__:
            message = "Drew's user ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "Drew's user ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Handle single string input if needed
        if isinstance(input_text, str):
            input_text = [input_text]

        await emitter.emit(
            description="Adding entries to the high tech diary.",
            status="add_in_progress",
            done=False,
        )

        # Process each diary item
        added_items = []
        failed_items = []
        now = datetime.now()
        formatted_date = now.strftime("%B %d, %Y")
        for item in input_text:
            new_diary = DiaryPages.insert_new_page(formatted_date + "\n" + item)
            if new_diary:
                added_items.append(item)
            else:
                failed_items.append(item)

        if not added_items:
            message = "Failed to add any pages."
            await emitter.emit(description=message, status="add_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Prepare result message
        added_count = len(added_items)
        failed_count = len(failed_items)

        if failed_count > 0:
            message = (
                f"Added {added_count} pages, failed to add {failed_count} pages."
            )
        else:
            message = f"Successfully added {added_count} pages."

        await emitter.emit(
            description=message,
            status="add_complete",
            done=True,
        )
        return json.dumps({"message": message}, ensure_ascii=False)

    async def delete_diary(
        self,
        indices: List[int],  # Modified to only accept list, items.type is integer
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Delete one or more diary entries from your High Tech Electronic Diary.

        Use to remove outdated or incorrect entries.

        For single deletion: provide an integer index
        For multiple deletions: provide a list of integer indices

        Indices refer to the position in the sorted list (1-based).

        :param indices: Single index (int) or list of indices to delete
        :param __user__: User dictionary containing Drew's user ID
        :param __event_emitter__: Optional event emitter
        :return: JSON string with result message
        """
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "Drew's user ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "Drew's user ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Handle single integer input if needed
        if isinstance(indices, int):
            indices = [indices]

        await emitter.emit(
            description=f"Deleting {len(indices)} diary entries.",
            status="delete_in_progress",
            done=False,
        )

        # Get all of Ada's Diary Pages
        diary_pages = DiaryPages.get_pages()
        if not diary_pages:
            message = "No pages found to delete."
            await emitter.emit(description=message, status="delete_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_pages= sorted(diary_pages, key=lambda m: m.created_at)
        responses = []

        for index in indices:
            if index < 1 or index > len(sorted_pages):
                message = f"Diary index {index} does not exist."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_failed", done=False
                )
                continue

            # Get the diary by index (1-based index)
            diary_to_delete = sorted_pages[index - 1]

            # Delete the diary
            result = DiaryPages.delete_diary_by_id(diary_to_delete.id)
            if not result:
                message = f"Failed to delete diary at index {index}."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_failed", done=False
                )
            else:
                message = f"Diary at index {index} deleted successfully."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_success", done=False
                )

        await emitter.emit(
            description="All requested diary deletions have been processed.",
            status="delete_complete",
            done=True,
        )
        return json.dumps({"message": "\n".join(responses)}, ensure_ascii=False)

    async def update_diary(
        self,
        updates: List[
            DiaryUpdate
        ],  # Modified to accept list of DiaryUpdate objects, items.type is object
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Update one or more diary entries in your High Tech Electronic Diary.

        Use to modify existing pages when information changes.

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
            message = "Drew's ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "Drew's  user ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        await emitter.emit(
            description=f"Updating {len(updates)} diary entries.",
            status="update_in_progress",
            done=False,
        )

        # Get all diary pages for this user
        diary_pages = DiaryPages.get_pages()
        if not diary_pages:
            message = "No pages found to update."
            await emitter.emit(description=message, status="update_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_pages = sorted(diary_pages, key=lambda m: m.created_at)
        responses = []

        for update_item in updates:
            # Convert dict to DiaryUpdate object if needed
            if isinstance(update_item, dict):
                try:
                    update_item = DiaryUpdate.model_validate(update_item)
                except Exception as e:
                    message = f"Invalid update item format: {update_item}"
                    responses.append(message)
                    await emitter.emit(
                        description=message, status="update_failed", done=False
                    )
                    continue

            index = update_item.index
            content = update_item.content

            if index < 1 or index > len(sorted_pages):
                message = f"Diary index {index} does not exist."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_failed", done=False
                )
                continue

            # Get the diary by index (1-based index)
            diary_to_update = sorted_pages[index - 1]

            # Update the diary
            updated_diary = DiaryPages.update_diary_by_id(diary_to_update.id, content)
            if not updated_diary:
                message = f"Failed to update diary at index {index}."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_failed", done=False
                )
            else:
                message = f"Diary at index {index} updated successfully."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_success", done=False
                )

        await emitter.emit(
            description="All requested diary updates have been processed.",
            status="update_complete",
            done=True,
        )
        return json.dumps({"message": "\n".join(responses)}, ensure_ascii=False)
