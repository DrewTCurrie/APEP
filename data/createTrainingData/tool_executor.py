"""
tool_executor.py

SQLite-based tool execution for memory, diary, reminders, and system utilities during training data generation.
Simulates OpenWebUI's persistence features.
"""

import ast
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from zoneinfo import ZoneInfo


ADA_TIMEZONE = ZoneInfo("America/Denver")


class ToolExecutor:
    """Executes SQLite tools during conversation generation"""
    
    def __init__(self, db_path: str = "data/trainingData/training_tools.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.setup_database()
    
    def setup_database(self):
        """Initialize memory and diary tables"""
        cursor = self.conn.cursor()
        
        # Memory table (matches OpenWebUI schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Diary table (matches OpenWebUI schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diary_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Reminders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    # =============================
    # Memory Operations
    # =============================
    
    def add_memory(self, user_id: str, content: str) -> Dict[str, Any]:
        """Add a memory entry"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memories (user_id, content) VALUES (?, ?)",
            (user_id, content)
        )
        self.conn.commit()
        
        return {
            "success": True,
            "message": "Successfully added 1 memory.",
            "memory_id": cursor.lastrowid
        }
    
    def recall_memories(self, user_id: str) -> Dict[str, Any]:
        """Retrieve all memories for a user"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT content FROM memories WHERE user_id = ? ORDER BY created_at",
            (user_id,)
        )
        memories = cursor.fetchall()
        
        if not memories:
            return {"message": "No memory stored."}
        
        content_list = [f"{i+1}. {mem[0]}" for i, mem in enumerate(memories)]
        return {
            "message": f"Memories from the users memory vault: {content_list}"
        }
    
    def delete_memory(self, user_id: str, index: int) -> Dict[str, Any]:
        """Delete a memory by index"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM memories WHERE user_id = ? ORDER BY created_at",
            (user_id,)
        )
        memories = cursor.fetchall()
        
        if index < 1 or index > len(memories):
            return {"success": False, "message": f"Memory index {index} does not exist."}
        
        memory_id = memories[index - 1][0]
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        
        return {"success": True, "message": f"Memory at index {index} deleted successfully."}
    
    # =============================
    # Diary Operations
    # =============================
    
    def add_diary(self, content: str) -> Dict[str, Any]:
        """Add a diary entry"""
        cursor = self.conn.cursor()
        now = datetime.now()
        formatted_content = f"{now.strftime('%B %d, %Y')}\n{content}"
        
        cursor.execute(
            "INSERT INTO diary_pages (content) VALUES (?)",
            (formatted_content,)
        )
        self.conn.commit()
        
        return {
            "success": True,
            "message": "Successfully added 1 page.",
            "diary_id": cursor.lastrowid
        }
    
    def recall_diary(self) -> Dict[str, Any]:
        """Retrieve all diary pages"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT content FROM diary_pages ORDER BY created_at")
        pages = cursor.fetchall()
        
        if not pages:
            return {"message": "No diary stored."}
        
        content_list = [f"{i+1}. {page[0]}" for i, page in enumerate(pages)]
        return {
            "message": f"Pages from the users diary vault: {content_list}"
        }
    
    def delete_diary(self, index: int) -> Dict[str, Any]:
        """Delete a diary page by index"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM diary_pages ORDER BY created_at")
        pages = cursor.fetchall()
        
        if index < 1 or index > len(pages):
            return {"success": False, "message": f"Diary index {index} does not exist."}
        
        page_id = pages[index - 1][0]
        cursor.execute("DELETE FROM diary_pages WHERE id = ?", (page_id,))
        self.conn.commit()
        
        return {"success": True, "message": f"Diary at index {index} deleted successfully."}

    # =============================
    # Reminder Operations
    # =============================

    def _now_iso(self) -> str:
        return datetime.now(ADA_TIMEZONE).isoformat()

    def add_reminder_entry(self, user_id: str, subject: str, body: str) -> Dict[str, Any]:
        """Add a reminder with subject/body"""
        subject = (subject or "").strip()
        body = (body or "").strip()
        if not subject:
            return {"success": False, "message": "Reminder subject cannot be empty."}
        if not body:
            return {"success": False, "message": "Reminder text cannot be empty."}

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (user_id, subject, body) VALUES (?, ?, ?)",
            (user_id, subject, body),
        )
        self.conn.commit()

        return {
            "success": True,
            "message": f"Reminder '{subject}' added.",
            "reminder_id": cursor.lastrowid,
        }

    def list_open_reminders(
        self, user_id: str, update_last_checked: bool = False
    ) -> List[Dict[str, Any]]:
        """Return open reminders, optionally stamping them as checked now"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, subject, body, created_at, last_checked_at
            FROM reminders
            WHERE user_id = ? AND completed_at IS NULL
            ORDER BY created_at
            """,
            (user_id,),
        )
        rows = cursor.fetchall()
        reminders = [
            {
                "id": row[0],
                "subject": row[1],
                "body": row[2],
                "created_at": row[3],
                "last_checked_at": row[4],
            }
            for row in rows
        ]

        if update_last_checked and reminders:
            now = self._now_iso()
            cursor.executemany(
                "UPDATE reminders SET last_checked_at = ? WHERE id = ?",
                [(now, reminder["id"]) for reminder in reminders],
            )
            self.conn.commit()
            for reminder in reminders:
                reminder["last_checked_at"] = now

        return reminders

    def _format_reminder_lines(self, reminders: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for idx, reminder in enumerate(reminders, start=1):
            created = reminder["created_at"] or "Unknown"
            checked = reminder["last_checked_at"] or "Not yet checked"
            lines.append(
                f"{idx}. {reminder['subject']} — {reminder['body']} "
                f"(Created: {created}, Last checked: {checked})"
            )
        return lines

    def recall_reminders_tool(self, user_id: str) -> Dict[str, Any]:
        """Return open reminders for the tool API"""
        reminders = self.list_open_reminders(user_id, update_last_checked=True)
        if not reminders:
            return {"message": "No reminders waiting."}
        lines = self._format_reminder_lines(reminders)
        return {"message": "Open reminders:\n" + "\n".join(lines)}

    def complete_reminder_entry(self, user_id: str, index: int) -> Dict[str, Any]:
        """Mark a reminder complete by its index in the open list"""
        reminders = self.list_open_reminders(user_id)
        if index < 1 or index > len(reminders):
            return {
                "success": False,
                "message": f"Reminder index {index} does not exist.",
            }

        reminder = reminders[index - 1]
        now = self._now_iso()
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE reminders SET completed_at = ?, last_checked_at = ? WHERE id = ?",
            (now, now, reminder["id"]),
        )
        self.conn.commit()
        return {
            "success": True,
            "message": f"Marked '{reminder['subject']}' as complete.",
        }

    def delete_reminder_entry(self, user_id: str, index: int) -> Dict[str, Any]:
        """Delete a reminder by its index in the open list"""
        reminders = self.list_open_reminders(user_id)
        if index < 1 or index > len(reminders):
            return {
                "success": False,
                "message": f"Reminder index {index} does not exist.",
            }

        reminder = reminders[index - 1]
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE id = ?", (reminder["id"],))
        self.conn.commit()
        return {
            "success": True,
            "message": f"Deleted reminder '{reminder['subject']}'.",
        }

    def reminder_bulletin(self, user_id: str) -> Optional[str]:
        """Return a bulletin string for system prompts listing open reminders"""
        reminders = self.list_open_reminders(user_id, update_last_checked=True)
        if not reminders:
            return None
        lines = self._format_reminder_lines(reminders)
        header = "🔔 Reminder bulletin before this chat:"
        return header + "\n" + "\n".join(lines)
    
    # =============================
    # Time Awareness Tool
    # =============================

    def check_watch(self) -> Dict[str, Any]:
        """Return the current date/time in Ada's local timezone (Bozeman/Mountain Time)"""
        now = datetime.now(ADA_TIMEZONE)
        friendly = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
        iso_timestamp = now.isoformat()
        return {
            "message": f"It's currently {friendly} (Bozeman time).",
            "timestamp": iso_timestamp,
        }
    
    # =============================
    # Tool Call Parsing
    # =============================
    
    def _extract_string_arg(self, tool_call: str, arg_name: str) -> Optional[str]:
        """Helper to parse string arguments from tool calls"""
        pattern = rf'{arg_name}\s*=\s*(\".*?\"|\'.*?\')'
        match = re.search(pattern, tool_call, re.DOTALL)
        if not match:
            return None
        try:
            return ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return None

    def parse_and_execute(
        self, 
        response: str, 
        user_id: str = "drew"
    ) -> List[Tuple[str, Any, Dict[str, Any]]]:
        """
        Parse tool calls from LLM response and execute them.
        
        Returns list of (tool_name, args, result) tuples.
        """
        tool_results = []
        
        # Pattern to match: <tool_call>function_name(args)</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                result = self._execute_tool_call(match, user_id)
                if result:
                    tool_results.append(result)
            except Exception as e:
                print(f"⚠️  Error executing tool call: {e}")
                continue
        
        return tool_results
    
    def _execute_tool_call(
        self, 
        tool_call: str, 
        user_id: str
    ) -> Optional[Tuple[str, Any, Dict[str, Any]]]:
        """Execute a single tool call"""
        
        # Memory tools
        if 'add_memory' in tool_call:
            content_match = re.search(r'input_text\s*=\s*\[(.*?)\]', tool_call, re.DOTALL)
            if content_match:
                try:
                    memories = eval(f"[{content_match.group(1)}]")
                    for mem in memories:
                        result = self.add_memory(user_id, mem)
                        return ("add_memory", mem, result)
                except:
                    pass
        
        elif 'recall_memories' in tool_call:
            result = self.recall_memories(user_id)
            return ("recall_memories", None, result)
        
        elif 'delete_memory' in tool_call:
            index_match = re.search(r'indices\s*=\s*\[(\d+)\]', tool_call)
            if index_match:
                index = int(index_match.group(1))
                result = self.delete_memory(user_id, index)
                return ("delete_memory", index, result)
        
        # Diary tools
        elif 'add_diary' in tool_call:
            content_match = re.search(r'input_text\s*=\s*\[(.*?)\]', tool_call, re.DOTALL)
            if content_match:
                try:
                    entries = eval(f"[{content_match.group(1)}]")
                    for entry in entries:
                        result = self.add_diary(entry)
                        return ("add_diary", entry, result)
                except:
                    pass
        
        elif 'recall_diary' in tool_call:
            result = self.recall_diary()
            return ("recall_diary", None, result)
        
        elif 'delete_diary' in tool_call:
            index_match = re.search(r'indices\s*=\s*\[(\d+)\]', tool_call)
            if index_match:
                index = int(index_match.group(1))
                result = self.delete_diary(index)
                return ("delete_diary", index, result)

        elif 'add_reminder' in tool_call:
            subject = self._extract_string_arg(tool_call, "subject")
            reminder_text = self._extract_string_arg(tool_call, "reminder_text")
            if subject and reminder_text:
                result = self.add_reminder_entry(user_id, subject, reminder_text)
                args = {"subject": subject, "reminder_text": reminder_text}
                return ("add_reminder", args, result)

        elif 'recall_reminders' in tool_call:
            result = self.recall_reminders_tool(user_id)
            return ("recall_reminders", None, result)

        elif 'complete_reminder' in tool_call:
            index_match = re.search(r'indices\s*=\s*\[(\d+)\]', tool_call)
            if index_match:
                index = int(index_match.group(1))
                result = self.complete_reminder_entry(user_id, index)
                return ("complete_reminder", index, result)

        elif 'delete_reminder' in tool_call:
            index_match = re.search(r'indices\s*=\s*\[(\d+)\]', tool_call)
            if index_match:
                index = int(index_match.group(1))
                result = self.delete_reminder_entry(user_id, index)
                return ("delete_reminder", index, result)
        
        elif 'check_watch' in tool_call:
            result = self.check_watch()
            return ("check_watch", None, result)
        
        return None
    
    # =============================
    # Cleanup
    # =============================
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM diary_pages")
        diary_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM reminders WHERE completed_at IS NULL")
        reminder_count = cursor.fetchone()[0]
        
        return {
            "memories": memory_count,
            "diary_pages": diary_count,
            "open_reminders": reminder_count,
        }
    
    def reset_database(self):
        """Clear all data (use with caution!)"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories")
        cursor.execute("DELETE FROM diary_pages")
        cursor.execute("DELETE FROM reminders")
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
