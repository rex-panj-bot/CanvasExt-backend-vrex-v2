"""
Chat Storage Manager
Handles persistence of chat sessions to SQLite database
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChatStorage:
    """Manages chat history storage in SQLite"""

    def __init__(self, db_path: str = "./data/chats.db"):
        """
        Initialize chat storage

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Chat sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0
                )
            """)

            # Chat messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
            """)

            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_course
                ON chat_sessions(course_id, updated_at DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id, timestamp)
            """)

            conn.commit()
            logger.info("Chat storage database initialized")

    def save_chat_session(
        self,
        session_id: str,
        course_id: str,
        messages: List[Dict],
        title: Optional[str] = None
    ) -> bool:
        """
        Save or update a chat session

        Args:
            session_id: Unique session identifier
            course_id: Course identifier
            messages: List of message dicts with 'role' and 'content'
            title: Optional chat title (auto-generated from first message if None)

        Returns:
            True if successful
        """
        try:
            # Auto-generate title from first user message if not provided
            if not title and messages:
                first_user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                if first_user_msg:
                    content = first_user_msg.get('content', '')
                    title = content[:50] + ('...' if len(content) > 50 else '')

            title = title or "New Chat"

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Insert or update session
                cursor.execute("""
                    INSERT INTO chat_sessions (session_id, course_id, title, message_count, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        title = excluded.title,
                        message_count = excluded.message_count,
                        updated_at = CURRENT_TIMESTAMP
                """, (session_id, course_id, title, len(messages)))

                # Delete existing messages for this session (we'll re-insert all)
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))

                # Insert all messages
                for msg in messages:
                    cursor.execute("""
                        INSERT INTO chat_messages (session_id, role, content)
                        VALUES (?, ?, ?)
                    """, (session_id, msg.get('role'), msg.get('content')))

                conn.commit()
                logger.info(f"Saved chat session {session_id} with {len(messages)} messages")
                return True

        except Exception as e:
            logger.error(f"Error saving chat session: {e}")
            return False

    def get_chat_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a chat session by ID

        Args:
            session_id: Session identifier

        Returns:
            Dict with session info and messages, or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get session info
                cursor.execute("""
                    SELECT * FROM chat_sessions WHERE session_id = ?
                """, (session_id,))

                session_row = cursor.fetchone()
                if not session_row:
                    return None

                session = dict(session_row)

                # Get messages
                cursor.execute("""
                    SELECT role, content, timestamp
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))

                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'role': row['role'],
                        'content': row['content'],
                        'timestamp': row['timestamp']
                    })

                session['messages'] = messages
                return session

        except Exception as e:
            logger.error(f"Error retrieving chat session: {e}")
            return None

    def get_recent_chats(self, course_id: str, limit: int = 20) -> List[Dict]:
        """
        Get recent chat sessions for a course

        Args:
            course_id: Course identifier
            limit: Maximum number of sessions to return

        Returns:
            List of session dicts (without full message content)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        session_id,
                        course_id,
                        title,
                        created_at,
                        updated_at,
                        message_count
                    FROM chat_sessions
                    WHERE course_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (course_id, limit))

                sessions = []
                for row in cursor.fetchall():
                    sessions.append(dict(row))

                return sessions

        except Exception as e:
            logger.error(f"Error retrieving recent chats: {e}")
            return []

    def delete_chat_session(self, session_id: str) -> bool:
        """
        Delete a chat session and all its messages

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                logger.info(f"Deleted chat session {session_id}")
                return True

        except Exception as e:
            logger.error(f"Error deleting chat session: {e}")
            return False

    def update_chat_title(self, session_id: str, title: str) -> bool:
        """
        Update the title of a chat session

        Args:
            session_id: Session identifier
            title: New title

        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE chat_sessions
                    SET title = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                """, (title, session_id))
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error updating chat title: {e}")
            return False

    def get_all_courses_with_chats(self) -> List[str]:
        """
        Get list of all course IDs that have chat sessions

        Returns:
            List of course IDs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT course_id FROM chat_sessions")
                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error retrieving courses with chats: {e}")
            return []
