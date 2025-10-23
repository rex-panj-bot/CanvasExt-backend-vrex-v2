"""
Chat Storage Manager
Handles persistence of chat sessions to database (PostgreSQL or SQLite)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging
from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class ChatStorage:
    """Manages chat history storage in PostgreSQL or SQLite"""

    def __init__(self, database_url: Optional[str] = None, db_path: str = "./data/chats.db"):
        """
        Initialize chat storage

        Args:
            database_url: PostgreSQL connection string (e.g., postgresql://user:pass@host:5432/db)
                         If provided, uses PostgreSQL. Otherwise falls back to SQLite.
            db_path: Path to SQLite database file (used only if database_url is None)
        """
        self.use_postgres = False
        self.engine = None

        # Try PostgreSQL first if database_url provided
        if database_url and database_url.startswith("postgresql"):
            try:
                self.database_url = database_url
                self.engine = create_engine(
                    database_url,
                    poolclass=pool.NullPool,  # Simple pooling for serverless
                    echo=False,
                    connect_args={"connect_timeout": 5}  # 5 second timeout
                )
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.use_postgres = True
                logger.info(f"✅ ChatStorage initialized with PostgreSQL")
            except Exception as e:
                logger.warning(f"⚠️  PostgreSQL connection failed ({e}), falling back to SQLite")
                self.engine = None
                self.use_postgres = False

        # Fall back to SQLite if PostgreSQL not available
        if not self.use_postgres:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ ChatStorage initialized with SQLite at {db_path}")

        self._initialize_db()

    def _initialize_db(self):
        """Create database tables if they don't exist"""
        if self.use_postgres:
            self._initialize_postgres()
        else:
            self._initialize_sqlite()

    def _initialize_postgres(self):
        """Create PostgreSQL tables"""
        try:
            with self.engine.connect() as conn:
                # Chat sessions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        course_id VARCHAR(255) NOT NULL,
                        title TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        message_count INTEGER DEFAULT 0
                    )
                """))

                # Chat messages table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                    )
                """))

                # Create indices for faster queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_course
                    ON chat_sessions(course_id, updated_at DESC)
                """))

                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON chat_messages(session_id, timestamp)
                """))

                conn.commit()
                logger.info("PostgreSQL chat storage tables initialized")
        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise

    def _initialize_sqlite(self):
        """Create SQLite tables"""
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
            logger.info("SQLite chat storage database initialized")

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
        if self.use_postgres:
            return self._save_chat_session_postgres(session_id, course_id, messages, title)
        else:
            return self._save_chat_session_sqlite(session_id, course_id, messages, title)

    def _save_chat_session_postgres(self, session_id, course_id, messages, title):
        """Save chat session to PostgreSQL"""
        try:
            # Auto-generate title
            if not title and messages:
                first_user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                if first_user_msg:
                    content = first_user_msg.get('content', '')
                    title = content[:50] + ('...' if len(content) > 50 else '')
            title = title or "New Chat"

            with self.engine.connect() as conn:
                # Insert or update session (PostgreSQL ON CONFLICT syntax)
                conn.execute(text("""
                    INSERT INTO chat_sessions (session_id, course_id, title, message_count, updated_at)
                    VALUES (:session_id, :course_id, :title, :count, NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        message_count = EXCLUDED.message_count,
                        updated_at = NOW()
                """), {"session_id": session_id, "course_id": course_id, "title": title, "count": len(messages)})

                # Delete existing messages
                conn.execute(text("DELETE FROM chat_messages WHERE session_id = :session_id"),
                           {"session_id": session_id})

                # Insert all messages
                for msg in messages:
                    conn.execute(text("""
                        INSERT INTO chat_messages (session_id, role, content)
                        VALUES (:session_id, :role, :content)
                    """), {"session_id": session_id, "role": msg.get('role'), "content": msg.get('content')})

                conn.commit()
                logger.info(f"Saved chat session {session_id} with {len(messages)} messages (PostgreSQL)")
                return True

        except Exception as e:
            logger.error(f"Error saving chat session (PostgreSQL): {e}")
            return False

    def _save_chat_session_sqlite(self, session_id, course_id, messages, title):
        """Save chat session to SQLite"""
        try:
            # Auto-generate title
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

                # Delete existing messages
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))

                # Insert all messages
                for msg in messages:
                    cursor.execute("""
                        INSERT INTO chat_messages (session_id, role, content)
                        VALUES (?, ?, ?)
                    """, (session_id, msg.get('role'), msg.get('content')))

                conn.commit()
                logger.info(f"Saved chat session {session_id} with {len(messages)} messages (SQLite)")
                return True

        except Exception as e:
            logger.error(f"Error saving chat session (SQLite): {e}")
            return False

    def get_chat_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a chat session by ID

        Args:
            session_id: Session identifier

        Returns:
            Dict with session info and messages, or None if not found
        """
        if self.use_postgres:
            return self._get_chat_session_postgres(session_id)
        else:
            return self._get_chat_session_sqlite(session_id)

    def _get_chat_session_postgres(self, session_id):
        """Get chat session from PostgreSQL"""
        try:
            with self.engine.connect() as conn:
                # Get session info
                result = conn.execute(text("""
                    SELECT * FROM chat_sessions WHERE session_id = :session_id
                """), {"session_id": session_id})

                session_row = result.fetchone()
                if not session_row:
                    return None

                session = dict(session_row._mapping)

                # Get messages
                result = conn.execute(text("""
                    SELECT role, content, timestamp
                    FROM chat_messages
                    WHERE session_id = :session_id
                    ORDER BY timestamp ASC
                """), {"session_id": session_id})

                messages = []
                for row in result:
                    messages.append({
                        'role': row.role,
                        'content': row.content,
                        'timestamp': str(row.timestamp)
                    })

                session['messages'] = messages
                return session

        except Exception as e:
            logger.error(f"Error retrieving chat session (PostgreSQL): {e}")
            return None

    def _get_chat_session_sqlite(self, session_id):
        """Get chat session from SQLite"""
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
            logger.error(f"Error retrieving chat session (SQLite): {e}")
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
        if self.use_postgres:
            return self._get_recent_chats_postgres(course_id, limit)
        else:
            return self._get_recent_chats_sqlite(course_id, limit)

    def _get_recent_chats_postgres(self, course_id, limit):
        """Get recent chats from PostgreSQL"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT
                        session_id,
                        course_id,
                        title,
                        created_at,
                        updated_at,
                        message_count
                    FROM chat_sessions
                    WHERE course_id = :course_id
                    ORDER BY updated_at DESC
                    LIMIT :limit
                """), {"course_id": course_id, "limit": limit})

                sessions = []
                for row in result:
                    sessions.append(dict(row._mapping))

                return sessions

        except Exception as e:
            logger.error(f"Error retrieving recent chats (PostgreSQL): {e}")
            return []

    def _get_recent_chats_sqlite(self, course_id, limit):
        """Get recent chats from SQLite"""
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
            logger.error(f"Error retrieving recent chats (SQLite): {e}")
            return []

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("DELETE FROM chat_sessions WHERE session_id = :session_id"),
                               {"session_id": session_id})
                    conn.commit()
            else:
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
        """Update the title of a chat session"""
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE chat_sessions
                        SET title = :title, updated_at = NOW()
                        WHERE session_id = :session_id
                    """), {"title": title, "session_id": session_id})
                    conn.commit()
            else:
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
        """Get list of all course IDs that have chat sessions"""
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT DISTINCT course_id FROM chat_sessions"))
                    return [row.course_id for row in result]
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT course_id FROM chat_sessions")
                    return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error retrieving courses with chats: {e}")
            return []
