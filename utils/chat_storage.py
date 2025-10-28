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

                # File summaries table for intelligent file selection
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS file_summaries (
                        doc_id VARCHAR(255) PRIMARY KEY,
                        course_id VARCHAR(255) NOT NULL,
                        filename TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        topics TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """))

                # PHASE 3: Gemini File API URI cache (48hr expiration)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS gemini_file_cache (
                        file_path VARCHAR(512) PRIMARY KEY,
                        course_id VARCHAR(255) NOT NULL,
                        filename TEXT NOT NULL,
                        gemini_uri TEXT NOT NULL,
                        gemini_name TEXT NOT NULL,
                        mime_type VARCHAR(100),
                        size_bytes BIGINT,
                        uploaded_at TIMESTAMP DEFAULT NOW(),
                        expires_at TIMESTAMP NOT NULL
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

                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_summaries_course
                    ON file_summaries(course_id)
                """))

                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_gemini_cache_course
                    ON gemini_file_cache(course_id)
                """))

                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_gemini_cache_expires
                    ON gemini_file_cache(expires_at)
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

            # File summaries table for intelligent file selection
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_summaries (
                    doc_id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    topics TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # PHASE 3: Gemini File API URI cache (48hr expiration)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gemini_file_cache (
                    file_path TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    gemini_uri TEXT NOT NULL,
                    gemini_name TEXT NOT NULL,
                    mime_type TEXT,
                    size_bytes INTEGER,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
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

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_course
                ON file_summaries(course_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_cache_course
                ON gemini_file_cache(course_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_gemini_cache_expires
                ON gemini_file_cache(expires_at)
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

    def save_file_summary(
        self,
        doc_id: str,
        course_id: str,
        filename: str,
        summary: str,
        topics: List[str] = None,
        metadata: Dict = None
    ) -> bool:
        """
        Save or update a file summary

        Args:
            doc_id: Document identifier
            course_id: Course identifier
            filename: Original filename
            summary: Text summary of the file
            topics: List of key topics/keywords
            metadata: Additional metadata

        Returns:
            True if successful
        """
        try:
            topics_json = json.dumps(topics or [])
            metadata_json = json.dumps(metadata or {})

            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO file_summaries (doc_id, course_id, filename, summary, topics, metadata, updated_at)
                        VALUES (:doc_id, :course_id, :filename, :summary, :topics, :metadata, NOW())
                        ON CONFLICT (doc_id) DO UPDATE SET
                            summary = EXCLUDED.summary,
                            topics = EXCLUDED.topics,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                    """), {
                        "doc_id": doc_id,
                        "course_id": course_id,
                        "filename": filename,
                        "summary": summary,
                        "topics": topics_json,
                        "metadata": metadata_json
                    })
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO file_summaries (doc_id, course_id, filename, summary, topics, metadata, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(doc_id) DO UPDATE SET
                            summary = excluded.summary,
                            topics = excluded.topics,
                            metadata = excluded.metadata,
                            updated_at = CURRENT_TIMESTAMP
                    """, (doc_id, course_id, filename, summary, topics_json, metadata_json))
                    conn.commit()

            logger.info(f"Saved file summary for {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving file summary: {e}")
            return False

    def get_file_summary(self, doc_id: str) -> Optional[Dict]:
        """Get a file summary by doc_id"""
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT * FROM file_summaries WHERE doc_id = :doc_id
                    """), {"doc_id": doc_id})
                    row = result.fetchone()
                    if row:
                        data = dict(row._mapping)
                        data['topics'] = json.loads(data.get('topics', '[]'))
                        data['metadata'] = json.loads(data.get('metadata', '{}'))
                        return data
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM file_summaries WHERE doc_id = ?
                    """, (doc_id,))
                    row = cursor.fetchone()
                    if row:
                        data = dict(row)
                        data['topics'] = json.loads(data.get('topics', '[]'))
                        data['metadata'] = json.loads(data.get('metadata', '{}'))
                        return data

            return None

        except Exception as e:
            logger.error(f"Error retrieving file summary: {e}")
            return None

    def get_all_summaries_for_course(self, course_id: str) -> List[Dict]:
        """Get all file summaries for a course"""
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT * FROM file_summaries WHERE course_id = :course_id
                    """), {"course_id": course_id})
                    summaries = []
                    for row in result:
                        data = dict(row._mapping)
                        data['topics'] = json.loads(data.get('topics', '[]'))
                        data['metadata'] = json.loads(data.get('metadata', '{}'))
                        summaries.append(data)
                    return summaries
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM file_summaries WHERE course_id = ?
                    """, (course_id,))
                    summaries = []
                    for row in cursor.fetchall():
                        data = dict(row)
                        data['topics'] = json.loads(data.get('topics', '[]'))
                        data['metadata'] = json.loads(data.get('metadata', '{}'))
                        summaries.append(data)
                    return summaries

        except Exception as e:
            logger.error(f"Error retrieving summaries for course: {e}")
            return []

    # ========== PHASE 3: Gemini File API URI Cache ==========

    def save_gemini_uri(
        self,
        file_path: str,
        course_id: str,
        filename: str,
        gemini_uri: str,
        gemini_name: str,
        mime_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        expires_hours: int = 48
    ) -> bool:
        """
        Save Gemini File API URI to cache (48 hour expiration)

        PHASE 3: Pre-warm Gemini cache to eliminate query-time upload wait

        Args:
            file_path: GCS blob path or local file path (unique key)
            course_id: Course identifier
            filename: Original filename
            gemini_uri: Gemini File API URI (e.g., "https://generativelanguage.googleapis.com/v1beta/files/...")
            gemini_name: Gemini file name (e.g., "files/abc123")
            mime_type: MIME type
            size_bytes: File size in bytes
            expires_hours: Expiration time in hours (default 48, max for Gemini File API)

        Returns:
            True if successful
        """
        try:
            from datetime import datetime, timedelta

            expires_at = datetime.now() + timedelta(hours=expires_hours)

            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO gemini_file_cache (file_path, course_id, filename, gemini_uri, gemini_name, mime_type, size_bytes, expires_at)
                        VALUES (:file_path, :course_id, :filename, :gemini_uri, :gemini_name, :mime_type, :size_bytes, :expires_at)
                        ON CONFLICT (file_path) DO UPDATE SET
                            gemini_uri = EXCLUDED.gemini_uri,
                            gemini_name = EXCLUDED.gemini_name,
                            mime_type = EXCLUDED.mime_type,
                            size_bytes = EXCLUDED.size_bytes,
                            uploaded_at = NOW(),
                            expires_at = EXCLUDED.expires_at
                    """), {
                        "file_path": file_path,
                        "course_id": course_id,
                        "filename": filename,
                        "gemini_uri": gemini_uri,
                        "gemini_name": gemini_name,
                        "mime_type": mime_type,
                        "size_bytes": size_bytes,
                        "expires_at": expires_at
                    })
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO gemini_file_cache (file_path, course_id, filename, gemini_uri, gemini_name, mime_type, size_bytes, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (file_path, course_id, filename, gemini_uri, gemini_name, mime_type, size_bytes, expires_at))
                    conn.commit()

            logger.info(f"✅ Cached Gemini URI for {filename} (expires in {expires_hours}h)")
            return True

        except Exception as e:
            logger.error(f"Failed to save Gemini URI for {filename}: {e}")
            return False

    def get_gemini_uri(self, file_path: str) -> Optional[Dict]:
        """
        Get cached Gemini File API URI if still valid

        PHASE 3: Retrieve cached URI to skip upload during queries

        Args:
            file_path: GCS blob path or local file path

        Returns:
            Dict with gemini_uri, gemini_name, etc. if cached and valid, None otherwise
        """
        try:
            from datetime import datetime

            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT * FROM gemini_file_cache
                        WHERE file_path = :file_path AND expires_at > NOW()
                    """), {"file_path": file_path})
                    row = result.fetchone()
                    if row:
                        return dict(row._mapping)
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM gemini_file_cache
                        WHERE file_path = ? AND expires_at > CURRENT_TIMESTAMP
                    """, (file_path,))
                    row = cursor.fetchone()
                    if row:
                        return dict(row)

            return None

        except Exception as e:
            logger.error(f"Error retrieving Gemini URI for {file_path}: {e}")
            return None

    def get_course_gemini_uris(self, course_id: str) -> List[Dict]:
        """
        Get all valid Gemini URIs for a course

        Args:
            course_id: Course identifier

        Returns:
            List of dicts with file info and Gemini URIs
        """
        try:
            from datetime import datetime

            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT * FROM gemini_file_cache
                        WHERE course_id = :course_id AND expires_at > NOW()
                        ORDER BY filename
                    """), {"course_id": course_id})
                    return [dict(row._mapping) for row in result.fetchall()]
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM gemini_file_cache
                        WHERE course_id = ? AND expires_at > CURRENT_TIMESTAMP
                        ORDER BY filename
                    """, (course_id,))
                    return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error retrieving Gemini URIs for course {course_id}: {e}")
            return []

    def cleanup_expired_gemini_uris(self) -> int:
        """
        Remove expired Gemini URIs from cache

        Returns:
            Number of expired entries deleted
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        DELETE FROM gemini_file_cache WHERE expires_at <= NOW()
                    """))
                    conn.commit()
                    deleted = result.rowcount
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        DELETE FROM gemini_file_cache WHERE expires_at <= CURRENT_TIMESTAMP
                    """)
                    conn.commit()
                    deleted = cursor.rowcount

            if deleted > 0:
                logger.info(f"🧹 Cleaned up {deleted} expired Gemini URI(s)")
            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up expired Gemini URIs: {e}")
            return 0
