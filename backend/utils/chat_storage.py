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
                        updated_at TIMESTAMP DEFAULT NOW(),
                        deleted_at TIMESTAMP NULL
                    )
                """))

                # Migration: Add deleted_at column if it doesn't exist (for existing tables)
                # Use a separate transaction to avoid breaking the main initialization
                try:
                    # Check if column exists first
                    check_result = conn.execute(text("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name='file_summaries'
                        AND column_name='deleted_at'
                    """))

                    if not check_result.fetchone():
                        # Column doesn't exist, add it
                        conn.execute(text("""
                            ALTER TABLE file_summaries ADD COLUMN deleted_at TIMESTAMP NULL
                        """))
                        logger.info("Added deleted_at column to file_summaries (PostgreSQL)")
                except Exception as e:
                    # If any error occurs, rollback this specific operation
                    logger.warning(f"Could not add deleted_at column (may already exist): {e}")
                    conn.rollback()
                    # Continue with the rest of initialization

                # Course metadata table (stores syllabus_id, etc.)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS course_metadata (
                        course_id VARCHAR(255) PRIMARY KEY,
                        syllabus_id VARCHAR(255),
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
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP NULL
                )
            """)

            # Migration: Add deleted_at column if it doesn't exist (for existing tables)
            try:
                # Check if column exists first
                cursor.execute("""
                    SELECT COUNT(*) FROM pragma_table_info('file_summaries')
                    WHERE name='deleted_at'
                """)
                exists = cursor.fetchone()[0] > 0

                if not exists:
                    # Column doesn't exist, add it
                    cursor.execute("""
                        ALTER TABLE file_summaries ADD COLUMN deleted_at TIMESTAMP NULL
                    """)
                    logger.info("Added deleted_at column to file_summaries (SQLite)")
            except Exception as e:
                # If any error occurs, just continue
                logger.warning(f"Could not add deleted_at column (may already exist): {e}")

            # Course metadata table (stores syllabus_id, etc.)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS course_metadata (
                    course_id TEXT PRIMARY KEY,
                    syllabus_id TEXT,
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
        """Get all file summaries for a course (excludes soft-deleted files)"""
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT * FROM file_summaries
                        WHERE course_id = :course_id AND deleted_at IS NULL
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
                        SELECT * FROM file_summaries
                        WHERE course_id = ? AND deleted_at IS NULL
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

    def delete_gemini_cache_entry(self, file_path: str) -> bool:
        """
        Delete a specific Gemini cache entry by file_path

        Args:
            file_path: GCS blob path (e.g., "course_id/filename.pdf")

        Returns:
            True if deleted, False if not found or error
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        DELETE FROM gemini_file_cache WHERE file_path = :file_path
                    """), {"file_path": file_path})
                    conn.commit()
                    deleted = result.rowcount > 0
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        DELETE FROM gemini_file_cache WHERE file_path = ?
                    """, (file_path,))
                    conn.commit()
                    deleted = cursor.rowcount > 0

            if deleted:
                logger.info(f"🗑️  Deleted Gemini cache entry: {file_path}")
            return deleted

        except Exception as e:
            logger.error(f"Error deleting Gemini cache entry for {file_path}: {e}")
            return False

    def get_all_cached_files_by_course(self) -> Dict[str, List[str]]:
        """
        Get all cached file paths grouped by course_id

        Returns:
            Dict mapping course_id -> [file_path1, file_path2, ...]
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT course_id, file_path FROM gemini_file_cache
                        WHERE expires_at > NOW()
                    """))
                    rows = result.fetchall()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT course_id, file_path FROM gemini_file_cache
                        WHERE expires_at > CURRENT_TIMESTAMP
                    """)
                    rows = cursor.fetchall()

            # Group by course_id
            cache_by_course = {}
            for row in rows:
                course_id = row[0]
                file_path = row[1]
                if course_id not in cache_by_course:
                    cache_by_course[course_id] = []
                cache_by_course[course_id].append(file_path)

            return cache_by_course

        except Exception as e:
            logger.error(f"Error getting cached files by course: {e}")
            return {}

    def cleanup_duplicate_file_summaries(self, course_id: str, dry_run: bool = True) -> Dict:
        """
        Find and remove duplicate file summaries (same course_id + filename)
        Keeps the most recently updated entry

        Args:
            course_id: Course identifier
            dry_run: If True, only report duplicates without deleting

        Returns:
            {
                'duplicates_found': int,
                'deleted_count': int,
                'deleted_entries': [...],
                'errors': [...]
            }
        """
        try:
            duplicates_found = 0
            deleted_entries = []
            errors = []

            if self.use_postgres:
                with self.engine.connect() as conn:
                    # Find duplicates: same course_id + filename, not deleted
                    result = conn.execute(text("""
                        SELECT course_id, filename, COUNT(*) as cnt,
                               array_agg(doc_id ORDER BY updated_at DESC) as doc_ids,
                               array_agg(updated_at ORDER BY updated_at DESC) as updated_ats
                        FROM file_summaries
                        WHERE course_id = :course_id AND deleted_at IS NULL
                        GROUP BY course_id, filename
                        HAVING COUNT(*) > 1
                    """), {"course_id": course_id})
                    duplicates = result.fetchall()

                    for dup in duplicates:
                        filename = dup[1]
                        count = dup[2]
                        doc_ids = dup[3]
                        updated_ats = dup[4]

                        duplicates_found += count - 1  # All but the newest

                        # Keep first (newest), delete rest
                        keep_id = doc_ids[0]
                        delete_ids = doc_ids[1:]

                        logger.info(f"Found {count} duplicates of '{filename}':")
                        logger.info(f"  Keeping: {keep_id} (updated: {updated_ats[0]})")
                        for i, delete_id in enumerate(delete_ids):
                            logger.info(f"  Deleting: {delete_id} (updated: {updated_ats[i+1]})")
                            deleted_entries.append({
                                'doc_id': delete_id,
                                'filename': filename,
                                'updated_at': str(updated_ats[i+1])
                            })

                        if not dry_run:
                            # Soft delete (set deleted_at timestamp)
                            for delete_id in delete_ids:
                                try:
                                    conn.execute(text("""
                                        UPDATE file_summaries
                                        SET deleted_at = NOW()
                                        WHERE doc_id = :doc_id
                                    """), {"doc_id": delete_id})
                                except Exception as e:
                                    errors.append({
                                        'doc_id': delete_id,
                                        'error': str(e)
                                    })
                            conn.commit()

            else:  # SQLite
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Find duplicates
                    cursor.execute("""
                        SELECT course_id, filename, COUNT(*) as cnt
                        FROM file_summaries
                        WHERE course_id = ? AND deleted_at IS NULL
                        GROUP BY course_id, filename
                        HAVING COUNT(*) > 1
                    """, (course_id,))
                    dup_files = cursor.fetchall()

                    for dup in dup_files:
                        filename = dup[1]
                        count = dup[2]

                        # Get all doc_ids for this filename
                        cursor.execute("""
                            SELECT doc_id, updated_at
                            FROM file_summaries
                            WHERE course_id = ? AND filename = ? AND deleted_at IS NULL
                            ORDER BY updated_at DESC
                        """, (course_id, filename))
                        entries = cursor.fetchall()

                        duplicates_found += len(entries) - 1

                        # Keep first (newest), delete rest
                        keep_id = entries[0][0]
                        delete_entries = entries[1:]

                        logger.info(f"Found {count} duplicates of '{filename}':")
                        logger.info(f"  Keeping: {keep_id} (updated: {entries[0][1]})")
                        for delete_entry in delete_entries:
                            delete_id = delete_entry[0]
                            updated_at = delete_entry[1]
                            logger.info(f"  Deleting: {delete_id} (updated: {updated_at})")
                            deleted_entries.append({
                                'doc_id': delete_id,
                                'filename': filename,
                                'updated_at': updated_at
                            })

                            if not dry_run:
                                try:
                                    cursor.execute("""
                                        UPDATE file_summaries
                                        SET deleted_at = CURRENT_TIMESTAMP
                                        WHERE doc_id = ?
                                    """, (delete_id,))
                                except Exception as e:
                                    errors.append({
                                        'doc_id': delete_id,
                                        'error': str(e)
                                    })
                    if not dry_run:
                        conn.commit()

            return {
                'duplicates_found': duplicates_found,
                'deleted_count': len(deleted_entries) if not dry_run else 0,
                'deleted_entries': deleted_entries,
                'errors': errors
            }

        except Exception as e:
            logger.error(f"Error cleaning up duplicate file summaries: {e}")
            return {
                'duplicates_found': 0,
                'deleted_count': 0,
                'deleted_entries': [],
                'errors': [{'error': str(e)}]
            }

    def get_files_needing_cache_refresh(self, hours_before_expiry: int = 6) -> List[Dict]:
        """
        Get files that will expire soon and need cache refresh

        Args:
            hours_before_expiry: Refresh files expiring within this many hours

        Returns:
            List of dicts with file_path, course_id, filename, expires_at
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT file_path, course_id, filename, gemini_uri, gemini_name,
                               mime_type, size_bytes, expires_at
                        FROM gemini_file_cache
                        WHERE expires_at > NOW()
                        AND expires_at <= NOW() + INTERVAL '1 hour' * :hours
                        ORDER BY expires_at ASC
                    """), {"hours": hours_before_expiry})
                    rows = result.fetchall()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT file_path, course_id, filename, gemini_uri, gemini_name,
                               mime_type, size_bytes, expires_at
                        FROM gemini_file_cache
                        WHERE expires_at > CURRENT_TIMESTAMP
                        AND expires_at <= datetime(CURRENT_TIMESTAMP, '+' || ? || ' hours')
                        ORDER BY expires_at ASC
                    """, (hours_before_expiry,))
                    rows = cursor.fetchall()

            files = []
            for row in rows:
                files.append({
                    'file_path': row[0],
                    'course_id': row[1],
                    'filename': row[2],
                    'gemini_uri': row[3],
                    'gemini_name': row[4],
                    'mime_type': row[5],
                    'size_bytes': row[6],
                    'expires_at': row[7]
                })

            return files

        except Exception as e:
            logger.error(f"Error getting files needing refresh: {e}")
            return []

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about Gemini cache

        Returns:
            Dict with total_files, total_bytes, courses_count, expiring_soon_count
        """
        try:
            stats = {
                'total_files': 0,
                'total_bytes': 0,
                'courses_count': 0,
                'expiring_soon_count': 0,
                'expired_count': 0
            }

            if self.use_postgres:
                with self.engine.connect() as conn:
                    # Total files and bytes
                    result = conn.execute(text("""
                        SELECT COUNT(*), COALESCE(SUM(size_bytes), 0)
                        FROM gemini_file_cache
                        WHERE expires_at > NOW()
                    """))
                    row = result.fetchone()
                    stats['total_files'] = row[0]
                    stats['total_bytes'] = row[1]

                    # Unique courses
                    result = conn.execute(text("""
                        SELECT COUNT(DISTINCT course_id) FROM gemini_file_cache
                        WHERE expires_at > NOW()
                    """))
                    stats['courses_count'] = result.fetchone()[0]

                    # Expiring soon (within 6 hours)
                    result = conn.execute(text("""
                        SELECT COUNT(*) FROM gemini_file_cache
                        WHERE expires_at > NOW() AND expires_at <= NOW() + INTERVAL '6 hours'
                    """))
                    stats['expiring_soon_count'] = result.fetchone()[0]

                    # Already expired
                    result = conn.execute(text("""
                        SELECT COUNT(*) FROM gemini_file_cache
                        WHERE expires_at <= NOW()
                    """))
                    stats['expired_count'] = result.fetchone()[0]
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Total files and bytes
                    cursor.execute("""
                        SELECT COUNT(*), COALESCE(SUM(size_bytes), 0)
                        FROM gemini_file_cache
                        WHERE expires_at > CURRENT_TIMESTAMP
                    """)
                    row = cursor.fetchone()
                    stats['total_files'] = row[0]
                    stats['total_bytes'] = row[1]

                    # Unique courses
                    cursor.execute("""
                        SELECT COUNT(DISTINCT course_id) FROM gemini_file_cache
                        WHERE expires_at > CURRENT_TIMESTAMP
                    """)
                    stats['courses_count'] = cursor.fetchone()[0]

                    # Expiring soon
                    cursor.execute("""
                        SELECT COUNT(*) FROM gemini_file_cache
                        WHERE expires_at > CURRENT_TIMESTAMP
                        AND expires_at <= datetime(CURRENT_TIMESTAMP, '+6 hours')
                    """)
                    stats['expiring_soon_count'] = cursor.fetchone()[0]

                    # Already expired
                    cursor.execute("""
                        SELECT COUNT(*) FROM gemini_file_cache
                        WHERE expires_at <= CURRENT_TIMESTAMP
                    """)
                    stats['expired_count'] = cursor.fetchone()[0]

            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def set_course_syllabus(self, course_id: str, syllabus_id: str):
        """
        Set the syllabus document ID for a course

        Args:
            course_id: Course identifier
            syllabus_id: Document ID of the syllabus
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO course_metadata (course_id, syllabus_id, updated_at)
                        VALUES (:course_id, :syllabus_id, NOW())
                        ON CONFLICT (course_id) DO UPDATE SET
                            syllabus_id = :syllabus_id,
                            updated_at = NOW()
                    """), {"course_id": course_id, "syllabus_id": syllabus_id})
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO course_metadata (course_id, syllabus_id, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (course_id, syllabus_id))
                    conn.commit()

            logger.info(f"✅ Set syllabus for course {course_id}: {syllabus_id}")

        except Exception as e:
            logger.error(f"Error setting course syllabus: {e}")

    def get_course_syllabus(self, course_id: str) -> Optional[str]:
        """
        Get the syllabus document ID for a course

        Args:
            course_id: Course identifier

        Returns:
            Syllabus document ID or None
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT syllabus_id FROM course_metadata WHERE course_id = :course_id
                    """), {"course_id": course_id})
                    row = result.fetchone()
                    return row[0] if row else None
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT syllabus_id FROM course_metadata WHERE course_id = ?
                    """, (course_id,))
                    row = cursor.fetchone()
                    return row[0] if row else None

        except Exception as e:
            logger.error(f"Error getting course syllabus: {e}")
            return None

    # ========== Soft Delete Operations ==========

    def soft_delete_file(self, doc_id: str) -> bool:
        """
        Soft delete a file by setting deleted_at timestamp

        Args:
            doc_id: Document identifier

        Returns:
            True if successful
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE file_summaries
                        SET deleted_at = NOW()
                        WHERE doc_id = :doc_id
                    """), {"doc_id": doc_id})
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE file_summaries
                        SET deleted_at = CURRENT_TIMESTAMP
                        WHERE doc_id = ?
                    """, (doc_id,))
                    conn.commit()

            logger.info(f"Soft deleted file {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error soft deleting file: {e}")
            return False

    def get_deleted_files(self, course_id: str) -> List[Dict]:
        """
        Get all soft-deleted files for a course

        Args:
            course_id: Course identifier

        Returns:
            List of deleted file dicts
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT doc_id, filename, deleted_at, metadata
                        FROM file_summaries
                        WHERE course_id = :course_id AND deleted_at IS NOT NULL
                        ORDER BY deleted_at DESC
                    """), {"course_id": course_id})
                    files = []
                    for row in result:
                        data = dict(row._mapping)
                        data['metadata'] = json.loads(data.get('metadata', '{}'))
                        files.append(data)
                    return files
            else:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT doc_id, filename, deleted_at, metadata
                        FROM file_summaries
                        WHERE course_id = ? AND deleted_at IS NOT NULL
                        ORDER BY deleted_at DESC
                    """, (course_id,))
                    files = []
                    for row in cursor.fetchall():
                        data = dict(row)
                        data['metadata'] = json.loads(data.get('metadata', '{}'))
                        files.append(data)
                    return files

        except Exception as e:
            logger.error(f"Error getting deleted files: {e}")
            return []

    def restore_file(self, doc_id: str) -> bool:
        """
        Restore a soft-deleted file by clearing deleted_at timestamp

        Args:
            doc_id: Document identifier

        Returns:
            True if successful
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE file_summaries
                        SET deleted_at = NULL
                        WHERE doc_id = :doc_id
                    """), {"doc_id": doc_id})
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE file_summaries
                        SET deleted_at = NULL
                        WHERE doc_id = ?
                    """, (doc_id,))
                    conn.commit()

            logger.info(f"Restored file {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error restoring file: {e}")
            return False

    def hard_delete_file(self, doc_id: str) -> bool:
        """
        Permanently delete a file from database

        Args:
            doc_id: Document identifier

        Returns:
            True if successful
        """
        try:
            if self.use_postgres:
                with self.engine.connect() as conn:
                    conn.execute(text("""
                        DELETE FROM file_summaries
                        WHERE doc_id = :doc_id
                    """), {"doc_id": doc_id})
                    conn.commit()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        DELETE FROM file_summaries
                        WHERE doc_id = ?
                    """, (doc_id,))
                    conn.commit()

            logger.info(f"Hard deleted file {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error hard deleting file: {e}")
            return False
