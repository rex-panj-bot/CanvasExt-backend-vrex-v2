# CLAUDE.md - Canvas Extension Backend

This file provides guidance to Claude Code when working with the Python backend code.

## Project Overview

Python backend server for Canvas LMS AI Study Assistant. Provides AI-powered document processing using Google Gemini 2.5 Flash with native PDF reading, intelligent file selection, and chat history persistence.

## Architecture

This is the **backend component only**. It serves a Chrome extension via HTTP and WebSocket APIs.

**Key Components:**
- **server.py**: FastAPI application with HTTP and WebSocket endpoints
- **agents/root_agent.py**: Gemini AI integration and query processing with smart file selection
- **agents/file_selector_agent.py**: AI-powered intelligent file selection based on query relevance
- **utils/document_manager.py**: Document catalog and metadata management (GCS or local)
- **utils/file_upload_manager.py**: Gemini File API upload with caching (48 hours)
- **utils/file_summarizer.py**: AI-powered file summarization for intelligent selection
- **utils/chat_storage.py**: PostgreSQL/SQLite chat history and file summaries storage
- **utils/storage_manager.py**: Google Cloud Storage integration
- **utils/file_converter.py**: Office document to PDF conversion

## **CRITICAL: Deployment Workflow**

⚠️ **THIS BACKEND IS DEPLOYED ON RAILWAY** ⚠️

### Making Backend Changes:

1. **Make changes locally** in `canvas-extension-backend/`
2. **Test locally** (optional but recommended)
3. **Commit and push to GitHub**:
   ```bash
   cd canvas-extension-backend
   git add .
   git commit -m "Your change description"
   git push origin main
   ```
4. **Railway auto-deploys** from GitHub (connected to repo)
5. **Check deployment logs** at Railway dashboard
6. **Frontend is already configured** to use Railway URL

### Important Notes:

- **DO NOT** manually deploy to Railway - it auto-deploys from GitHub
- **Always update requirements.txt** if adding Python packages
- **Test with production URL**: `https://web-production-9aaba7.up.railway.app`
- **Environment variables** are configured in Railway dashboard
- **Database** is PostgreSQL hosted on Railway (with SQLite fallback)
- **Cloud Storage** is Google Cloud Storage (with local fallback)

### Environment Variables (Railway):

```bash
GOOGLE_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://...  # Auto-configured by Railway
GCS_BUCKET_NAME=canvas-extension-pdfs
GCS_PROJECT_ID=your_project_id
GCP_SERVICE_ACCOUNT_BASE64=base64_encoded_credentials
```

## File Structure

```
canvas-extension-backend/
├── server.py                          # FastAPI app, endpoints, WebSocket
├── agents/
│   ├── root_agent.py                 # Main Gemini integration + smart selection
│   └── file_selector_agent.py        # AI-powered file relevance selection
├── utils/
│   ├── document_manager.py           # Document catalog (GCS/local)
│   ├── file_upload_manager.py        # Gemini File API uploads
│   ├── file_summarizer.py            # AI file summarization
│   ├── chat_storage.py               # PostgreSQL/SQLite chat history
│   ├── storage_manager.py            # Google Cloud Storage
│   ├── file_converter.py             # Office to PDF conversion
│   └── mime_types.py                 # MIME type detection
├── uploads/                           # Local storage fallback (gitignored)
├── data/                              # SQLite database fallback (gitignored)
├── .backend-venv/                     # Python venv (gitignored)
├── .env                               # API keys (gitignored, not needed for Railway)
├── .env.example                       # Template
├── requirements.txt                   # Python dependencies
├── Procfile                           # Railway deployment config
└── railway.json                       # Railway build config
```

## Development Commands

```bash
# Initial setup (local development)
python3 -m venv .backend-venv
source .backend-venv/bin/activate  # Windows: .backend-venv\Scripts\activate
pip install -r requirements.txt

# Create .env file (for local development only)
cp .env.example .env
# Edit .env and add required API keys

# Start server locally
python server.py
# Runs on http://localhost:8000

# Test
curl http://localhost:8000/  # Health check
```

## Critical Patterns

### 1. Smart File Selection (NEW)

**How it works:**
1. When files are uploaded, they're automatically summarized by AI
2. Summaries are stored in database (PostgreSQL/SQLite)
3. When user asks a question with "Smart Selection" enabled:
   - File Selector Agent analyzes the query
   - Retrieves all file summaries for the course
   - Uses AI to select 3-5 most relevant files
   - Returns selected files with relevance scores

**Code flow:**
```python
# server.py: After file upload, generate summaries asynchronously
asyncio.create_task(_generate_summaries_background(course_id, successful_uploads))

# _generate_summaries_background():
# - Uploads file to Gemini File API
# - Calls file_summarizer.summarize_file()
# - Saves to chat_storage.save_file_summary()

# root_agent.py: When use_smart_selection=True
if use_smart_selection:
    summaries = chat_storage.get_all_summaries_for_course(course_id)
    selected = file_selector_agent.select_relevant_files(
        user_query, summaries, syllabus_summary, max_files=5
    )
    materials_to_use = [m for m in all_materials if m["id"] in selected_doc_ids]
```

**Database schema** (chat_storage.py):
```sql
CREATE TABLE file_summaries (
    doc_id VARCHAR(255) PRIMARY KEY,
    course_id VARCHAR(255) NOT NULL,
    filename TEXT NOT NULL,
    summary TEXT NOT NULL,              -- 150-200 word summary
    topics TEXT,                        -- JSON array of keywords
    metadata TEXT,                      -- JSON: doc_type, time_references, etc.
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 2. File Upload & Storage

**Endpoint**: `POST /upload_pdfs?course_id={id}`

**Process** (server.py):
1. Receive files via multipart form data (supports PDF, DOCX, PPTX, images, etc.)
2. Auto-detect MIME types
3. Upload to Google Cloud Storage (or local storage as fallback)
4. Add to document catalog
5. **NEW**: Generate summaries asynchronously in background
6. Return success status

**Storage locations:**
- **Production (Railway)**: Google Cloud Storage bucket
- **Local development**: `uploads/{course_id}_{filename}`

**File naming**: `course_id/filename.ext` (GCS) or `course_id_filename.ext` (local)

### 3. Session-Based File Caching

**Problem**: Uploading files to Gemini File API is slow

**Solution** (root_agent.py):
```python
# Class-level cache managed by FileUploadManager
self.file_upload_manager = FileUploadManager(
    self.client,
    cache_duration_hours=48,
    storage_manager=storage_manager
)

# Cache reuses uploaded file URIs for 48 hours
# If file already uploaded: returns cached URI (instant)
# If cache expired: re-uploads to Gemini
```

**Cache scope**: Per root_agent instance (shared across WebSocket messages)

### 4. WebSocket Streaming

**Endpoint**: `WS /ws/chat/{course_id}`

**Message format from frontend**:
```json
{
  "message": "user question",
  "history": [{"role": "user/assistant", "content": "..."}],
  "selected_docs": ["course_123_syllabus", "course_123_lecture1"],
  "syllabus_id": "course_123_syllabus",
  "session_id": "session_123_1234567890",
  "api_key": "user_gemini_key_or_null",
  "enable_web_search": false,
  "use_smart_selection": true  // NEW: Enable AI file selection
}
```

**Server response format**:
```json
{"type": "chunk", "content": "streaming text"}
{"type": "done"}
{"type": "error", "message": "error description"}
```

**Message handling** (server.py):
```python
@app.websocket("/ws/chat/{course_id}")
async def websocket_chat(websocket, course_id):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()

        # Extract parameters including NEW use_smart_selection
        use_smart_selection = data.get("use_smart_selection", False)

        # Process with streaming
        async for chunk in root_agent.process_query_stream(
            course_id=course_id,
            user_message=data["message"],
            conversation_history=data["history"],
            selected_docs=data["selected_docs"],
            use_smart_selection=use_smart_selection,  # NEW
            ...
        ):
            await websocket.send_json({"type": "chunk", "content": chunk})

        await websocket.send_json({"type": "done"})

        # Auto-save to chat history
        chat_storage.save_chat_session(...)
```

### 5. Chat History Persistence

**Database**: PostgreSQL (production) or SQLite (local fallback)

**Tables**:
- `chat_sessions`: Session metadata (course_id, title, message_count)
- `chat_messages`: Individual messages (session_id, role, content)
- `file_summaries`: AI-generated file summaries (NEW)

**Auto-save behavior**: After each query completes, conversation is saved to database

**API endpoints**:
- `GET /chats/{course_id}` - Get recent chats (limit=20)
- `GET /chats/{course_id}/{session_id}` - Load specific chat session
- `DELETE /chats/{course_id}/{session_id}` - Delete chat session

### 6. Cloud Storage Integration

**Storage Manager** (utils/storage_manager.py):
```python
# Production: Google Cloud Storage
storage_manager = StorageManager(
    bucket_name="canvas-extension-pdfs",
    project_id=GCS_PROJECT_ID,
    credentials_path=GOOGLE_APPLICATION_CREDENTIALS
)

# Upload file
blob_name = storage_manager.upload_pdf(course_id, filename, content, mime_type)
# Returns: "course_id/filename.ext"

# Download file
content = storage_manager.download_pdf(blob_name)

# List files
files = storage_manager.list_files()
```

**Fallback**: If GCS not configured, uses local `uploads/` directory

### 7. Document Catalog Management

**Document Manager** (utils/document_manager.py):

**Build catalog** (on startup):
```python
# Scans GCS bucket or local uploads/ directory
catalog = {
    "course_123": [
        {
            "id": "course_123_syllabus",
            "name": "syllabus",
            "filename": "syllabus.pdf",
            "path": "course_123/syllabus.pdf",  # GCS path
            "size_mb": 0.5,
            "type": "syllabus",
            "storage": "gcs"
        },
        ...
    ]
}
```

**Incremental updates**: `add_files_to_catalog()` for fast catalog updates

### 8. System Prompt with Citations

**Location**: `root_agent.py`

**Instructions include**:
```python
system_instruction = f"""
You are an AI study assistant with access to course materials.

When citing information, use EXACT format:
[Source: DocumentName, Page X]

Available documents:
{doc_list}
"""
```

**Citation parsing**: Frontend parses `[Source: X, Page Y]` into clickable links

## API Endpoints

### HTTP Endpoints

- `GET /` - Health check
- `POST /upload_pdfs?course_id={id}` - Upload files (multi-file support)
- `GET /collections/{course_id}/status` - Get uploaded files info
- `GET /chats/{course_id}` - Get recent chat sessions
- `GET /chats/{course_id}/{session_id}` - Load specific chat
- `DELETE /chats/{course_id}/{session_id}` - Delete chat
- `PATCH /chats/{course_id}/{session_id}/title` - Update chat title

### WebSocket Endpoints

- `WS /ws/chat/{course_id}` - Streaming AI chat

## Configuration

### Required Environment Variables

**For Railway (production)**:
```bash
GOOGLE_API_KEY=AIza...                           # Gemini API key
DATABASE_URL=postgresql://...                    # Auto-set by Railway
GCS_BUCKET_NAME=canvas-extension-pdfs
GCS_PROJECT_ID=your-project-id
GCP_SERVICE_ACCOUNT_BASE64=base64_encoded_json   # For Railway
```

**For local development**:
```bash
GOOGLE_API_KEY=AIza...                           # Required
# DATABASE_URL not set = uses SQLite at ./data/chats.db
# GCS not configured = uses local ./uploads/ directory
```

### CORS Settings

```python
# Allows Chrome extension to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Gemini Model

**Current**: `gemini-2.5-flash` (root_agent.py, file_summarizer.py, file_selector_agent.py)

**Why Flash?**
- Fast response times (<1s for cached queries)
- Cost-efficient for large contexts
- Native PDF/image reading
- 1M token context window
- Sufficient quality for study assistant tasks

## Performance Metrics

### File Upload & Summarization
- **Upload to GCS**: ~100-500ms per file
- **Summary generation**: ~1-2s per file (runs in background)
- **Parallel upload**: Batch size of 10 files at a time

### Query Processing
- **First query** (with smart selection):
  - Get summaries from DB: ~50-100ms
  - AI file selection: ~1-2s
  - Upload selected files to Gemini: ~3-5s
  - Total: ~5-8s
- **Follow-up query** (cached files): <1s
- **Manual selection** (no AI): 3-5s first query, <1s after

### Smart File Selection Benefits
- **Token reduction**: 60-80% (only 3-5 files vs all files)
- **Cost savings**: Proportional to token reduction
- **Response quality**: Often improved due to focused context

## Common Issues & Solutions

### "ModuleNotFoundError: No module named 'X'"
**Solution**: Add package to `requirements.txt`, commit and push to GitHub
```bash
# requirements.txt
new-package-name
```

### WebSocket disconnects immediately
- Check Railway logs for errors
- Verify frontend uses correct WebSocket URL: `wss://web-production-9aaba7.up.railway.app`
- Check CORS settings allow extension

### Files not uploading
- **Check Railway logs** for GCS errors
- Verify GCS credentials are set correctly in Railway environment
- Check bucket name and permissions

### Summaries not generating
- Check Railway logs for `_generate_summaries_background` errors
- Verify `GOOGLE_API_KEY` is set
- Ensure files uploaded successfully to Gemini File API
- Check database connection (PostgreSQL on Railway)

### Smart selection not working
- Verify `use_smart_selection: true` in WebSocket payload
- Check that file summaries exist in database: `GET /chats/{course_id}` (add debug endpoint if needed)
- Ensure `file_selector_agent` is initialized correctly
- Check Railway logs for AI selection errors

## Modifying System Behavior

### Add New Python Dependency

1. Add to `requirements.txt`:
```bash
new-package==1.0.0
```

2. Commit and push:
```bash
git add requirements.txt
git commit -m "Add new-package dependency"
git push origin main
```

3. Railway will auto-install on next deployment

### Change Smart Selection Behavior

**File Selector Agent** (agents/file_selector_agent.py):
```python
# Adjust selection prompt
# Modify max_files parameter (default: 5)
# Change relevance scoring criteria
```

**File Summarizer** (utils/file_summarizer.py):
```python
# Adjust summary length (default: 150-200 words)
# Modify topics extraction (default: 5-10 keywords)
# Change summarization prompt
```

### Add New WebSocket Message Type

1. Update message parsing in `server.py`:
```python
new_param = message_data.get("new_param", default_value)
```

2. Pass to root_agent:
```python
async for chunk in root_agent.process_query_stream(
    ...,
    new_param=new_param
):
```

3. Update root_agent signature and logic

4. **Update frontend** to send new parameter

### Add New Database Table

1. Update `chat_storage.py`:
```python
def _initialize_postgres(self):
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS new_table (
            id SERIAL PRIMARY KEY,
            ...
        )
    """))

def _initialize_sqlite(self):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS new_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ...
        )
    """)
```

2. Add methods for CRUD operations

3. Commit and push - Railway will auto-run migrations on startup

## Testing

### Local Testing

```bash
# Health check
curl http://localhost:8000/

# Upload file
curl -X POST "http://localhost:8000/upload_pdfs?course_id=test123" \
  -F "files=@sample.pdf"

# Check chat history (if database set up)
curl http://localhost:8000/chats/test123
```

### Production Testing

```bash
# Health check
curl https://web-production-9aaba7.up.railway.app/

# WebSocket (requires wscat or similar)
wscat -c wss://web-production-9aaba7.up.railway.app/ws/chat/test123
```

## Security Considerations

1. **API Keys**: Stored in Railway environment variables (encrypted)
2. **CORS**: Configured to allow Chrome extension origin
3. **File paths**: Sanitized to prevent directory traversal
4. **Database**: PostgreSQL with connection pooling
5. **Cloud Storage**: GCS with service account authentication
6. **Rate limiting**: Consider adding for production (not currently implemented)

## Troubleshooting Deployment

### Check Railway Logs

1. Go to Railway dashboard
2. Select your project
3. Click "Deployments" tab
4. View logs for latest deployment

### Common Deployment Errors

**"No module named 'packaging'"**
- Add `packaging` to requirements.txt
- Commit and push

**"Could not connect to database"**
- Check `DATABASE_URL` is set in Railway
- Verify PostgreSQL service is running

**"GCS authentication failed"**
- Check `GCP_SERVICE_ACCOUNT_BASE64` is set correctly
- Verify base64 encoding is valid
- Ensure service account has Storage Admin role

## Related Repositories

- **Frontend (Chrome Extension)**: `canvas-extension-frontend/` (in parent directory)
- **Backend (this repo)**: Deployed on Railway from GitHub

## Architecture Diagram

```
┌─────────────────┐
│  Chrome Ext     │
│  (Frontend)     │
└────────┬────────┘
         │ HTTP/WebSocket
         │
┌────────▼────────────────────────┐
│  FastAPI Server (server.py)     │
│  - Railway deployment           │
│  - WebSocket endpoint           │
│  - File upload endpoint         │
└────────┬────────────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│ Root │  │ File  │
│Agent │  │Selector│
└───┬──┘  │Agent  │
    │     └───┬───┘
    │         │
┌───▼─────────▼──────┐
│ Document Manager   │
│ File Summarizer    │
│ Chat Storage       │
│ Storage Manager    │
└────────┬───────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│ GCS  │  │ Gemini│
│Bucket│  │ API  │
└──────┘  └──────┘
```
