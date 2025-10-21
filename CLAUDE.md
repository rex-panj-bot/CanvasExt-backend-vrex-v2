# CLAUDE.md - Canvas Extension Backend

This file provides guidance to Claude Code when working with the Python backend code.

## Project Overview

Python backend server for Canvas LMS AI Study Assistant. Provides AI-powered document processing using Google Gemini 2.5 Flash with native PDF reading.

## Architecture

This is the **backend component only**. It serves a Chrome extension via HTTP and WebSocket APIs.

**Key Components:**
- **server.py**: FastAPI application with HTTP and WebSocket endpoints
- **agents/root_agent.py**: Gemini AI integration and query processing
- **utils/document_manager.py**: PDF catalog and metadata management
- **utils/file_upload_manager.py**: Gemini File API upload with caching

## Development Commands

```bash
# Initial setup
python3 -m venv .backend-venv
source .backend-venv/bin/activate  # Windows: .backend-venv\Scripts\activate
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add: GOOGLE_API_KEY=your_api_key

# Start server
python server.py
# Runs on http://localhost:8000

# Test
curl http://localhost:8000/  # Health check
curl http://localhost:8000/pdfs/{course_id}/{filename}.pdf
```

## File Structure

```
canvas-extension-backend/
├── server.py                      # FastAPI app, endpoints
├── agents/
│   └── root_agent.py             # Gemini integration
├── utils/
│   ├── document_manager.py       # PDF catalog
│   └── file_upload_manager.py    # Gemini uploads
├── uploads/                      # PDF storage (gitignored)
├── .backend-venv/                # Python venv (gitignored)
├── .env                          # API keys (gitignored)
├── .env.example                  # Template
└── requirements.txt
```

## Critical Patterns

### 1. PDF Upload & Storage

**Endpoint**: `POST /upload_pdfs?course_id={id}`

**Process** (`server.py` lines ~120-150):
1. Receive PDFs via multipart form data
2. Save to `uploads/{course_id}_{filename}`
3. Track in document catalog
4. Return success status

**Storage format**: `uploads/12345_lecture1.pdf`

### 2. Session-Based File Caching

**Problem**: Uploading 25 PDFs to Gemini takes 5-10 seconds per query.

**Solution** (`root_agent.py`):
```python
# Class-level cache for uploaded file URIs
self.uploaded_files = {}  # {doc_id: file_uri}

# On first query with new docs:
if doc_id not in self.uploaded_files:
    file_uri = await upload_to_gemini(pdf_path)
    self.uploaded_files[doc_id] = file_uri

# Subsequent queries reuse file_uri
```

**Cache scope**: Per WebSocket connection (cleared on disconnect)

### 3. WebSocket Streaming

**Endpoint**: `WS /ws/chat/{course_id}`

**Message handling** (`server.py` lines ~200-250):
```python
@app.websocket("/ws/chat/{course_id}")
async def websocket_chat(websocket, course_id):
    await websocket.accept()
    agent = RootAgent()  # Session-level instance

    while True:
        data = await websocket.receive_json()
        async for chunk in agent.process_query_stream(...):
            await websocket.send_json({"type": "chunk", "content": chunk})
        await websocket.send_json({"type": "done"})
```

**Important**: Each WebSocket connection gets its own `RootAgent` instance for session caching.

### 4. Gemini File API Integration

**Location**: `utils/file_upload_manager.py`

**Upload process**:
```python
def upload_to_gemini(file_path, cache_duration_hours=48):
    # Upload PDF to Gemini File API
    file = genai.upload_file(file_path)

    # Wait for processing
    while file.state == "PROCESSING":
        time.sleep(1)
        file = genai.get_file(file.name)

    return file.uri  # Use in API calls
```

**Caching**: Gemini caches files for 48 hours (configurable)

### 5. System Prompt with Citations

**Location**: `root_agent.py` lines ~134-148

**Key instructions**:
```python
system_instruction = f"""
You are an AI study assistant with access to:
{doc_list}

When referencing information, use this EXACT format:
[Source: DocumentName, Page X]

Examples:
- [Source: Syllabus, Page 2]
- [Source: Lecture_1, Page 5]
"""
```

**Important**: Citation format must match frontend regex in `chat.js`

### 6. Document Selection & Filtering

**Frontend sends**: `selected_docs` array of doc IDs

**Backend filters** (`root_agent.py` lines ~160-180):
```python
def get_selected_docs(course_id, selected_doc_ids, syllabus_id):
    all_docs = document_manager.get_documents(course_id)

    # Filter to selected
    docs = [d for d in all_docs if d['id'] in selected_doc_ids]

    # Always include syllabus if provided
    if syllabus_id and syllabus_id not in selected_doc_ids:
        syllabus = document_manager.get_document(syllabus_id)
        if syllabus:
            docs.insert(0, syllabus)

    return docs
```

## API Endpoints

### HTTP

- `GET /` - Health check
- `POST /upload_pdfs?course_id={id}` - Upload PDFs
- `GET /collections/{course_id}/status` - Get uploaded files
- `GET /pdfs/{course_id}/{filename}` - Serve PDF

### WebSocket

- `WS /ws/chat/{course_id}` - Streaming AI chat

**Client message format**:
```json
{
  "message": "user question",
  "history": [{"role": "user/model", "content": "..."}],
  "selected_docs": ["12345_syllabus", "12345_lecture1"],
  "syllabus_id": "12345_syllabus"
}
```

**Server response format**:
```json
{"type": "chunk", "content": "text"}
{"type": "done"}
{"type": "error", "message": "..."}
```

## Configuration

### Environment Variables

**Required**:
- `GOOGLE_API_KEY`: Gemini API key

**Optional**:
- `HOST`: Server host (default: localhost)
- `PORT`: Server port (default: 8000)

### CORS Settings

**Development** (`server.py`):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Production**: Change `allow_origins` to specific Chrome extension ID:
```python
allow_origins=["chrome-extension://your-extension-id"]
```

### Gemini Model

**Current model**: `gemini-2.5-flash` (`root_agent.py` line 26)

**Why Flash?**
- Fast response times (<1s for cached queries)
- Cost-efficient
- Native PDF reading (no OCR needed)
- 1M token context window

**To change**:
```python
MODEL_NAME = "gemini-2.5-pro"  # or another model
```

## Performance Optimization

### Current Optimizations

1. **Parallel uploads**: `asyncio.gather()` for simultaneous PDF uploads
2. **Session caching**: Files uploaded once per WebSocket connection
3. **No preprocessing**: PDFs sent directly to Gemini (no text extraction)
4. **Streaming**: Chunks sent immediately, not buffered
5. **Gemini file caching**: 48-hour server-side cache

### Metrics

- **First query** (uploads + processing): 5-10 seconds
- **Follow-up query** (cached files): <1 second
- **Parallel upload** (25 PDFs): ~5 seconds

## Common Issues

### "Google API key not found"
- Check `.env` file exists in root directory
- Verify `GOOGLE_API_KEY=...` is set
- Restart server after editing `.env`

### WebSocket disconnects immediately
- Check CORS settings allow extension origin
- Verify no firewall blocking WebSocket connections
- Look for errors in server console

### PDFs not uploading
- Check `uploads/` directory exists and is writable
- Verify file size limits (default: no limit, but consider setting one)
- Look for errors in server logs

### Slow responses
- First query per session uploads files (5-10s expected)
- Follow-up queries should be <1s
- Check Gemini API rate limits if consistently slow

## Modifying System Behavior

### Change System Prompt

**Location**: `root_agent.py` ~line 134

```python
system_instruction = f"""
Your custom instructions here...

Available documents:
{doc_list}

Citation format: [Source: DocumentName, Page X]
"""
```

**Warning**: Keep citation format or frontend links will break.

### Adjust File Cache Duration

**Location**: `root_agent.py` ~line 160

```python
cache_duration_hours = 48  # Change to desired hours
```

**Note**: Gemini File API may have its own limits.

### Add New Endpoint

**Location**: `server.py`

```python
@app.get("/your-endpoint")
async def your_endpoint():
    """Docstring"""
    try:
        # Your logic
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Deployment

### Local Network Access

```bash
# Allow connections from local network
python server.py --host 0.0.0.0
```

Access from other devices: `http://192.168.1.100:8000`

### Production (Cloud)

**Railway / Render / Heroku**:
1. Add `Procfile`: `web: uvicorn server:app --host 0.0.0.0 --port $PORT`
2. Set `GOOGLE_API_KEY` environment variable
3. Deploy per platform instructions

**Docker**:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t canvas-backend .
docker run -p 8000:8000 --env-file .env canvas-backend
```

## Security Considerations

1. **API Keys**: Never commit `.env` to git (already in `.gitignore`)
2. **CORS**: Restrict to extension ID in production
3. **File paths**: Validated to prevent directory traversal
4. **Rate limiting**: Consider adding for production
5. **File size limits**: Consider adding max file size check

## Dependencies

**Core** (see `requirements.txt`):
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `google-generativeai`: Gemini SDK
- `python-dotenv`: Environment variables
- `websockets`: WebSocket support

## Testing

```bash
# Health check
curl http://localhost:8000/

# Upload PDFs (requires files)
curl -X POST "http://localhost:8000/upload_pdfs?course_id=12345" \
  -F "files=@test.pdf"

# Check collection status
curl http://localhost:8000/collections/12345/status

# Serve PDF
curl http://localhost:8000/pdfs/12345/test.pdf
```

## Related Repository

Chrome extension: https://github.com/your-username/canvas-extension
