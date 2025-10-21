# Canvas Extension Backend

Python backend server for the Canvas LMS AI Study Assistant. Provides AI-powered document processing using Google Gemini 2.5 Flash with native PDF reading.

## Features

- FastAPI-based REST API and WebSocket server
- Google Gemini 2.5 Flash integration for AI responses
- Native PDF processing (no OCR or chunking needed)
- Session-based file caching for performance
- Streaming responses via WebSocket
- Citation system for source attribution

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/canvas-extension-backend.git
cd canvas-extension-backend
```

### 2. Create Virtual Environment

```bash
python3 -m venv .backend-venv
source .backend-venv/bin/activate  # On Windows: .backend-venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Google Gemini API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

## Running the Server

### Development

```bash
# Activate virtual environment
source .backend-venv/bin/activate

# Start the server
python server.py
```

The server will start on `http://localhost:8000`

### Production

For production deployment, use a production ASGI server:

```bash
# Install uvicorn with production extras
pip install uvicorn[standard]

# Run with uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### HTTP Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "Canvas Study Assistant Backend"
}
```

#### `POST /upload_pdfs`
Upload PDFs for a course.

**Parameters:**
- `course_id` (query): Course identifier
- `files` (form-data): PDF files to upload

**Response:**
```json
{
  "status": "success",
  "course_id": "12345",
  "files_received": 25
}
```

#### `GET /collections/{course_id}/status`
Get status of uploaded files for a course.

**Response:**
```json
{
  "success": true,
  "course_id": "12345",
  "files": ["12345_syllabus", "12345_lecture1", ...]
}
```

#### `GET /pdfs/{course_id}/{filename}`
Serve a PDF file.

**Example:**
```
GET /pdfs/12345/syllabus.pdf#page=5
```

### WebSocket Endpoint

#### `WS /ws/chat/{course_id}`
WebSocket connection for real-time AI chat.

**Client Message Format:**
```json
{
  "message": "Explain the key concepts",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "model", "content": "..."}
  ],
  "selected_docs": ["doc_id1", "doc_id2"],
  "syllabus_id": "doc_id"
}
```

**Server Response Format:**
```json
{"type": "chunk", "content": "text chunk"}
{"type": "done"}
{"type": "error", "message": "error description"}
```

## Project Structure

```
canvas-extension-backend/
├── server.py                      # FastAPI application
├── agents/
│   └── root_agent.py             # Gemini AI integration
├── utils/
│   ├── document_manager.py       # PDF catalog management
│   └── file_upload_manager.py    # Gemini File API uploads
├── uploads/                      # PDF storage (gitignored)
├── .backend-venv/                # Python virtual environment (gitignored)
├── .env                          # Environment variables (gitignored)
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional
HOST=0.0.0.0
PORT=8000
```

### CORS Settings

The server is configured to accept requests from all origins for development:

```python
# server.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For production**, update `allow_origins` to specific domains:
```python
allow_origins=["chrome-extension://your-extension-id"]
```

### File Upload Cache

Files uploaded to Gemini File API are cached for 48 hours by default.

To change cache duration, edit `agents/root_agent.py`:

```python
# Line ~160
cache_duration_hours = 48  # Change to desired hours
```

## Gemini Model Configuration

**Current Model:** `gemini-2.5-flash`

This model was chosen for:
- Fast response times
- Cost efficiency
- Native PDF reading capability
- 1M token context window

To change the model, edit `agents/root_agent.py`:

```python
MODEL_NAME = "gemini-2.5-flash"  # Change to another Gemini model
```

## Development

### Adding New Endpoints

```python
# server.py
@app.get("/your-endpoint")
async def your_endpoint():
    return {"message": "Hello"}
```

### Modifying System Prompt

Edit the system instruction in `agents/root_agent.py`:

```python
# Line ~134
system_instruction = """
Your custom prompt here...
"""
```

**Important:** Keep citation format instructions to maintain compatibility with the frontend.

### Testing

Test the health endpoint:
```bash
curl http://localhost:8000/
```

Test PDF serving:
```bash
curl http://localhost:8000/pdfs/12345/filename.pdf
```

## Deployment

### Local Network

To allow devices on your local network to access the backend:

```bash
# Run on all interfaces
python server.py --host 0.0.0.0
```

Then access from other devices using your computer's IP:
```
http://192.168.1.100:8000
```

### Cloud Deployment

#### Railway / Render / Heroku

1. Add a `Procfile`:
   ```
   web: uvicorn server:app --host 0.0.0.0 --port $PORT
   ```

2. Set environment variables:
   - `GOOGLE_API_KEY`

3. Deploy according to platform instructions

#### Docker

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. Build and run:
   ```bash
   docker build -t canvas-backend .
   docker run -p 8000:8000 --env-file .env canvas-backend
   ```

## Troubleshooting

### "Module not found" errors
```bash
# Ensure virtual environment is activated
source .backend-venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### "Google API key not found"
- Check that `.env` file exists
- Verify `GOOGLE_API_KEY` is set correctly
- Restart the server after editing `.env`

### WebSocket connection fails
- Check firewall settings
- Verify the port (8000) is not blocked
- Ensure CORS settings allow the extension origin

### PDFs not uploading
- Check `uploads/` directory permissions
- Verify sufficient disk space
- Review server logs for errors

## Performance Optimization

Current optimizations:
- **Parallel uploads**: Multiple PDFs uploaded concurrently
- **Session caching**: Files uploaded once per WebSocket session
- **No text extraction**: Gemini reads PDFs natively
- **Streaming**: Responses sent incrementally

**Metrics:**
- First upload (25 PDFs): 5-10 seconds
- Cached query: <1 second
- Follow-up messages: <1 second

## Security Considerations

1. **API Keys**: Store in `.env`, never commit to git
2. **CORS**: Restrict origins in production
3. **File Validation**: PDFs are validated before processing
4. **Path Traversal**: File paths are sanitized
5. **Rate Limiting**: Consider adding for production

## Dependencies

Core dependencies (see `requirements.txt`):
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `google-generativeai`: Gemini AI SDK
- `python-dotenv`: Environment variables
- `websockets`: WebSocket support

## Support

For issues or questions:
- Check the [Extension Setup Guide](https://github.com/your-username/canvas-extension)
- Review troubleshooting section above
- Open an issue on GitHub

## License

MIT License - see LICENSE file for details

## Credits

- Powered by Google Gemini 2.5 Flash
- Built with FastAPI
