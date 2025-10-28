"""
FastAPI Server for AI Study Assistant
Handles WebSocket connections, PDF uploads, and agent coordination
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from typing import Dict, List, Optional
from urllib.parse import unquote

# Load environment variables
load_dotenv()

# Import our modules
from utils.document_manager import DocumentManager
from agents.root_agent import RootAgent
from utils.chat_storage import ChatStorage
from utils.storage_manager import StorageManager
from utils.mime_types import get_mime_type, get_file_extension
from utils.file_summarizer import FileSummarizer
from google import genai

app = FastAPI(title="AI Study Assistant Backend")

# CORS middleware for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
root_agent = None
document_manager = None
chat_storage = None
storage_manager = None
file_summarizer = None

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# PDF storage directory (deprecated - using GCS now, but kept for backward compatibility)
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global root_agent, document_manager, chat_storage, storage_manager, file_summarizer

    print("🚀 Starting AI Study Assistant Backend...")

    # Decode base64 GCP credentials if provided (for Railway deployment)
    if os.getenv("GCP_SERVICE_ACCOUNT_BASE64"):
        try:
            import base64
            import tempfile
            creds_json = base64.b64decode(os.getenv("GCP_SERVICE_ACCOUNT_BASE64"))
            temp_creds_path = "/tmp/gcp-service-account.json"
            with open(temp_creds_path, "wb") as f:
                f.write(creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path
            print("✅ Decoded GCP credentials from base64")
        except Exception as e:
            print(f"⚠️  Failed to decode GCP credentials: {e}")

    # Initialize Storage Manager (GCS)
    try:
        storage_manager = StorageManager(
            bucket_name=os.getenv("GCS_BUCKET_NAME", "canvas-extension-pdfs"),
            project_id=os.getenv("GCS_PROJECT_ID", ""),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        print(f"✅ Storage Manager (GCS) initialized")
    except Exception as e:
        print(f"⚠️  Warning: GCS not configured ({e}), falling back to local storage")
        storage_manager = None

    # Initialize Document Manager
    if storage_manager:
        document_manager = DocumentManager(storage_manager=storage_manager)
    else:
        document_manager = DocumentManager(upload_dir="./uploads")
    print(f"✅ Document Manager initialized")

    # Initialize Chat Storage (PostgreSQL or SQLite fallback)
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("postgresql"):
        chat_storage = ChatStorage(database_url=database_url)
        print(f"✅ Chat Storage (PostgreSQL) initialized")
    else:
        chat_storage = ChatStorage(db_path="./data/chats.db")
        print(f"✅ Chat Storage (SQLite) initialized")

    # Initialize Root Agent with Gemini 2.5 Flash
    root_agent = RootAgent(
        document_manager=document_manager,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        storage_manager=storage_manager,
        chat_storage=chat_storage
    )
    print(f"✅ Root Agent initialized")

    # Initialize File Summarizer
    file_summarizer = FileSummarizer(google_api_key=os.getenv("GOOGLE_API_KEY"))
    print(f"✅ File Summarizer initialized")

    print("🎉 Backend ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down backend...")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "AI Study Assistant Backend is running"
    }


async def _process_single_upload(course_id: str, file: UploadFile) -> Dict:
    """Process a single file upload (for parallel execution)"""
    try:
        # Auto-detect MIME type
        mime_type = get_mime_type(file.filename)
        ext = get_file_extension(file.filename) or 'file'

        if not mime_type:
            print(f"⚠️  Skipping {file.filename} - unsupported file type")
            return {
                "filename": file.filename,
                "status": "skipped",
                "error": "Unsupported file type"
            }

        print(f"📥 Processing: {file.filename} ({ext.upper()}, {mime_type})")

        content = await file.read()

        # Upload to GCS if available, otherwise save locally
        if storage_manager:
            # Run synchronous GCS upload in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            blob_name = await loop.run_in_executor(
                None,
                storage_manager.upload_pdf,
                course_id,
                file.filename,
                content,
                mime_type  # Pass MIME type
            )
            return {
                "filename": file.filename,
                "status": "uploaded",
                "size_bytes": len(content),
                "path": blob_name,  # GCS blob path
                "storage": "gcs",
                "mime_type": mime_type
            }
        else:
            # Fallback to local storage
            file_path = UPLOAD_DIR / f"{course_id}_{file.filename}"

            # Run file write in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: open(file_path, "wb").write(content)
            )

            return {
                "filename": file.filename,
                "status": "uploaded",
                "size_bytes": len(content),
                "path": str(file_path),
                "storage": "local",
                "mime_type": mime_type
            }
    except Exception as e:
        print(f"❌ Upload error for {file.filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "filename": file.filename,
            "status": "failed",
            "error": str(e)
        }


async def _generate_single_summary(
    file_info: Dict,
    course_id: str,
    file_uploader,
    file_summarizer,
    chat_storage
) -> Dict:
    """Generate summary for a single file"""
    try:
        filename = file_info["filename"]
        file_path = file_info["path"]

        # Create doc_id to match document_manager format
        original_name = filename
        if '.' in filename:
            original_name = '.'.join(filename.split('.')[:-1])
        doc_id = f"{course_id}_{original_name}"

        # Check if summary already exists (cached in database)
        existing = chat_storage.get_file_summary(doc_id)
        if existing:
            print(f"✅ Using cached summary for {filename}")
            return {"status": "cached", "filename": filename}

        # Upload file to Gemini File API (FileUploadManager handles caching)
        upload_result = file_uploader.upload_pdf(
            file_path=file_path,
            display_name=filename,
            mime_type=file_info.get("mime_type")
        )

        if "error" in upload_result:
            return {"status": "error", "filename": filename, "error": upload_result["error"]}

        file_uri = upload_result["file"].uri
        mime_type = upload_result["mime_type"]

        # Generate summary (simple LLM call)
        summary, topics, metadata = await file_summarizer.summarize_file(
            file_uri=file_uri,
            filename=filename,
            mime_type=mime_type
        )

        # Save to database (cache for future uploads)
        success = chat_storage.save_file_summary(
            doc_id=doc_id,
            course_id=course_id,
            filename=filename,
            summary=summary,
            topics=topics,
            metadata=metadata
        )

        if success:
            return {"status": "success", "filename": filename}
        else:
            return {"status": "error", "filename": filename, "error": "Failed to save"}

    except Exception as e:
        return {"status": "error", "filename": file_info.get("filename", "unknown"), "error": str(e)}


async def _generate_summaries_background(course_id: str, successful_uploads: List[Dict]):
    """
    Background task to generate summaries for uploaded files in parallel
    """
    try:
        from utils.file_upload_manager import FileUploadManager

        # Use default API key for summary generation
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  No GOOGLE_API_KEY found, skipping summary generation")
            return

        # Create file upload manager for uploading to Gemini
        file_upload_client = genai.Client(api_key=api_key)
        file_uploader = FileUploadManager(
            file_upload_client,
            cache_duration_hours=48,
            storage_manager=storage_manager
        )

        print(f"📝 Generating summaries for {len(successful_uploads)} files in parallel...")

        # Generate all summaries in parallel
        tasks = [
            _generate_single_summary(file_info, course_id, file_uploader, file_summarizer, chat_storage)
            for file_info in successful_uploads
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count results
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        cached_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "cached")
        error_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "error")
        exception_count = sum(1 for r in results if isinstance(r, Exception))

        print(f"✅ Summary generation complete: {success_count} new, {cached_count} cached, {error_count + exception_count} errors")

        # Log errors
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Exception: {result}")
            elif isinstance(result, dict) and result.get("status") == "error":
                print(f"❌ Error for {result['filename']}: {result.get('error')}")

    except Exception as e:
        print(f"❌ Critical error in summary generation: {e}")
        import traceback
        traceback.print_exc()


@app.post("/upload_pdfs")
async def upload_pdfs(
    course_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload files for a course (supports PDFs, documents, images, etc.)

    Args:
        course_id: Course identifier (used as filename prefix)
        files: List of files to upload (PDF, DOCX, TXT, images, etc.)

    Process:
        1. Saves all files to storage in parallel (GCS or local)
        2. Auto-detects file types and applies correct MIME types
        3. Incrementally updates document catalog
        4. No text extraction or processing (done on-demand during queries)

    Returns:
        JSON with upload results, success/failure counts

    Performance: ~1-2 seconds for 25 files
    """
    try:
        print(f"\n{'='*80}")
        print(f"📤 UPLOAD REQUEST:")
        print(f"   Course ID: {course_id}")
        print(f"   Number of files: {len(files)}")
        print(f"   File names received:")
        for f in files[:10]:  # Show first 10
            print(f"      - \"{f.filename}\"")
        print(f"   Storage manager available: {storage_manager is not None}")
        print(f"{'='*80}")

        # Process files in parallel (up to 100 files at once)
        # Modern async runtime can handle this efficiently
        BATCH_SIZE = 100
        print(f"📤 Uploading {len(files)} files {'in parallel' if len(files) <= BATCH_SIZE else f'in batches of {BATCH_SIZE}'}...")

        processed_results = []
        for i in range(0, len(files), BATCH_SIZE):
            batch = files[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} files)...")

            upload_tasks = [_process_single_upload(course_id, file) for file in batch]
            batch_results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"❌ Exception for file {batch[j].filename}: {result}")
                    import traceback
                    traceback.print_exception(type(result), result, result.__traceback__)
                    processed_results.append({
                        "filename": batch[j].filename,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

        # Count successes and failures
        successful = [r for r in processed_results if r["status"] == "uploaded"]
        failed = [r for r in processed_results if r["status"] == "failed"]
        skipped = [r for r in processed_results if r["status"] == "skipped"]

        print(f"✅ Upload results: {len(successful)} succeeded, {len(failed)} failed, {len(skipped)} skipped")
        if failed:
            print(f"❌ Failed files: {[f['filename'] for f in failed]}")
            for f in failed:
                print(f"   - {f['filename']}: {f.get('error', 'Unknown error')}")
        if skipped:
            print(f"⏭️  Skipped files: {[s['filename'] for s in skipped]}")
            for s in skipped:
                print(f"   - {s['filename']}: {s.get('error', 'Unknown reason')}")

        # Incrementally add new files to catalog (much faster than full rescan)
        if document_manager and successful:
            print(f"📚 Adding {len(successful)} files to catalog...")
            new_paths = [r["path"] for r in successful]
            document_manager.add_files_to_catalog(new_paths)
            print(f"✅ Catalog updated")

        # Generate summaries for uploaded files (asynchronously)
        if file_summarizer and chat_storage and successful:
            print(f"📝 Generating summaries for {len(successful)} files...")
            asyncio.create_task(_generate_summaries_background(course_id, successful))

        return {
            "success": True,
            "message": f"Uploaded {len(successful)}/{len(files)} PDFs successfully",
            "files": processed_results,
            "uploaded_count": len(successful),
            "failed_count": len(failed)
        }

    except Exception as e:
        print(f"❌ CRITICAL ERROR in upload_pdfs endpoint:")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ========== CHAT HISTORY ENDPOINTS ==========

@app.get("/chats/{course_id}")
async def get_recent_chats(course_id: str, limit: int = 20):
    """
    Get recent chat sessions for a course

    Args:
        course_id: Course identifier
        limit: Maximum number of sessions to return (default 20)

    Returns:
        JSON with:
        - success: bool
        - chats: List[Dict] with session info (id, title, timestamps, message count)
    """
    try:
        chats = chat_storage.get_recent_chats(course_id, limit)
        return {
            "success": True,
            "chats": chats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/chats/{course_id}/{session_id}")
async def get_chat_session(course_id: str, session_id: str):
    """
    Load a specific chat session

    Args:
        course_id: Course identifier (for validation)
        session_id: Session identifier

    Returns:
        JSON with:
        - success: bool
        - session: Dict with session info and full message history
    """
    try:
        session = chat_storage.get_chat_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        # Validate course_id matches
        if session.get('course_id') != course_id:
            raise HTTPException(status_code=403, detail="Course ID mismatch")

        return {
            "success": True,
            "session": session
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/chats/{course_id}")
async def save_chat_session(course_id: str, session_data: dict):
    """
    Save a chat session

    Args:
        course_id: Course identifier
        session_data: Dict containing:
            - session_id: str
            - messages: List[Dict] with role and content
            - title: str (optional)

    Returns:
        JSON with success status
    """
    try:
        session_id = session_data.get('session_id')
        messages = session_data.get('messages', [])
        title = session_data.get('title')

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        success = chat_storage.save_chat_session(
            session_id=session_id,
            course_id=course_id,
            messages=messages,
            title=title
        )

        return {
            "success": success,
            "message": "Chat saved successfully" if success else "Failed to save chat"
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.delete("/chats/{course_id}/{session_id}")
async def delete_chat_session(course_id: str, session_id: str):
    """
    Delete a chat session

    Args:
        course_id: Course identifier (for validation)
        session_id: Session identifier

    Returns:
        JSON with success status
    """
    try:
        # Verify session exists and belongs to this course
        session = chat_storage.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        if session.get('course_id') != course_id:
            raise HTTPException(status_code=403, detail="Course ID mismatch")

        success = chat_storage.delete_chat_session(session_id)

        return {
            "success": success,
            "message": "Chat deleted successfully" if success else "Failed to delete chat"
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.patch("/chats/{course_id}/{session_id}/title")
async def update_chat_title(course_id: str, session_id: str, data: dict):
    """
    Update the title of a chat session

    Args:
        course_id: Course identifier (for validation)
        session_id: Session identifier
        data: Dict containing:
            - title: str (new title)

    Returns:
        JSON with success status
    """
    try:
        # Verify session exists and belongs to this course
        session = chat_storage.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        if session.get('course_id') != course_id:
            raise HTTPException(status_code=403, detail="Course ID mismatch")

        title = data.get('title')
        if not title:
            raise HTTPException(status_code=400, detail="title is required")

        success = chat_storage.update_chat_title(session_id, title)

        return {
            "success": success,
            "message": "Title updated successfully" if success else "Failed to update title"
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.websocket("/ws/chat/{course_id}")
async def websocket_chat(websocket: WebSocket, course_id: str):
    """
    WebSocket endpoint for real-time AI chat

    Args:
        websocket: WebSocket connection
        course_id: Course identifier

    Process:
        1. Accepts WebSocket connection
        2. Receives query messages from client
        3. Processes with RootAgent (Gemini 2.5 Flash)
        4. Streams response chunks back to client
        5. Maintains session context (file uploads cached)

    Message Format (Client → Server):
        {
            "message": "user question",
            "history": [{"role": "user/model", "content": "..."}],
            "selected_docs": ["doc_id1", ...],  // Optional
            "syllabus_id": "doc_id",  // Optional
            "session_id": "unique_session_id"  // Optional, for chat history saving
        }

    Response Format (Server → Client):
        {"type": "chunk", "content": "text chunk"}
        {"type": "done"}
        {"type": "error", "message": "error description"}
    """
    await websocket.accept()
    connection_id = f"{course_id}_{id(websocket)}"
    active_connections[connection_id] = websocket

    print(f"🔌 WebSocket connected: {connection_id}")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            user_message = message_data.get("message", "")
            conversation_history = message_data.get("history", [])
            selected_docs = message_data.get("selected_docs", [])
            syllabus_id = message_data.get("syllabus_id")
            chat_session_id = message_data.get("session_id")  # For saving chat history
            enable_web_search = message_data.get("enable_web_search", False)  # Web search toggle
            user_api_key = message_data.get("api_key")  # User's Gemini API key
            use_smart_selection = message_data.get("use_smart_selection", False)  # Smart file selection toggle

            print(f"\n{'='*80}")
            print(f"📥 WebSocket received message:")
            print(f"   User message: {user_message[:100]}...")
            print(f"   History length: {len(conversation_history)}")
            print(f"   Selected docs count: {len(selected_docs)}")
            print(f"   Selected docs: {selected_docs[:3] if len(selected_docs) > 3 else selected_docs}...")
            print(f"   Syllabus ID: {syllabus_id}")
            print(f"   Session ID: {chat_session_id}")
            print(f"   Web Search: {enable_web_search}")
            print(f"   Smart Selection: {use_smart_selection}")
            print(f"   User API Key: {'Provided' if user_api_key else 'Not provided (using default)'}")
            print(f"{'='*80}")

            # Process with Root Agent and stream response
            chunk_count = 0
            assistant_response = ""
            async for chunk in root_agent.process_query_stream(
                course_id=course_id,
                user_message=user_message,
                conversation_history=conversation_history,
                selected_docs=selected_docs,
                syllabus_id=syllabus_id,
                session_id=connection_id,
                enable_web_search=enable_web_search,
                user_api_key=user_api_key,
                use_smart_selection=use_smart_selection
            ):
                chunk_count += 1
                assistant_response += chunk
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })

            # Send completion signal
            print(f"✅ Completed ({chunk_count} chunks)")
            await websocket.send_json({
                "type": "done"
            })

            # Auto-save chat history if session_id provided
            if chat_session_id and user_message and assistant_response:
                try:
                    # Build updated conversation history
                    updated_history = conversation_history + [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_response}
                    ]

                    # Save to database
                    chat_storage.save_chat_session(
                        session_id=chat_session_id,
                        course_id=course_id,
                        messages=updated_history
                    )
                    print(f"💾 Chat session saved: {chat_session_id}")
                except Exception as e:
                    print(f"⚠️  Failed to save chat session: {e}")

    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {connection_id}")
        # Clear session cache
        root_agent.clear_session(connection_id)
        del active_connections[connection_id]
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        await websocket.close()


@app.get("/collections/{course_id}/status")
async def get_collection_status(course_id: str):
    """
    Get status of a course collection (for upload caching)

    Args:
        course_id: Course identifier

    Returns:
        JSON with:
        - success: bool
        - course_id: str
        - documents: int (total count)
        - files: List[str] (file IDs already uploaded)

    Used by frontend to check which files are already uploaded,
    enabling smart caching and skipping re-uploads.
    """
    try:
        catalog = document_manager.get_material_catalog(course_id)

        response = {
            "success": True,
            "course_id": course_id,
            "documents": catalog.get("total_documents", 0),
            "files": [mat["id"] for mat in catalog.get("materials", [])]
        }

        return response

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }




@app.get("/courses/{course_id}/syllabus")
async def detect_syllabus(course_id: str):
    """Auto-detect syllabus document for a course"""
    try:
        syllabus_id = document_manager.find_syllabus(course_id)

        if syllabus_id:
            syllabus_doc = document_manager.get_document_summary(course_id, syllabus_id)
            return {
                "success": True,
                "syllabus_id": syllabus_id,
                "syllabus_name": syllabus_doc.get("name") if syllabus_doc else None
            }
        else:
            return {
                "success": True,
                "syllabus_id": None,
                "message": "No syllabus found"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/pdfs/{course_id}/{filename}")
async def serve_pdf(course_id: str, filename: str):
    """
    Serve PDF files for viewing in browser (for inline citations)

    Args:
        course_id: Course identifier
        filename: PDF filename (with or without .pdf extension)

    Returns:
        PDF file with content-disposition for browser viewing

    Example:
        GET /pdfs/12345/Lecture_Notes.pdf#page=5
        Opens Lecture_Notes.pdf at page 5
    """
    try:
        # Sanitize and decode filename
        filename = unquote(filename)

        # Ensure .pdf extension
        if not filename.endswith('.pdf'):
            filename += '.pdf'

        # Construct full path: uploads/{course_id}_{filename}
        file_path = UPLOAD_DIR / f"{course_id}_{filename}"

        # Security check: ensure file exists and is within uploads directory
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")

        if not file_path.is_relative_to(UPLOAD_DIR):
            raise HTTPException(status_code=403, detail="Access denied")

        # Return PDF with proper headers for browser viewing
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=filename,
            headers={
                "Content-Disposition": f"inline; filename={filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving PDF: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )
