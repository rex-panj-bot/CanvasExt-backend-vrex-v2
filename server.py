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

    # PHASE 3: Cleanup expired Gemini URIs on startup
    if chat_storage:
        try:
            deleted = chat_storage.cleanup_expired_gemini_uris()
            if deleted > 0:
                print(f"🧹 Cleaned up {deleted} expired Gemini URI(s)")
        except Exception as e:
            print(f"⚠️  Failed to cleanup expired URIs: {e}")

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
        from utils.file_converter import convert_office_to_pdf, needs_conversion

        # Auto-detect MIME type
        mime_type = get_mime_type(file.filename)
        ext = get_file_extension(file.filename) or 'file'

        # NEVER SKIP: If MIME type unknown, try to infer or use generic
        if not mime_type:
            # Try to infer from extension
            ext_to_mime = {
                'txt': 'text/plain',
                'md': 'text/markdown',
                'csv': 'text/csv',
                'json': 'application/json',
                'xml': 'application/xml',
                'html': 'text/html',
                'htm': 'text/html',
                'pdf': 'application/pdf',
            }
            mime_type = ext_to_mime.get(ext.lower(), 'application/octet-stream')
            print(f"⚠️  Unknown file type for {file.filename}, using inferred MIME: {mime_type}")

        print(f"📥 Processing: {file.filename} ({ext.upper()}, {mime_type})")

        content = await file.read()
        original_filename = file.filename
        actual_filename = file.filename
        conversion_info = None

        # PHASE 1: Convert files to AI-readable formats before uploading to GCS
        # This ensures all files stored in GCS are readable by Gemini
        if needs_conversion(file.filename):
            from utils.file_converter import convert_to_text

            # Determine conversion type based on file extension
            web_formats = ['html', 'htm', 'xml', 'json']
            is_web_format = ext.lower() in web_formats

            if is_web_format:
                # Convert web/data formats to plain text
                print(f"🔄 Converting {file.filename} ({ext.upper()}) to TXT before upload...")
                try:
                    loop = asyncio.get_event_loop()
                    text_bytes = await loop.run_in_executor(
                        None,
                        convert_to_text,
                        content,
                        file.filename
                    )

                    if text_bytes:
                        content = text_bytes
                        actual_filename = file.filename.rsplit('.', 1)[0] + '.txt'
                        mime_type = 'text/plain'
                        conversion_info = f"Converted {original_filename} → {actual_filename} ({len(text_bytes):,} bytes)"
                        print(f"✅ {conversion_info}")
                    else:
                        # Conversion failed - mark as unreadable
                        print(f"❌ Failed to convert {file.filename} - file is unreadable by AI")
                        return {
                            "filename": file.filename,
                            "status": "failed",
                            "error": "Could not convert file to AI-readable format",
                            "unreadable": True
                        }
                except Exception as conv_error:
                    print(f"❌ Conversion error for {file.filename}: {conv_error}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "filename": file.filename,
                        "status": "failed",
                        "error": f"Conversion error: {str(conv_error)}",
                        "unreadable": True
                    }
            else:
                # Convert Office/OpenDocument formats to PDF
                print(f"🔄 Converting {file.filename} ({ext.upper()}) to PDF before upload...")
                try:
                    loop = asyncio.get_event_loop()
                    pdf_bytes = await loop.run_in_executor(
                        None,
                        convert_office_to_pdf,
                        content,
                        file.filename
                    )

                    if pdf_bytes:
                        content = pdf_bytes
                        actual_filename = file.filename.rsplit('.', 1)[0] + '.pdf'
                        mime_type = 'application/pdf'
                        conversion_info = f"Converted {original_filename} → {actual_filename} ({len(pdf_bytes):,} bytes)"
                        print(f"✅ {conversion_info}")
                    else:
                        # Conversion failed - mark as unreadable
                        print(f"❌ Failed to convert {file.filename} - file is unreadable by AI")
                        return {
                            "filename": file.filename,
                            "status": "failed",
                            "error": "Could not convert file to PDF format",
                            "unreadable": True
                        }
                except Exception as conv_error:
                    print(f"❌ Conversion error for {file.filename}: {conv_error}")
                    import traceback
                    traceback.print_exc()
                    return {
                        "filename": file.filename,
                        "status": "failed",
                        "error": f"Conversion error: {str(conv_error)}",
                        "unreadable": True
                    }

        # Upload to GCS if available, otherwise save locally
        if storage_manager:
            # Run synchronous GCS upload in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            blob_name = await loop.run_in_executor(
                None,
                storage_manager.upload_pdf,
                course_id,
                actual_filename,  # Use converted filename
                content,
                mime_type  # Pass MIME type
            )
            result = {
                "filename": original_filename,  # Original name for frontend
                "actual_filename": actual_filename,  # What's stored in GCS
                "status": "uploaded",
                "size_bytes": len(content),
                "path": blob_name,  # GCS blob path
                "storage": "gcs",
                "mime_type": mime_type
            }
            if conversion_info:
                result["conversion"] = conversion_info
            return result
        else:
            # Fallback to local storage
            file_path = UPLOAD_DIR / f"{course_id}_{actual_filename}"  # Use converted filename

            # Run file write in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: open(file_path, "wb").write(content)
            )

            result = {
                "filename": original_filename,  # Original name for frontend
                "actual_filename": actual_filename,  # What's stored locally
                "status": "uploaded",
                "size_bytes": len(content),
                "path": str(file_path),
                "storage": "local",
                "mime_type": mime_type
            }
            if conversion_info:
                result["conversion"] = conversion_info
            return result
    except Exception as e:
        print(f"❌ Upload error for {file.filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "filename": file.filename,
            "status": "failed",
            "error": str(e)
        }


def _sync_summarize_file(summarizer, file_uri, filename, mime_type):
    """Synchronous wrapper for file summarization - runs in thread pool"""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            summarizer.summarize_file(file_uri, filename, mime_type)
        )
    finally:
        loop.close()


async def _generate_single_summary(
    file_info: Dict,
    course_id: str,
    file_uploader,
    file_summarizer,
    chat_storage
) -> Dict:
    """Generate summary for a single file (optimized with thread pools)"""
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

        # NEVER SKIP: Process all files, even small ones
        # Small files might contain important information (syllabus, instructions, etc.)
        file_size = file_info.get("size_bytes", 0)

        # Upload file to Gemini File API (run in thread pool - BLOCKING operation)
        upload_result = await asyncio.to_thread(
            file_uploader.upload_pdf,
            file_path,
            filename,
            file_info.get("mime_type")
        )

        if "error" in upload_result:
            return {"status": "error", "filename": filename, "error": upload_result["error"]}

        file_uri = upload_result["file"].uri
        mime_type = upload_result["mime_type"]

        # Generate summary (run in thread pool - BLOCKING LLM call)
        summary, topics, metadata = await asyncio.to_thread(
            _sync_summarize_file,
            file_summarizer,
            file_uri,
            filename,
            mime_type
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


async def _upload_to_gemini_background(course_id: str, successful_uploads: List[Dict]):
    """
    PHASE 3: Background task to pre-warm Gemini File API cache

    Uploads files to Gemini immediately after GCS upload to eliminate
    query-time upload wait (10-30 seconds saved on first query)
    """
    try:
        from utils.file_upload_manager import FileUploadManager

        # Use default API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️  No GOOGLE_API_KEY found, skipping Gemini pre-warm")
            return

        # Create file upload manager with database caching
        file_upload_client = genai.Client(api_key=api_key)
        file_uploader = FileUploadManager(
            file_upload_client,
            cache_duration_hours=48,
            storage_manager=storage_manager,
            chat_storage=chat_storage
        )

        # Limit concurrent uploads (max 10 at a time to avoid overwhelming Gemini API)
        semaphore = asyncio.Semaphore(10)

        async def upload_single_file(file_info):
            async with semaphore:
                try:
                    filename = file_info.get("filename") or file_info.get("actual_filename")
                    file_path = file_info["path"]
                    mime_type = file_info.get("mime_type")

                    # Upload to Gemini (will cache in database automatically)
                    upload_result = await asyncio.to_thread(
                        file_uploader.upload_pdf,
                        file_path,
                        filename,
                        mime_type
                    )

                    if "error" not in upload_result:
                        print(f"✅ Pre-warmed Gemini cache: {filename}")
                        return {"status": "success", "filename": filename}
                    else:
                        print(f"⚠️  Gemini pre-warm failed: {filename}: {upload_result['error']}")
                        return {"status": "error", "filename": filename}

                except Exception as e:
                    print(f"❌ Gemini pre-warm error for {file_info.get('filename')}: {e}")
                    return {"status": "error", "filename": file_info.get("filename")}

        print(f"🔥 Pre-warming Gemini cache for {len(successful_uploads)} files (background task)...")
        tasks = [upload_single_file(file_info) for file_info in successful_uploads]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        print(f"✅ Gemini pre-warm complete: {success_count}/{len(successful_uploads)} files cached")

    except Exception as e:
        print(f"❌ Critical error in Gemini pre-warm: {e}")
        import traceback
        traceback.print_exc()


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
            storage_manager=storage_manager,
            chat_storage=chat_storage  # PHASE 3: Enable database caching
        )

        # Semaphore to limit concurrent processing (max 20 at a time)
        semaphore = asyncio.Semaphore(20)

        async def process_with_limit(file_info):
            """Wrapper to apply semaphore limit"""
            async with semaphore:
                return await _generate_single_summary(
                    file_info, course_id, file_uploader, file_summarizer, chat_storage
                )

        print(f"📝 Generating summaries for {len(successful_uploads)} files (max 20 concurrent)...")

        # Generate summaries with concurrency limit
        tasks = [process_with_limit(file_info) for file_info in successful_uploads]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count results
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        cached_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "cached")
        skipped_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
        error_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "error")
        exception_count = sum(1 for r in results if isinstance(r, Exception))

        print(f"✅ Summary generation complete: {success_count} new, {cached_count} cached, {skipped_count} skipped, {error_count + exception_count} errors")

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

        # PHASE 4: Process files in parallel (reduced from 100 to 50 for Railway safety)
        # Prevents memory exhaustion on Railway's limited resources
        BATCH_SIZE = 50
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

        # PHASE 3: Pre-warm Gemini File API cache (background task)
        # This eliminates 10-30s upload wait on first query
        if successful and chat_storage:
            print(f"🔥 Starting Gemini pre-warm for {len(successful)} files...")
            asyncio.create_task(_upload_to_gemini_background(course_id, successful))

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


@app.post("/check_files_exist")
async def check_files_exist(
    course_id: str,
    files: List[Dict]
):
    """
    Check which files exist in GCS and return signed URLs for existing files

    PHASE 2 OPTIMIZATION: Allows frontend to skip downloading from Canvas
    if files already exist in GCS (much faster to download from GCS)

    Args:
        course_id: Course identifier
        files: List of file objects with 'name' and optionally 'size' fields

    Returns:
        {
            "exists": [{"name": "file.pdf", "url": "https://...", "size": 123}],
            "missing": ["other_file.pdf"]
        }

    Performance: Checks 100 files in ~500ms
    """
    try:
        print(f"\n{'='*80}")
        print(f"📋 CHECK FILES EXIST REQUEST:")
        print(f"   Course ID: {course_id}")
        print(f"   Files to check: {len(files)}")
        print(f"{'='*80}")

        if not storage_manager:
            # If no GCS, return all as missing
            return {
                "exists": [],
                "missing": [f["name"] for f in files]
            }

        exists = []
        missing = []

        for file_info in files:
            file_name = file_info.get("name")
            if not file_name:
                continue

            # Check both original name and potential .pdf conversion
            # (since Phase 1 converts Office files to PDF)
            blob_name = f"{course_id}/{file_name}"
            pdf_blob_name = None

            # If it's an Office file, also check for PDF version
            if file_name.endswith(('.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls', '.rtf')):
                pdf_name = file_name.rsplit('.', 1)[0] + '.pdf'
                pdf_blob_name = f"{course_id}/{pdf_name}"

            # Check if file exists (try PDF version first if applicable)
            found_blob_name = None
            if pdf_blob_name and storage_manager.file_exists(pdf_blob_name):
                found_blob_name = pdf_blob_name
            elif storage_manager.file_exists(blob_name):
                found_blob_name = blob_name

            if found_blob_name:
                # File exists - generate signed URL (valid for 1 hour)
                signed_url = storage_manager.get_signed_url(found_blob_name, expires_in_seconds=3600)

                if signed_url:
                    # Get file size from GCS
                    blob = storage_manager.bucket.blob(found_blob_name)
                    blob.reload()  # Load metadata

                    exists.append({
                        "name": file_name,  # Original name for frontend
                        "actual_name": found_blob_name.split('/')[-1],  # What's stored in GCS
                        "url": signed_url,
                        "size": blob.size,
                        "from_gcs": True
                    })
                else:
                    # Failed to generate URL, treat as missing
                    missing.append(file_name)
            else:
                missing.append(file_name)

        print(f"✅ Check results: {len(exists)} exist in GCS, {len(missing)} missing")
        if exists:
            print(f"   Existing files: {[f['name'] for f in exists[:5]]}" + ("..." if len(exists) > 5 else ""))
        if missing:
            print(f"   Missing files: {missing[:5]}" + ("..." if len(missing) > 5 else ""))

        return {
            "exists": exists,
            "missing": missing
        }

    except Exception as e:
        print(f"❌ ERROR in check_files_exist endpoint:")
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

    # Store stop flag in connection dict so it can be accessed globally
    active_connections[connection_id] = {
        "websocket": websocket,
        "stop_streaming": False
    }

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Check if this is a stop signal
            if message_data.get("type") == "stop":
                import time
                print(f"🛑 Stop signal received for {connection_id} at {time.time()}")
                active_connections[connection_id]["stop_streaming"] = True
                print(f"🛑 Set stop_streaming flag to: {active_connections[connection_id]['stop_streaming']}")
                await websocket.send_json({
                    "type": "stopped",
                    "message": "Generation stopped by user"
                })
                continue

            # Reset stop flag for new queries
            active_connections[connection_id]["stop_streaming"] = False
            import time
            print(f"🚀 Starting new query at {time.time()}")

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

            # Define stop check callback
            def should_stop():
                return active_connections.get(connection_id, {}).get("stop_streaming", False)

            async for chunk in root_agent.process_query_stream(
                course_id=course_id,
                user_message=user_message,
                conversation_history=conversation_history,
                selected_docs=selected_docs,
                syllabus_id=syllabus_id,
                session_id=connection_id,
                enable_web_search=enable_web_search,
                user_api_key=user_api_key,
                use_smart_selection=use_smart_selection,
                stop_check_callback=should_stop
            ):
                # Check if stop signal received (double check in case callback didn't trigger)
                if should_stop():
                    print(f"🛑 Stopping stream early at chunk {chunk_count}")
                    break

                chunk_count += 1
                assistant_response += chunk
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk
                })

            # Send completion signal (only if not stopped)
            if not active_connections.get(connection_id, {}).get("stop_streaming", False):
                print(f"✅ Completed ({chunk_count} chunks)")
                await websocket.send_json({
                    "type": "done"
                })
            else:
                print(f"🛑 Stream stopped at {chunk_count} chunks (saved tokens!)")

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
        if connection_id in active_connections:
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
    """Auto-detect or retrieve stored syllabus document for a course"""
    try:
        # First, check if there's a stored syllabus_id
        stored_syllabus_id = chat_storage.get_course_syllabus(course_id) if chat_storage else None

        if stored_syllabus_id:
            syllabus_doc = document_manager.get_document_summary(course_id, stored_syllabus_id)
            return {
                "success": True,
                "syllabus_id": stored_syllabus_id,
                "syllabus_name": syllabus_doc.get("name") if syllabus_doc else None,
                "source": "stored"
            }

        # If not stored, try auto-detection
        syllabus_id = document_manager.find_syllabus(course_id)

        if syllabus_id:
            syllabus_doc = document_manager.get_document_summary(course_id, syllabus_id)
            # Auto-save detected syllabus
            if chat_storage:
                chat_storage.set_course_syllabus(course_id, syllabus_id)
            return {
                "success": True,
                "syllabus_id": syllabus_id,
                "syllabus_name": syllabus_doc.get("name") if syllabus_doc else None,
                "source": "auto-detected"
            }
        else:
            return {
                "success": True,
                "syllabus_id": None,
                "message": "No syllabus found",
                "source": None
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/courses/{course_id}/syllabus")
async def set_syllabus(course_id: str, syllabus_id: str):
    """Manually set the syllabus document for a course"""
    try:
        if not chat_storage:
            return {
                "success": False,
                "error": "Chat storage not available"
            }

        # Verify the syllabus document exists
        syllabus_doc = document_manager.get_document_summary(course_id, syllabus_id)
        if not syllabus_doc:
            return {
                "success": False,
                "error": f"Document {syllabus_id} not found"
            }

        # Store the syllabus_id
        chat_storage.set_course_syllabus(course_id, syllabus_id)

        return {
            "success": True,
            "syllabus_id": syllabus_id,
            "syllabus_name": syllabus_doc.get("name"),
            "message": "Syllabus set successfully"
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
