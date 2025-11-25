"""
FastAPI Server for AI Study Assistant
Handles WebSocket connections, PDF uploads, and agent coordination

Latest updates:
- JSON mode for guaranteed valid summaries
- Correct Gemini model names (gemini-2.0-flash-lite)
- Optimized rate limiting (3 concurrent, 3.0s delay)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Header
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

# Enable Gzip compression for responses (60-80% size reduction for JSON)
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global instances
root_agent = None
document_manager = None
chat_storage = None
storage_manager = None
file_summarizer = None

# Global semaphore for file conversions (limit concurrency to prevent memory overload)
conversion_semaphore = asyncio.Semaphore(3)  # 3 concurrent conversions (optimized for speed+memory)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# In-memory cache for filename mappings: (course_id, original_filename) -> actual_gcs_filename
# This eliminates the need to guess/check multiple filenames when serving files
# Populated during check_files_exist and uploads
filename_cache: Dict[tuple, str] = {}

# PDF storage directory (deprecated - using GCS now, but kept for backward compatibility)
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ========== Summary Retry System ==========
from collections import deque
import time

# Failed summary queue: {course_id: [{"file_id": ..., "filename": ..., "attempts": 0, "last_error": ...}]}
failed_summary_queue: Dict[str, List[Dict]] = {}

# Rate limit tracking: rolling window of last 20 API calls (True=success, False=failure)
recent_api_results = deque(maxlen=20)

# Adaptive delay state
current_delay = 2.0  # Start with 2 seconds
rate_limit_cooldown_until = 0  # Timestamp when cooldown ends

# Background retry worker state
retry_worker_task = None


async def _summary_retry_worker():
    """Background worker that retries failed summaries every 10 minutes"""
    while True:
        try:
            await asyncio.sleep(600)  # Wait 10 minutes between retry cycles

            if not failed_summary_queue:
                continue

            print(f"🔄 Summary retry worker: checking {sum(len(v) for v in failed_summary_queue.values())} failed summaries...")

            for course_id, failed_items in list(failed_summary_queue.items()):
                if not failed_items:
                    continue

                print(f"   Retrying {len(failed_items)} summaries for course {course_id}")

                # Retry each failed item
                retry_results = []
                for item in failed_items[:]:  # Copy list to allow modification
                    item["attempts"] += 1

                    # Max 3 background retries per file
                    if item["attempts"] > 3:
                        print(f"   ❌ Max retries reached for {item['filename']}, removing from queue")
                        failed_items.remove(item)
                        continue

                    # Retry summary generation
                    try:
                        from utils.file_upload_manager import FileUploadManager
                        api_key = os.getenv("GOOGLE_API_KEY")
                        file_upload_client = genai.Client(api_key=api_key)
                        file_uploader = FileUploadManager(
                            file_upload_client,
                            cache_duration_hours=48,
                            storage_manager=storage_manager,
                            chat_storage=chat_storage
                        )

                        result = await _generate_single_summary(
                            item["file_info"], course_id, file_uploader, file_summarizer, chat_storage, None,
                            document_manager=document_manager, storage_manager=storage_manager
                        )

                        if result.get("status") in ["success", "cached"]:
                            print(f"   ✅ Retry successful for {item['filename']}")
                            failed_items.remove(item)
                        elif result.get("status") == "removed":
                            # File was removed due to validation failure - don't retry
                            print(f"   🗑️  Removed invalid file from queue: {item['filename']}")
                            failed_items.remove(item)
                        else:
                            print(f"   ⚠️  Retry failed for {item['filename']}: {result.get('error', 'Unknown')}")
                            item["last_error"] = result.get('error', 'Unknown')[:200]

                    except Exception as e:
                        print(f"   ❌ Retry exception for {item['filename']}: {e}")
                        item["last_error"] = str(e)[:200]

                # Clean up empty course entries
                if not failed_items:
                    del failed_summary_queue[course_id]

        except Exception as e:
            print(f"❌ Error in retry worker: {e}")
            import traceback
            traceback.print_exc()


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

    # PHASE 4: Lazy Loading Cache System (No Startup Pre-Warming)
    # Files are cached on-demand during queries for scalability
    # This ensures fast startup regardless of file count (100K+ files supported)
    if chat_storage:
        try:
            cache_stats = chat_storage.get_cache_stats()
            print(f"📊 Gemini Cache Stats:")
            print(f"   Files cached: {cache_stats.get('total_files', 0)}")
            print(f"   Courses covered: {cache_stats.get('courses_count', 0)}")
            print(f"   Expiring soon: {cache_stats.get('expiring_soon_count', 0)}")
            print(f"   Cache health: {'✅ Excellent' if cache_stats.get('expiring_soon_count', 0) == 0 else '⚠️  Good'}")
        except Exception as e:
            print(f"⚠️  Failed to get cache stats: {e}")

        # Start proactive cache refresh loop (refreshes expiring files every 6 hours)
        try:
            asyncio.create_task(_proactive_cache_refresh_loop())
            print(f"🔄 Proactive cache refresh loop started")
        except Exception as e:
            print(f"⚠️  Failed to start cache refresh loop: {e}")

    # Start background retry worker for failed summaries
    global retry_worker_task
    try:
        retry_worker_task = asyncio.create_task(_summary_retry_worker())
        print(f"🔄 Summary retry worker started")
    except Exception as e:
        print(f"⚠️  Failed to start retry worker: {e}")

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


async def _process_single_upload(course_id: str, file: UploadFile, precomputed_hash: Optional[str] = None) -> Dict:
    """Process a single file upload (for parallel execution)

    Args:
        course_id: Course identifier
        file: File to process
        precomputed_hash: Optional pre-computed SHA-256 hash (for optimization)
    """
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
            mime_type = ext_to_mime.get(ext.lower(), 'application/pdf')  # Treat unknown as PDF
            print(f"⚠️  Unknown file type for {file.filename}, treating as PDF")

        print(f"📥 Processing: {file.filename} ({ext.upper()}, {mime_type})")

        # Read file content
        content = await file.read()
        original_filename = file.filename
        actual_filename = file.filename
        conversion_info = None

        # Use pre-computed hash if available, otherwise compute it
        if precomputed_hash:
            content_hash = precomputed_hash
            print(f"🔑 Using pre-computed hash: {content_hash[:16]}... (optimization)")
        else:
            # CRITICAL: Compute SHA-256 hash of ORIGINAL content (before conversion)
            # This ensures same file = same hash even after PPTX→PDF conversion
            import hashlib
            content_hash = hashlib.sha256(content).hexdigest()
            print(f"🔑 Content hash: {content_hash[:16]}...")

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
                # Convert Office/OpenDocument formats to PDF (with concurrency limit)
                print(f"🔄 Converting {file.filename} ({ext.upper()}) to PDF before upload...")
                try:
                    loop = asyncio.get_event_loop()
                    async with conversion_semaphore:  # Limit concurrent conversions
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
            # HASH-BASED: Use hash for GCS path with correct extension
            # Extract extension from actual_filename to preserve file type (.txt, .pdf, etc.)
            import os
            file_ext = os.path.splitext(actual_filename)[1] or '.pdf'  # Default to .pdf if no extension
            hash_filename = f"{content_hash}{file_ext}"
            blob_name = await loop.run_in_executor(
                None,
                storage_manager.upload_pdf,
                course_id,
                hash_filename,  # Use hash-based filename with correct extension
                content,
                mime_type  # Pass MIME type
            )
            # HASH-BASED: doc_id is now {course_id}_{hash} instead of {course_id}_{filename}
            doc_id = f"{course_id}_{content_hash}"
            result = {
                "filename": original_filename,  # Original name for frontend display
                "doc_id": doc_id,  # Hash-based ID for matching
                "hash": content_hash,  # Include hash for frontend
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
            # HASH-BASED: Use hash for local path with correct extension
            import os
            file_ext = os.path.splitext(actual_filename)[1] or '.pdf'  # Default to .pdf if no extension
            hash_filename = f"{content_hash}{file_ext}"
            file_path = UPLOAD_DIR / f"{course_id}_{hash_filename}"

            # Run file write in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: open(file_path, "wb").write(content)
            )

            # HASH-BASED: doc_id is now {course_id}_{hash}
            doc_id = f"{course_id}_{content_hash}"
            result = {
                "filename": original_filename,  # Original name for frontend display
                "doc_id": doc_id,  # Hash-based ID for matching
                "hash": content_hash,  # Include hash for frontend
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


def _sync_summarize_text_content(summarizer, text_content, filename):
    """Synchronous wrapper for text content summarization - runs in thread pool"""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            summarizer.summarize_text_content(text_content, filename)
        )
    finally:
        loop.close()


# Global tracking for removed files (for status endpoint)
removed_files_tracker: Dict[str, List[Dict]] = {}


async def _remove_invalid_file(
    course_id: str,
    file_info: Dict,
    error_reason: str,
    document_manager,
    chat_storage,
    storage_manager
):
    """
    Remove invalid file from catalog, database, and storage when validation fails

    Args:
        course_id: Course identifier
        file_info: File information dict
        error_reason: Why the file is invalid
        document_manager: DocumentManager instance
        chat_storage: ChatStorage instance
        storage_manager: StorageManager instance
    """
    doc_id = file_info.get('doc_id')
    filename = file_info.get('filename')
    gcs_path = file_info.get('gcs_path')

    print(f"🗑️  Removing invalid file: {filename} (Reason: {error_reason})")

    # 1. Remove from document catalog
    document_manager.remove_material(course_id, doc_id)

    # 2. Soft delete from database
    chat_storage.soft_delete_file(doc_id)

    # 3. Delete from GCS storage (optional - saves space)
    if gcs_path and storage_manager:
        try:
            storage_manager.delete_file(gcs_path)
            print(f"   ✅ Deleted from GCS: {gcs_path}")
        except Exception as e:
            print(f"   ⚠️  Failed to delete from GCS: {e}")

    # 4. Track for frontend notification
    if course_id not in removed_files_tracker:
        removed_files_tracker[course_id] = []

    removed_files_tracker[course_id].append({
        'doc_id': doc_id,
        'filename': filename,
        'reason': error_reason,
        'timestamp': time.time()
    })

    print(f"   ✅ File removed from all systems: {filename}")


async def _generate_single_summary(
    file_info: Dict,
    course_id: str,
    file_uploader,
    file_summarizer,
    chat_storage,
    canvas_user_id: Optional[str] = None,
    document_manager=None,
    storage_manager=None
) -> Dict:
    """Generate summary for a single file (optimized with thread pools)"""
    try:
        # Removed verbose log - batch summary shows progress
        # HASH-BASED: Use doc_id from upload result (format: {course_id}_{hash})
        doc_id = file_info.get("doc_id")
        content_hash = file_info.get("hash")
        filename = file_info.get("filename")  # Original filename for display
        file_path = file_info["path"]

        # Handle reconstructed upload_info from catalog (missing doc_id/hash)
        # Extract from file_id which has format: {course_id}_{hash}
        if not doc_id:
            doc_id = file_info.get("file_id")

        if not content_hash and doc_id:
            # Extract hash from doc_id (format: {course_id}_{hash})
            parts = doc_id.split('_', 1)
            if len(parts) == 2:
                content_hash = parts[1]

        # DEBUG: Log canvas_user_id to track propagation
        print(f"🔍 DEBUG _generate_single_summary for {filename}: canvas_user_id={canvas_user_id}")

        # CRITICAL: doc_id and hash are required for hash-based system
        if not doc_id or not content_hash:
            error_msg = f"Missing doc_id or hash for {filename} - upload may have failed"
            print(f"❌ {error_msg}")
            return {"status": "error", "filename": filename, "error": error_msg}

        # Check if summary already exists (cached in database)
        existing = chat_storage.get_file_summary(doc_id)
        if existing:
            # Update canvas_user_id even for cached summaries to ensure proper user tracking
            if canvas_user_id and not existing.get('canvas_user_id'):
                chat_storage.save_file_summary(
                    doc_id=doc_id,
                    course_id=course_id,
                    filename=filename,
                    summary=existing['summary'],
                    topics=existing.get('topics', []),
                    metadata=existing.get('metadata', {}),
                    content_hash=content_hash,
                    canvas_user_id=canvas_user_id
                )
                print(f"✅ Using cached summary for {filename} (updated user tracking)")
            else:
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
            # Check if this is a validation error (proactively detected 400 error)
            if upload_result.get("validation_failed"):
                # Remove file from catalog/database/storage (don't retry)
                await _remove_invalid_file(
                    course_id,
                    file_info,
                    upload_result["error"],
                    document_manager,
                    chat_storage,
                    storage_manager
                )
                return {"status": "removed", "filename": filename, "error": upload_result["error"]}

            # Other errors - allow retry
            return {"status": "error", "filename": filename, "error": upload_result["error"]}

        # Check if this is a text file (assignments/pages)
        if upload_result.get("is_text"):
            # Text file - use text content directly for summarization
            text_content = upload_result.get("text_content")
            if not text_content:
                return {"status": "error", "filename": filename, "error": "No text content found"}

            mime_type = upload_result["mime_type"]

            # Generate summary from text content (run in thread pool - BLOCKING LLM call)
            try:
                summary, topics, metadata = await asyncio.to_thread(
                    _sync_summarize_text_content,
                    file_summarizer,
                    text_content,
                    filename
                )
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Failed to generate summary for text file {filename}: {error_msg[:100]}")
                return {"status": "error", "filename": filename, "error": error_msg}
        else:
            # Regular file with Gemini File API URI
            file_uri = upload_result["file"].uri
            mime_type = upload_result["mime_type"]

            # Generate summary (run in thread pool - BLOCKING LLM call)
            try:
                summary, topics, metadata = await asyncio.to_thread(
                    _sync_summarize_file,
                    file_summarizer,
                    file_uri,
                    filename,
                    mime_type
                )
            except Exception as e:
                # Don't save errors as summaries - return error status for retry later
                error_msg = str(e)
                print(f"❌ Failed to generate summary for {filename}: {error_msg[:100]}")
                return {"status": "error", "filename": filename, "error": error_msg}

        # Save to database (cache for future uploads) with hash
        summary_preview = summary[:50] + "..." if len(summary) > 50 else summary
        print(f"✅ Summary: {filename[:40]}... → \"{summary_preview}\"")
        print(f"🔍 DEBUG Saving summary with canvas_user_id={canvas_user_id}, gcs_path={file_path}")
        success = chat_storage.save_file_summary(
            doc_id=doc_id,
            course_id=course_id,
            filename=filename,
            summary=summary,
            topics=topics,
            metadata=metadata,
            content_hash=content_hash,  # Store hash for matching
            canvas_user_id=canvas_user_id,  # Track who uploaded
            gcs_path=file_path  # Store actual GCS path for deletion
        )

        if success:
            return {"status": "success", "filename": filename}
        else:
            return {"status": "error", "filename": filename, "error": "Failed to save"}

    except Exception as e:
        return {"status": "error", "filename": file_info.get("filename", "unknown"), "error": str(e)}


async def _generate_batch_summaries(
    files_batch: List[Dict],
    course_id: str,
    file_uploader,
    file_summarizer,
    chat_storage,
    canvas_user_id: Optional[str] = None
) -> List[Dict]:
    """
    Generate summaries for a batch of files using a single API call

    Args:
        files_batch: List of file info dicts (already uploaded to Gemini)
        course_id: Canvas course ID
        file_uploader: FileUploadManager instance
        file_summarizer: FileSummarizer instance
        chat_storage: ChatStorage instance
        canvas_user_id: Optional Canvas user ID

    Returns:
        List of result dicts with status for each file
    """
    try:
        # Prepare batch input for file_summarizer
        batch_input = []
        for file_info in files_batch:
            batch_input.append({
                'file_id': file_info.get('file_id'),
                'filename': file_info.get('filename'),
                'uri': file_info.get('uri'),
                'mime_type': file_info.get('mime_type'),
                'text_content': file_info.get('text_content')  # For text files
            })

        # Call batch summarizer (async method with thread-pooled Gemini calls)
        batch_result = await file_summarizer.summarize_files_batch(batch_input)

        # Process results
        results = []

        # Handle successful summaries
        for summary_info in batch_result.get('summaries', []):
            file_id = summary_info['file_id']
            filename = summary_info['filename']
            summary = summary_info['summary']
            topics = summary_info['topics']
            metadata = summary_info['metadata']

            # Find original file info for additional metadata
            original_file = next((f for f in files_batch if f.get('file_id') == file_id), {})
            file_path = original_file.get('gcs_path', f"{course_id}/{filename}")
            content_hash = original_file.get('content_hash', '')

            # Save to database
            success = chat_storage.save_file_summary(
                file_id=file_id,
                course_id=course_id,
                filename=filename,
                summary=summary,
                topics=topics,
                metadata=metadata,
                content_hash=content_hash,
                canvas_user_id=canvas_user_id,
                gcs_path=file_path
            )

            if success:
                results.append({"status": "success", "filename": filename, "file_id": file_id})
            else:
                results.append({"status": "error", "filename": filename, "file_id": file_id, "error": "Failed to save"})

        # Handle failed summaries
        for failed_info in batch_result.get('failed', []):
            results.append({
                "status": "error",
                "filename": failed_info['filename'],
                "file_id": failed_info['file_id'],
                "error": failed_info['error']
            })

        return results

    except Exception as e:
        # Entire batch failed - return error for all files
        print(f"❌ Batch summary generation failed: {e}")
        return [
            {"status": "error", "filename": f.get('filename', 'unknown'), "file_id": f.get('file_id'), "error": str(e)}
            for f in files_batch
        ]


async def _upload_to_gemini_background(course_id: str, successful_uploads: List[Dict], canvas_user_id: Optional[str] = None):
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

        # PHASE 4: Priority upload queue - Reduced concurrency to prevent rate limits
        # Low concurrency (2) to avoid API quota conflicts during pre-warming
        semaphore = asyncio.Semaphore(2)

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

        print(f"🔥 Pre-warming Gemini cache for {len(successful_uploads)} files (priority queue, 10 concurrent)...")
        tasks = [upload_single_file(file_info) for file_info in successful_uploads]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        print(f"✅ Gemini pre-warm complete: {success_count}/{len(successful_uploads)} files cached (ready for fast queries!)")

    except Exception as e:
        print(f"❌ Critical error in Gemini pre-warm: {e}")
        import traceback
        traceback.print_exc()


async def _generate_summaries_background(course_id: str, successful_uploads: List[Dict], canvas_user_id: Optional[str] = None):
    """
    Background task to generate summaries for uploaded files in parallel
    """
    try:
        print(f"📝 _generate_summaries_background called with canvas_user_id: {canvas_user_id}")
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

        # Semaphore to limit concurrent processing (reduced to 3 to prevent rate limit bursts)
        # Keep low to avoid Gemini API rate limits on free tier (30 RPM for gemini-2.0-flash-lite)
        semaphore = asyncio.Semaphore(3)

        # Check ALL files in catalog for missing summaries (not just newly uploaded)
        files_to_summarize = []
        catalog = document_manager.get_material_catalog(course_id)
        all_materials = catalog.get("materials", [])

        print(f"🔍 Checking {len(all_materials)} files for missing summaries...")

        for material in all_materials:
            file_id = material.get('id')
            if not file_id:
                continue

            # Check if summary exists in database
            existing_summary = chat_storage.get_file_summary(file_id)
            if not existing_summary:
                # Find matching upload info from this batch, or create minimal info
                upload_info = next(
                    (f for f in successful_uploads if f.get('file_id') == file_id),
                    {
                        'file_id': file_id,
                        'filename': material['name'],  # Use original display name, not hash-based filename
                        'path': material['path'],
                        'gcs_path': f"{course_id}/{material['filename']}",
                        'size_mb': material.get('size_mb', 0),
                        'num_pages': material.get('num_pages', 0)
                    }
                )
                files_to_summarize.append(upload_info)

        if not files_to_summarize:
            print("✅ All files already have summaries!")
            return

        print(f"📝 Found {len(files_to_summarize)} files missing summaries")

        # Replace successful_uploads with files that need summaries
        successful_uploads = files_to_summarize

        # PHASE 2: BATCH SUMMARIZATION
        # Step 1: Upload all files to Gemini to get URIs (pre-warming phase)
        print(f"\n📤 PHASE 1: Uploading {len(successful_uploads)} files to Gemini...")

        # Upload files with high concurrency to get URIs quickly
        upload_semaphore = asyncio.Semaphore(10)  # Higher concurrency for uploads

        async def upload_with_limit(file_info):
            async with upload_semaphore:
                filename = file_info.get("filename") or file_info.get("actual_filename")
                file_path = file_info["path"]
                mime_type = file_info.get("mime_type")

                upload_result = await asyncio.to_thread(
                    file_uploader.upload_pdf,
                    file_path,
                    filename,
                    mime_type
                )

                # Check for validation failures
                if "error" in upload_result:
                    if upload_result.get("validation_failed"):
                        # Remove invalid file
                        await _remove_invalid_file(
                            course_id,
                            file_info,
                            upload_result["error"],
                            document_manager,
                            chat_storage,
                            storage_manager
                        )
                        return {"status": "removed", "file_info": file_info, "error": upload_result["error"]}

                    # Other upload errors
                    return {"status": "upload_error", "file_info": file_info, "error": upload_result["error"]}

                # Success - add upload result to file_info for batch processing
                return {
                    "status": "uploaded",
                    "file_info": {
                        **file_info,
                        "uri": upload_result.get("uri"),
                        "text_content": upload_result.get("text_content"),
                        "mime_type": upload_result.get("mime_type")
                    }
                }

        # Upload all files concurrently
        upload_tasks = [upload_with_limit(f) for f in successful_uploads]
        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)

        # Separate uploaded files from failed uploads
        uploaded_files = []
        for result in upload_results:
            if isinstance(result, dict) and result.get("status") == "uploaded":
                uploaded_files.append(result["file_info"])
            elif isinstance(result, dict) and result.get("status") == "removed":
                # File was removed due to validation - skip it
                pass
            else:
                # Upload error - will be added to retry queue below
                pass

        print(f"✅ UPLOAD PHASE COMPLETE: {len(uploaded_files)} files uploaded to Gemini\n")

        if not uploaded_files:
            print("⚠️  No files to summarize (all failed validation or upload)")
            return

        # Step 2: Group files into batches and summarize
        BATCH_SIZE = 12  # Optimal batch size (12 files per request)
        batches = [uploaded_files[i:i + BATCH_SIZE] for i in range(0, len(uploaded_files), BATCH_SIZE)]

        print(f"📝 PHASE 2: Summarizing {len(uploaded_files)} files in {len(batches)} batches (12 files per batch)...\n")

        all_results = []
        for batch_num, batch in enumerate(batches, 1):
            # Check for cooldown
            if time.time() < rate_limit_cooldown_until:
                wait_time = rate_limit_cooldown_until - time.time()
                print(f"   ⏸️  Rate limit cooldown: waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

            print(f"   Batch {batch_num}/{len(batches)}: Processing {len(batch)} files...")

            # Process batch
            batch_results = await _generate_batch_summaries(
                batch,
                course_id,
                file_uploader,
                file_summarizer,
                chat_storage,
                canvas_user_id
            )

            all_results.extend(batch_results)

            # Check for rate limits in batch results
            has_rate_limit = any(
                isinstance(r, dict) and '429' in r.get('error', '').lower()
                for r in batch_results
            )

            if has_rate_limit:
                print(f"   ⚠️  Rate limit detected! Entering 30s cooldown...")
                rate_limit_cooldown_until = time.time() + 30
                current_delay = 5.0
            else:
                # Adaptive delay between batches
                await asyncio.sleep(current_delay)

        results = all_results

        # Count results
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        cached_count = 0  # No caching in batch mode (checked earlier)
        skipped_count = 0
        error_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "error")
        exception_count = sum(1 for r in results if isinstance(r, Exception))
        removed_count = sum(1 for r in upload_results if isinstance(r, dict) and r.get("status") == "removed")

        print(f"✅ BATCH COMPLETE: {success_count} new, {error_count + exception_count} errors, {removed_count} removed\n")

        # Add failed summaries to retry queue
        if course_id not in failed_summary_queue:
            failed_summary_queue[course_id] = []

        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Exception: {result}")
                # We don't have file_info here, skip
                continue
            elif isinstance(result, dict) and result.get("status") == "error":
                # Find original file_info
                file_id = result.get('file_id')
                file_info = next((f for f in uploaded_files if f.get('file_id') == file_id), None)

                if file_info:
                    failed_summary_queue[course_id].append({
                        "file_id": file_id,
                        "filename": result['filename'],
                        "attempts": 0,
                        "last_error": result.get('error', '')[:200],
                        "file_info": file_info
                    })

        # Log failed queue status
        if failed_summary_queue[course_id]:
            print(f"📋 Added {len(failed_summary_queue[course_id])} failed summaries to retry queue for course {course_id}")

    except Exception as e:
        print(f"❌ Critical error in summary generation: {e}")
        import traceback
        traceback.print_exc()


async def _proactive_cache_refresh_loop():
    """
    PHASE 4: Proactive cache refresh background task

    Runs every 6 hours to refresh files expiring within 6 hours.
    Prevents cache misses for frequently-used courses.
    """
    try:
        # Wait 10 minutes after startup before starting refresh loop
        await asyncio.sleep(600)

        while True:
            try:
                print(f"\n🔄 Checking for files needing cache refresh...")

                # Get files expiring within 6 hours
                files_to_refresh = chat_storage.get_files_needing_cache_refresh(hours_before_expiry=6)

                if not files_to_refresh:
                    print(f"✅ No files need refresh - all caches are fresh")
                else:
                    print(f"🔄 Refreshing {len(files_to_refresh)} files expiring soon...")

                    # Set up file uploader
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        print("⚠️  No GOOGLE_API_KEY found, skipping refresh")
                        await asyncio.sleep(21600)  # 6 hours
                        continue

                    from utils.file_upload_manager import FileUploadManager
                    file_upload_client = genai.Client(api_key=api_key)
                    file_uploader = FileUploadManager(
                        file_upload_client,
                        cache_duration_hours=48,
                        storage_manager=storage_manager,
                        chat_storage=chat_storage
                    )

                    # Refresh files with concurrency limit
                    semaphore = asyncio.Semaphore(5)  # Lower concurrency for background refresh

                    async def refresh_file(file_info):
                        async with semaphore:
                            try:
                                # Upload to Gemini (will update cache automatically)
                                upload_result = await asyncio.to_thread(
                                    file_uploader.upload_pdf,
                                    file_info['file_path'],
                                    file_info['filename']
                                )

                                if "error" not in upload_result:
                                    return {"status": "success", "filename": file_info['filename']}
                                else:
                                    return {"status": "error", "filename": file_info['filename']}

                            except Exception as e:
                                return {"status": "error", "filename": file_info['filename'], "error": str(e)}

                    tasks = [refresh_file(file_info) for file_info in files_to_refresh]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
                    print(f"✅ Cache refresh complete: {success_count}/{len(files_to_refresh)} files refreshed")

            except Exception as e:
                print(f"⚠️  Error in cache refresh cycle: {e}")

            # Wait 6 hours before next refresh cycle
            print(f"💤 Next cache refresh check in 6 hours")
            await asyncio.sleep(21600)  # 6 hours

    except Exception as e:
        print(f"❌ Critical error in cache refresh loop: {e}")
        import traceback
        traceback.print_exc()


async def _process_uploads_background(course_id: str, files_in_memory: List[Dict], canvas_user_id: Optional[str] = None):
    """
    Background task to process file uploads after instant response

    This function does ALL the heavy lifting AFTER the frontend already opened the chat:
    1. Convert files (Office→PDF, HTML→TXT)
    2. Upload to GCS/local storage
    3. Update document catalog
    4. Pre-warm Gemini cache
    5. Generate summaries

    The user doesn't wait for any of this - they get instant chat access!
    """
    try:
        print(f"\n🔄 BACKGROUND PROCESSING STARTED: {len(files_in_memory)} files for course {course_id} (user: {canvas_user_id})")

        # Create UploadFile-like objects from memory
        from fastapi import UploadFile
        from io import BytesIO

        upload_files = []
        file_hashes = []  # Track pre-computed hashes
        for file_data in files_in_memory:
            # Create a file-like object from bytes
            file_obj = BytesIO(file_data['content'])
            file_obj.name = file_data['filename']

            # Recreate UploadFile
            upload_file = UploadFile(
                filename=file_data['filename'],
                file=file_obj
            )
            upload_files.append(upload_file)

            # Store pre-computed hash (if available)
            file_hashes.append(file_data.get('hash'))

        # PHASE 1: Process files in parallel (GCS upload + conversion)
        # Dynamic batch sizing: Adjust based on average file size to prevent Railway memory spikes
        total_size = sum(len(f['content']) for f in files_in_memory)
        avg_size = total_size / len(files_in_memory) if files_in_memory else 0

        # Target: Keep batch memory under 200MB (Railway constraint)
        MAX_BATCH_MEMORY = 200 * 1024 * 1024  # 200MB
        if avg_size > 0:
            BATCH_SIZE = int(MAX_BATCH_MEMORY / avg_size)
            BATCH_SIZE = max(10, min(BATCH_SIZE, 100))  # Clamp between 10-100
        else:
            BATCH_SIZE = 50  # Default

        avg_size_mb = avg_size / (1024 * 1024) if avg_size > 0 else 0
        print(f"📊 Dynamic batch size: {BATCH_SIZE} (avg file: {avg_size_mb:.2f}MB, target: <200MB per batch)")
        print(f"📤 Processing {len(upload_files)} files in batches of {BATCH_SIZE}...")

        processed_results = []
        for i in range(0, len(upload_files), BATCH_SIZE):
            batch = upload_files[i:i + BATCH_SIZE]
            batch_hashes = file_hashes[i:i + BATCH_SIZE]  # Corresponding hashes
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(upload_files) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"\n📤 BATCH {batch_num}/{total_batches}: Uploading {len(batch)} files to GCS...")

            # Pass pre-computed hashes to avoid re-computation
            upload_tasks = [
                _process_single_upload(course_id, file, precomputed_hash=batch_hashes[j])
                for j, file in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"❌ Exception for {batch[j].filename}: {result}")
                    processed_results.append({
                        "filename": batch[j].filename,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

            # Count batch results
            batch_successful = sum(1 for r in batch_results if isinstance(r, dict) and r.get("status") == "uploaded")
            batch_skipped = sum(1 for r in batch_results if isinstance(r, dict) and r.get("status") == "skipped")
            batch_failed = sum(1 for r in batch_results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("status") == "failed"))
            print(f"✅ Batch complete: {batch_successful} uploaded, {batch_skipped} skipped, {batch_failed} failed")

            # Progressive summarization: Start summaries for this batch immediately
            successful_batch = [r for r in batch_results
                               if isinstance(r, dict) and r.get("status") == "uploaded"]

            if successful_batch and file_summarizer and chat_storage:
                print(f"   ➜ Starting summaries for {len(successful_batch)} new files...")
                asyncio.create_task(
                    _generate_summaries_background(
                        course_id, successful_batch, canvas_user_id
                    )
                )

        # Count results
        successful = [r for r in processed_results if r["status"] == "uploaded"]
        failed = [r for r in processed_results if r["status"] == "failed"]
        skipped = [r for r in processed_results if r["status"] == "skipped"]

        print(f"✅ Background upload complete: {len(successful)} succeeded, {len(failed)} failed, {len(skipped)} skipped")

        # PHASE 2: Update catalog (include both uploaded and skipped files)
        files_for_catalog = successful + skipped  # Skipped files already exist in GCS, add them too
        if document_manager and files_for_catalog:
            print(f"📚 Adding {len(files_for_catalog)} files to catalog ({len(successful)} new, {len(skipped)} existing)...")
            # HASH-BASED: Pass full result objects to include hash info
            document_manager.add_files_to_catalog_with_metadata(files_for_catalog)
            print(f"✅ Catalog updated")

        print(f"✅ BACKGROUND PROCESSING COMPLETE for course {course_id}")
        print(f"   Files uploaded successfully. Summaries are being generated progressively.")

    except Exception as e:
        print(f"❌ Critical error in background upload processing: {e}")
        import traceback
        traceback.print_exc()


@app.post("/upload_pdfs")
async def upload_pdfs(
    course_id: str,
    files: List[UploadFile] = File(...),
    x_canvas_user_id: Optional[str] = Header(None, alias="X-Canvas-User-Id")
):
    """
    Upload files for a course (supports PDFs, documents, images, etc.)

    INSTANT RESPONSE MODE: Returns immediately after accepting files,
    processes everything in background for instant study bot creation.

    Args:
        course_id: Course identifier (used as filename prefix)
        files: List of files to upload (PDF, DOCX, TXT, images, etc.)
        x_canvas_user_id: Canvas user ID (passed via X-Canvas-User-Id header)

    Process:
        1. Accept files and store in memory
        2. Return IMMEDIATE success response (instant chat opening!)
        3. Background: Save to storage, update catalog, pre-warm Gemini cache

    Returns:
        JSON with instant success (files processed in background)

    Performance: <100ms response time (instant!)
    """
    try:
        print(f"\n{'='*80}")
        print(f"📤 UPLOAD REQUEST (INSTANT MODE):")
        print(f"   Course ID: {course_id}")
        print(f"   Canvas User ID: {x_canvas_user_id}")
        print(f"   Number of files: {len(files)}")
        print(f"   File names received:")
        for f in files[:10]:  # Show first 10
            print(f"      - \"{f.filename}\"")
        print(f"   Storage manager available: {storage_manager is not None}")
        print(f"{'='*80}")

        # Read all files into memory and compute hashes IMMEDIATELY
        # Hash computation is fast (~10ms per MB) and critical for file identification
        files_in_memory = []
        files_metadata = []

        for file in files:
            content = await file.read()

            # Compute hash immediately (needed for IndexedDB and file opening)
            import hashlib
            content_hash = hashlib.sha256(content).hexdigest()
            doc_id = f"{course_id}_{content_hash}"

            files_in_memory.append({
                'filename': file.filename,
                'content': content,
                'content_type': file.content_type,
                'hash': content_hash,  # Pre-computed hash for background processing
                'doc_id': doc_id
            })

            files_metadata.append({
                'filename': file.filename,
                'hash': content_hash,  # Return hash to frontend
                'doc_id': doc_id,      # Return doc_id to frontend
                'status': 'processing'
            })

        print(f"✅ Files accepted and hashed ({len(files_in_memory)} files, {sum(len(f['content']) for f in files_in_memory) / 1024 / 1024:.1f} MB)")
        print(f"   Sample hashes: {[f['hash'][:16] + '...' for f in files_in_memory[:3]]}")

        # Start background processing (non-blocking)
        asyncio.create_task(_process_uploads_background(course_id, files_in_memory, x_canvas_user_id))

        # FAST RESPONSE with hash data - User can open files immediately!
        return {
            "success": True,
            "message": f"Processing {len(files)} files in background",
            "files": files_metadata,  # Include hash and doc_id
            "uploaded_count": len(files_in_memory),
            "failed_count": 0,
            "instant_mode": True
        }

    except Exception as e:
        print(f"❌ CRITICAL ERROR in upload_pdfs endpoint:")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check_files_exist")
async def check_files_exist(request: Dict):
    """
    Check which files exist in GCS using hash-based identification

    Request format:
        {
            "course_id": "123456",
            "files": [{"name": "file.pdf", "hash": "abc123...", "url": "..."}, ...]
        }

    Returns:
        {
            "exists": [{"name": "file.pdf", "doc_id": "{course_id}_{hash}", "hash": "abc123..."}],
            "missing": [{"name": "file.pdf", "hash": "abc123..."}]
        }
    """
    try:
        course_id = request.get("course_id")
        files = request.get("files", [])

        if not course_id or not files:
            raise HTTPException(status_code=400, detail="course_id and files required")

        print(f"📋 [HASH-BASED] Checking {len(files)} files for course {course_id}")

        if not storage_manager:
            # If no GCS, return all as missing
            return {
                "exists": [],
                "missing": files
            }

        # Check which files exist by hash
        exists = []
        missing = []

        for file_info in files:
            file_hash = file_info.get("hash")
            file_name = file_info.get("name", "unknown")

            if not file_hash:
                # No hash provided - can't check, assume missing
                print(f"   ⚠️  No hash for {file_name}, marking as missing")
                missing.append(file_info)
                continue

            # Build the GCS blob path: course_id/hash.ext
            # Try multiple extensions since we don't know what was uploaded
            # Priority: .txt (assignments), .pdf (most common), no extension
            possible_extensions = ['.txt', '.pdf', '']
            blob_exists = False
            blob_name = None

            # Check if blob exists in GCS
            try:
                for ext in possible_extensions:
                    test_blob_name = f"{course_id}/{file_hash}{ext}"
                    if storage_manager.file_exists(test_blob_name):
                        blob_exists = True
                        blob_name = test_blob_name
                        break

                if blob_exists:
                    doc_id = f"{course_id}_{file_hash}"
                    exists.append({
                        "name": file_name,
                        "hash": file_hash,
                        "doc_id": doc_id
                    })
                    print(f"   ✅ Found: {file_name} (hash: {file_hash[:16]}...)")
                else:
                    missing.append(file_info)
                    print(f"   ❌ Missing: {file_name} (hash: {file_hash[:16]}...)")
            except Exception as e:
                print(f"   ⚠️  Error checking {file_name}: {e}")
                # On error, assume missing to be safe
                missing.append(file_info)

        print(f"✅ [HASH-BASED] Check complete: {len(exists)} exist, {len(missing)} missing")

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


@app.post("/process_canvas_files")
async def process_canvas_files(
    request: Dict,
    x_canvas_user_id: Optional[str] = Header(None, alias="X-Canvas-User-Id")
):
    """
    Process files from Canvas URLs - backend downloads and uploads to GCS

    This eliminates frontend download phase - much faster study bot creation!

    Args:
        {
            "course_id": "123456",
            "files": [
                {"name": "lecture.pdf", "url": "https://canvas.../files/123/download?..."},
                ...
            ]
        }

    Returns:
        {
            "success": True,
            "processed": 5,
            "skipped": 2,  # Already in GCS
            "failed": 0
        }
    """
    try:
        course_id = request.get("course_id")
        files = request.get("files", [])
        canvas_cookies = request.get("cookies", "")  # Session cookies for Canvas auth
        skip_existence_check = request.get("skip_check", False)  # NEW: Allow frontend to skip redundant check

        if not course_id or not files:
            raise HTTPException(status_code=400, detail="course_id and files required")

        print(f"🌐 Process {len(files)} files for course {course_id} (skip_check: {skip_existence_check})")
        print(f"   Canvas User ID: {x_canvas_user_id}")

        # Check which files already exist in GCS (skip if frontend already checked)
        existing_files = set()
        if not skip_existence_check:
            exists_check = await check_files_exist({"course_id": course_id, "files": files})
            existing_files = {f["name"] for f in exists_check["exists"]}

        processed = 0
        skipped = len(existing_files)
        failed = 0

        # Helper function to process a single file (for parallel execution)
        async def process_single_file(file_info, session):
            nonlocal processed, failed

            file_name = file_info.get("name")
            file_url = file_info.get("url")
            canvas_id = file_info.get("id")  # Canvas file ID for fallback matching

            if not file_name or not file_url:
                print(f"⏭️  Skipping {file_name or 'unnamed'}: missing name or url (canvas_id: {canvas_id})")
                return {"status": "skipped", "reason": "missing name or url", "filename": file_name, "canvas_id": str(canvas_id) if canvas_id else None}

            if file_name in existing_files:
                return {"status": "skipped", "reason": "already exists", "filename": file_name}

            try:
                # Convert Canvas API URL to download URL if needed
                download_url = file_url
                if '/api/v1/' in file_url and '/files/' in file_url:
                    import re
                    match = re.search(r'/files/(\d+)', file_url)
                    if match:
                        file_id = match.group(1)
                        base_url = file_url.split('/api/v1/')[0]
                        download_url = f"{base_url}/files/{file_id}/download?download_frd=1"

                # Download file from Canvas with retry logic
                headers = {}
                if canvas_cookies:
                    headers['Cookie'] = canvas_cookies

                try:
                    file_content, status, content_type_header = await download_with_retry(
                        session, download_url, headers, file_name
                    )

                    if status != 200:
                        print(f"❌ {file_name}: HTTP {status}")
                        failed += 1
                        return {"status": "failed", "error": f"HTTP {status}", "filename": file_name}
                except Exception as e:
                    # DNS/connection failure after retries
                    print(f"❌ {file_name}: {str(e)}")
                    failed += 1
                    return {"status": "failed", "error": str(e), "filename": file_name}

                # Process and upload (same logic as before)
                from utils.file_converter import convert_office_to_pdf, needs_conversion
                mime_type = get_mime_type(file_name)
                ext = get_file_extension(file_name)

                if not ext and content_type_header:
                    mime_to_ext = {
                        'application/pdf': 'pdf',
                        'application/vnd.ms-powerpoint': 'ppt',
                        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
                        'application/msword': 'doc',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                        'application/vnd.ms-excel': 'xls',
                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                        'text/plain': 'txt',
                        'text/html': 'html',
                        'image/png': 'png',
                        'image/jpeg': 'jpg',
                    }
                    ext = mime_to_ext.get(content_type_header)
                    if ext:
                        print(f"   Detected file type from Content-Type: {content_type_header} → .{ext}")
                        file_name = f"{file_name}.{ext}"
                        mime_type = content_type_header

                ext = ext or 'file'

                safe_filename = file_name.replace('/', '-')

                original_filename = file_name
                actual_filename = safe_filename

                # HASH-BASED: Compute SHA-256 hash of ORIGINAL content (before conversion)
                import hashlib
                content_hash = hashlib.sha256(file_content).hexdigest()
                print(f"🔑 Content hash: {content_hash[:16]}...")

                # Check if this file (by hash) already exists in GCS
                # Try multiple extensions since we don't know what was uploaded
                import os
                file_ext = os.path.splitext(actual_filename)[1] or '.pdf'
                hash_blob_name = f"{course_id}/{content_hash}{file_ext}"
                # Also check .pdf and .txt as fallback (for legacy files)
                possible_extensions = [file_ext, '.txt', '.pdf', '']
                hash_blob_exists = False
                for test_ext in possible_extensions:
                    test_blob_name = f"{course_id}/{content_hash}{test_ext}"
                    if storage_manager and storage_manager.file_exists(test_blob_name):
                        hash_blob_name = test_blob_name
                        hash_blob_exists = True
                        break

                if hash_blob_exists:
                    print(f"⏭️  {file_name} already exists in GCS (hash: {content_hash[:16]}...)")
                    doc_id = f"{course_id}_{content_hash}"
                    return {
                        "status": "skipped",
                        "reason": "already exists (by hash)",
                        "filename": file_name,
                        "doc_id": doc_id,
                        "hash": content_hash,
                        "path": hash_blob_name,
                        "storage": "gcs",
                        "size_bytes": len(file_content),
                        "canvas_id": str(canvas_id) if canvas_id else None
                    }

                if needs_conversion(safe_filename):
                    from utils.file_converter import convert_to_text
                    web_formats = ['html', 'htm', 'xml', 'json']
                    is_web_format = ext.lower() in web_formats

                    if is_web_format:
                        async with conversion_semaphore:  # Limit concurrent conversions
                            loop = asyncio.get_event_loop()
                            text_bytes = await loop.run_in_executor(None, convert_to_text, file_content, safe_filename)
                        if text_bytes:
                            file_content = text_bytes
                            actual_filename = safe_filename.rsplit('.', 1)[0] + '.txt'
                            mime_type = 'text/plain'
                    else:
                        async with conversion_semaphore:  # Limit concurrent conversions
                            loop = asyncio.get_event_loop()
                            pdf_bytes = await loop.run_in_executor(None, convert_office_to_pdf, file_content, safe_filename)
                        if pdf_bytes:
                            file_content = pdf_bytes
                            actual_filename = safe_filename.rsplit('.', 1)[0] + '.pdf'
                            mime_type = 'application/pdf'

                if storage_manager:
                    # HASH-BASED: Use hash for GCS filename with correct extension
                    import os
                    file_ext = os.path.splitext(actual_filename)[1] or '.pdf'  # Default to .pdf if no extension
                    hash_filename = f"{content_hash}{file_ext}"
                    blob_name = storage_manager.upload_pdf(
                        course_id,
                        hash_filename,
                        file_content,
                        mime_type,
                        original_filename=original_filename  # Store original name in GCS metadata
                    )
                    processed += 1
                    print(f"✅ {original_filename}")

                    # HASH-BASED: doc_id is {course_id}_{hash}
                    doc_id = f"{course_id}_{content_hash}"

                    # Return hash-based metadata for catalog
                    return {
                        "status": "uploaded",
                        "filename": original_filename,  # Original name for display
                        "doc_id": doc_id,  # Hash-based ID
                        "hash": content_hash,  # SHA-256 hash
                        "path": blob_name,  # GCS path
                        "size_bytes": len(file_content),
                        "mime_type": mime_type,
                        "storage": "gcs",
                        "canvas_id": str(canvas_id) if canvas_id else None  # Preserve Canvas file ID
                    }
                else:
                    failed += 1
                    return {"status": "failed", "error": "No storage manager", "filename": file_name}

            except Exception as e:
                print(f"❌ Error processing {file_name}: {e}")
                failed += 1
                return {"status": "failed", "error": str(e), "filename": file_name}

        # Process files in parallel (8 concurrent downloads/uploads)
        import aiohttp
        import asyncio

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(24)  # Optimized for speed (24 concurrent downloads)

        async def download_with_retry(session, url, headers, filename, max_retries=3):
            """Download file with retry logic for DNS/connection failures"""
            for attempt in range(max_retries):
                try:
                    async with session.get(url, headers=headers, allow_redirects=True) as response:
                        if response.status != 200:
                            return None, response.status, None
                        content = await response.read()
                        content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
                        return content, 200, content_type
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError, aiohttp.ClientError) as e:
                    error_msg = str(e)
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"   ⚠️  {filename}: DNS/connection error, retry {attempt+2}/{max_retries} in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        # Final attempt failed
                        print(f"   ❌ {filename}: Failed after {max_retries} retries: {error_msg}")
                        raise
            return None, None, None

        async def process_with_semaphore(file_info, session):
            async with semaphore:
                return await process_single_file(file_info, session)

        # Create single aiohttp session for all requests with timeout and connector config
        timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_read=60)
        connector = aiohttp.TCPConnector(limit=24)  # Removed force_close to prevent incomplete transfers
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Create tasks for all files
            tasks = [process_with_semaphore(file_info, session) for file_info in files]

            # Execute all tasks in parallel (limited to 8 concurrent)
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count results and collect file metadata with hashes
        uploaded_files = []
        skipped_files = []
        skipped_no_url = 0
        for result in results:
            if isinstance(result, Exception):
                failed += 1
            elif isinstance(result, dict):
                if result.get("status") == "uploaded":
                    uploaded_files.append(result)  # Full result with hash metadata
                elif result.get("status") == "skipped":
                    reason = result.get("reason", "")
                    if "already exists" in reason:
                        # File exists in GCS (by hash or name)
                        skipped_files.append(result)  # Skipped files also have hash metadata
                        skipped += 1  # Count hash-based skips
                    elif "missing name or url" in reason:
                        skipped_no_url += 1
                    else:
                        skipped_files.append(result)

        print(f"✅ Complete: {processed} uploaded, {skipped} skipped, {failed} failed, {skipped_no_url} missing URLs")

        # Add uploaded AND skipped files to catalog with hash-based metadata
        files_for_catalog = uploaded_files + skipped_files
        if files_for_catalog and document_manager:
            try:
                print(f"📚 Adding {len(files_for_catalog)} files to catalog ({len(uploaded_files)} new, {len(skipped_files)} existing)...")
                document_manager.add_files_to_catalog_with_metadata(files_for_catalog)
                print(f"✅ Catalog updated")
            except Exception as e:
                print(f"⚠️ Catalog update failed: {e}")

        # Generate summaries for uploaded files with hash-based metadata
        if file_summarizer and chat_storage and processed > 0:
            print(f"📝 Generating summaries for {processed} uploaded files...")
            # uploaded_files already has correct format with doc_id, hash, etc.
            asyncio.create_task(_generate_summaries_background(course_id, uploaded_files, x_canvas_user_id))

        return {
            "success": True,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "total": len(files),
            "uploaded_files": uploaded_files + skipped_files  # Include both new and existing files for frontend
        }

    except Exception as e:
        print(f"❌ ERROR in process_canvas_files endpoint:")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/regenerate_summaries/{course_id}")
async def regenerate_missing_summaries(
    course_id: str,
    force: bool = False
):
    """
    Regenerate summaries for files that don't have them.

    Args:
        course_id: Course identifier
        force: If True, regenerate ALL summaries (not just missing ones)

    Returns:
        Status of summary regeneration task
    """
    try:
        if not document_manager:
            raise HTTPException(status_code=500, detail="Document manager not initialized")

        if not file_summarizer or not chat_storage:
            raise HTTPException(status_code=500, detail="File summarizer or chat storage not initialized")

        # Get all materials from catalog
        catalog = document_manager.get_material_catalog(course_id)
        all_materials = catalog.get("materials", [])

        if not all_materials:
            return {
                "success": True,
                "total_files": 0,
                "existing_summaries": 0,
                "missing_summaries": 0,
                "status": "no_files",
                "message": "No files found in catalog"
            }

        # Get all existing summaries
        existing_summaries = chat_storage.get_all_summaries_for_course(course_id)
        existing_doc_ids = {s["doc_id"] for s in existing_summaries}

        print(f"📊 Backfill check for course {course_id}:")
        print(f"   Total files in catalog: {len(all_materials)}")
        print(f"   Existing summaries: {len(existing_summaries)}")

        # Find missing summaries
        files_to_summarize = []
        for material in all_materials:
            doc_id = material["id"]

            # Skip if summary exists (unless force=True)
            if not force and doc_id in existing_doc_ids:
                continue

            # Add to batch for summarization
            files_to_summarize.append({
                "filename": material["filename"],
                "actual_filename": material["filename"],
                "path": material["path"],
                "status": "uploaded",
                "mime_type": "application/pdf"
            })

        if not files_to_summarize:
            return {
                "success": True,
                "total_files": len(all_materials),
                "existing_summaries": len(existing_summaries),
                "missing_summaries": 0,
                "status": "complete",
                "message": "All files already have summaries"
            }

        # Generate summaries in background
        print(f"📝 Backfilling {len(files_to_summarize)} missing summaries...")
        asyncio.create_task(_generate_summaries_background(course_id, files_to_summarize))

        return {
            "success": True,
            "total_files": len(all_materials),
            "existing_summaries": len(existing_summaries),
            "missing_summaries": len(files_to_summarize),
            "status": "generating",
            "message": f"Generating summaries for {len(files_to_summarize)} files in background"
        }

    except Exception as e:
        print(f"❌ ERROR in regenerate_summaries endpoint:")
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


@app.post("/chats/{course_id}/{session_id}/generate-title")
async def generate_chat_title(course_id: str, session_id: str, data: dict):
    """
    Generate AI title for chat based on first message

    Uses Gemini Flash to create a concise, descriptive title (3-6 words)
    that summarizes the user's question.

    Args:
        course_id: Course identifier (for validation)
        session_id: Session identifier
        data: Dict containing:
            - first_message: str (user's first question)

    Returns:
        JSON with success status and generated title
    """
    try:
        # Verify session exists and belongs to this course
        session = chat_storage.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")

        if session.get('course_id') != course_id:
            raise HTTPException(status_code=403, detail="Course ID mismatch")

        first_message = data.get('first_message', '')
        if not first_message:
            raise HTTPException(status_code=400, detail="first_message is required")

        # Generate title using Gemini
        try:
            from google import genai
            from google.genai import types

            # Use default API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise Exception("GOOGLE_API_KEY not configured")

            client = genai.Client(api_key=api_key)

            # Prompt for title generation
            prompt = f"""Generate a concise, descriptive title (max 32 characters) for this user question.
The title should capture the main topic or intent of the question.
Do not use quotes or punctuation in the title.
Keep it SHORT and specific.

User question: {first_message[:200]}

Title:"""

            # Call Gemini Flash-Lite (fast and cheap) - uses separate quota from main model
            response = client.models.generate_content(
                model='gemini-2.0-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=15
                )
            )

            generated_title = response.text.strip()

            # Enforce 32 character limit
            if len(generated_title) > 32:
                generated_title = generated_title[:29] + "..."

            # Fallback if generation fails
            if not generated_title:
                from datetime import datetime
                generated_title = f"Chat {datetime.now().strftime('%b %d')}"

            print(f"✨ Generated title: {generated_title} ({len(generated_title)} chars)")

        except Exception as e:
            print(f"⚠️ Title generation failed: {e}")
            # Fallback title
            from datetime import datetime
            generated_title = f"Chat {datetime.now().strftime('%b %d, %Y')}"

        # Update title in database
        success = chat_storage.update_chat_title(session_id, generated_title)

        return {
            "success": success,
            "title": generated_title,
            "message": "Title generated successfully" if success else "Failed to save title"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating chat title: {e}")
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

    # Create a task to handle incoming messages (including stop signals)
    async def handle_incoming_messages():
        """Background task that listens for stop signals"""
        try:
            while True:
                data = await websocket.receive_text()
                message_data = json.loads(data)

                if message_data.get("type") == "ping":
                    # Respond to ping with pong (keepalive heartbeat)
                    # This prevents Railway from timing out idle WebSocket connections (~55-60s)
                    print(f"💓 Received ping from {connection_id}")
                    try:
                        await websocket.send_json({"type": "pong"})
                        print(f"💓 Sent pong to {connection_id}")
                    except Exception as e:
                        print(f"⚠️ Failed to send pong to {connection_id} (connection closed): {e}")
                        break  # Exit message handler if connection is dead
                    continue  # Don't process ping as a regular message
                elif message_data.get("type") == "stop":
                    import time
                    print(f"🛑 Stop signal received for {connection_id} at {time.time()}")
                    active_connections[connection_id]["stop_streaming"] = True
                    print(f"🛑 Set stop_streaming flag to: {active_connections[connection_id]['stop_streaming']}")
                    try:
                        await websocket.send_json({
                            "type": "stopped",
                            "message": "Generation stopped by user"
                        })
                    except Exception as e:
                        print(f"⚠️ Failed to send stop confirmation (connection closed): {e}")
                else:
                    # Queue the message for processing
                    active_connections[connection_id]["pending_message"] = message_data
        except Exception as e:
            print(f"Message handler error: {e}")

    # Start the message handler task
    message_task = asyncio.create_task(handle_incoming_messages())

    try:
        while True:
            # Wait for a pending message
            while "pending_message" not in active_connections[connection_id]:
                await asyncio.sleep(0.01)

            message_data = active_connections[connection_id].pop("pending_message")

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

                # Check if connection is still open before sending
                try:
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk
                    })
                except Exception as send_error:
                    print(f"⚠️ Failed to send chunk (connection closed): {send_error}")
                    break  # Stop streaming if connection is dead

            # Send completion signal (only if not stopped)
            if not active_connections.get(connection_id, {}).get("stop_streaming", False):
                print(f"✅ Completed ({chunk_count} chunks)")
                try:
                    await websocket.send_json({
                        "type": "done"
                    })
                except Exception as send_error:
                    print(f"⚠️ Failed to send 'done' signal (connection closed): {send_error}")
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
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass  # Connection already closed
    finally:
        # Always cleanup resources
        print(f"🧹 Cleaning up connection: {connection_id}")

        # Cancel message handler task
        try:
            message_task.cancel()
            await message_task  # Wait for cancellation to complete
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"⚠️ Error cancelling message task: {e}")

        # Clear session cache
        try:
            root_agent.clear_session(connection_id)
        except Exception as e:
            print(f"⚠️ Error clearing session: {e}")

        # Remove from active connections
        if connection_id in active_connections:
            del active_connections[connection_id]

        print(f"✅ Cleanup complete for {connection_id}")


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


@app.get("/collections/{course_id}/materials")
async def get_course_materials(course_id: str):
    """
    Get full materials catalog with hash-based IDs for a course

    Args:
        course_id: Course identifier

    Returns:
        JSON with:
        - success: bool
        - course_id: str
        - materials: List[Dict] (full material metadata with hash-based IDs)

    Used by frontend to get hash-based doc_ids after uploading files.
    Each material includes: id (hash-based), name, filename, hash, type, size_mb, etc.
    """
    try:
        if not document_manager:
            raise HTTPException(status_code=500, detail="Document manager not initialized")

        catalog = document_manager.get_material_catalog(course_id)

        return {
            "success": True,
            "course_id": course_id,
            "total_documents": catalog.get("total_documents", 0),
            "materials": catalog.get("materials", [])
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/collections/{course_id}/refresh")
async def refresh_collection_catalog(course_id: str):
    """
    Refresh the in-memory catalog for a course by rescanning GCS

    This fixes stale catalog issues where files were deleted from GCS
    but still appear in the catalog.

    Args:
        course_id: Course identifier

    Returns:
        JSON with refreshed catalog status
    """
    try:
        if not document_manager or not storage_manager:
            raise HTTPException(status_code=503, detail="Services not available")

        print(f"\n{'='*80}")
        print(f"🔄 REFRESH CATALOG REQUEST:")
        print(f"   Course ID: {course_id}")
        print(f"{'='*80}")

        # Get current catalog count (before refresh)
        old_catalog = document_manager.get_material_catalog(course_id)
        old_count = old_catalog.get("total_documents", 0)

        # Clear the in-memory catalog for this course
        if course_id in document_manager.catalog:
            del document_manager.catalog[course_id]
            print(f"🗑️  Cleared in-memory catalog (had {old_count} entries)")

        # Rebuild catalog from GCS
        gcs_files = storage_manager.list_files(course_id=course_id)
        print(f"📂 Found {len(gcs_files)} files in GCS")

        if len(gcs_files) > 0:
            # Re-add files to catalog
            document_manager.add_materials_to_catalog(course_id, gcs_files)
            print(f"✅ Rebuilt catalog from GCS")

        # Get new catalog count
        new_catalog = document_manager.get_material_catalog(course_id)
        new_count = new_catalog.get("total_documents", 0)

        print(f"✅ Catalog refreshed: {old_count} → {new_count} documents")

        return {
            "success": True,
            "course_id": course_id,
            "old_count": old_count,
            "new_count": new_count,
            "files_removed": old_count - new_count,
            "message": f"Catalog refreshed: {new_count} documents from GCS"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error refreshing catalog: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/courses/{course_id}/cache-status")
async def get_course_cache_status(course_id: str):
    """
    Get Gemini cache status for a specific course

    Returns:
        JSON with cache statistics for the course:
        - files_cached: Number of files with valid cache
        - files_total: Total number of files in course
        - files_pending: Number of files not yet cached
        - cache_coverage_percent: Percentage of files cached
        - estimated_ready_time: Estimated seconds until all files cached
    """
    try:
        if not chat_storage:
            return {
                "success": False,
                "error": "Chat storage not available"
            }

        # Get all files for this course from catalog
        catalog = document_manager.get_material_catalog(course_id)
        total_files = catalog.get("total_documents", 0)
        all_file_paths = [mat['path'] for mat in catalog.get("materials", [])]

        # Get cached files for this course
        cached_files_by_course = chat_storage.get_all_cached_files_by_course()
        cached_paths = set(cached_files_by_course.get(course_id, []))

        files_cached = len(cached_paths)
        files_pending = total_files - files_cached
        cache_coverage = (files_cached / total_files * 100) if total_files > 0 else 0

        # Estimate time until all files are cached (2 seconds per file average)
        estimated_ready_time = files_pending * 2

        return {
            "success": True,
            "course_id": course_id,
            "files_cached": files_cached,
            "files_total": total_files,
            "files_pending": files_pending,
            "cache_coverage_percent": round(cache_coverage, 1),
            "estimated_ready_time_seconds": estimated_ready_time,
            "is_ready": files_pending == 0,
            "message": "All files cached and ready!" if files_pending == 0 else f"Caching {files_pending} files..."
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/courses/{course_id}/summary-status")
async def get_course_summary_status(course_id: str):
    """
    Get summary generation status for a specific course

    Returns:
        JSON with summary statistics for the course:
        - summaries_ready: Number of files with summaries
        - total_files: Total number of files in course
        - summaries_pending: Number of files without summaries
        - completion_percent: Percentage of summaries complete
        - is_ready: Whether all summaries are ready
        - estimated_time_seconds: Estimated seconds until all summaries ready
    """
    try:
        if not chat_storage:
            return {
                "success": False,
                "error": "Chat storage not available"
            }

        # Get all files for this course from catalog
        catalog = document_manager.get_material_catalog(course_id)
        total_files = catalog.get("total_documents", 0)

        # Count summaries for this course
        summaries_ready = chat_storage.count_summaries_for_course(course_id)
        summaries_pending = max(0, total_files - summaries_ready)
        completion_percent = (summaries_ready / total_files * 100) if total_files > 0 else 0

        print(f"📊 Summary status for {course_id}: {summaries_ready}/{total_files} ready ({completion_percent:.1f}%)")

        # Estimate time until all summaries are ready (3 seconds per summary average)
        # This accounts for: 3 concurrent + 3.0s delay = ~3s per file average
        estimated_ready_time = summaries_pending * 3

        # Get failed summary info from retry queue
        failed_items = failed_summary_queue.get(course_id, [])
        failed_count = len(failed_items)
        retry_queue_count = sum(1 for item in failed_items if item["attempts"] < 3)

        # Get error details for failed summaries
        error_details = []
        for item in failed_items[:5]:  # Limit to first 5 for response size
            error_details.append({
                "filename": item["filename"],
                "attempts": item["attempts"],
                "last_error": item["last_error"]
            })

        # Get removed files for this course (validation failures)
        removed_files_list = removed_files_tracker.get(course_id, [])

        return {
            "success": True,
            "course_id": course_id,
            "summaries_ready": summaries_ready,
            "total_files": total_files,
            "summaries_pending": summaries_pending,
            "completion_percent": round(completion_percent, 1),
            "is_ready": summaries_pending == 0 and failed_count == 0,
            "estimated_time_seconds": estimated_ready_time,
            "failed_count": failed_count,
            "retry_queue_count": retry_queue_count,
            "error_details": error_details,
            "removed_files": removed_files_list,  # Files removed due to validation failures
            "message": "All summaries ready!" if (summaries_pending == 0 and failed_count == 0) else
                      f"Generating {summaries_pending} summaries... ({failed_count} failed, {retry_queue_count} retrying)"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/courses/{course_id}/regenerate-failed-summaries")
async def regenerate_failed_summaries(course_id: str):
    """
    Manually trigger retry for all failed summaries in a course

    Returns:
        JSON with:
        - success: Whether the regeneration was triggered
        - files_retrying: List of files being retried
        - retry_count: Number of files being retried
    """
    try:
        if course_id not in failed_summary_queue or not failed_summary_queue[course_id]:
            return {
                "success": True,
                "message": "No failed summaries to retry",
                "retry_count": 0,
                "files_retrying": []
            }

        failed_items = failed_summary_queue[course_id]
        files_retrying = [{"filename": item["filename"], "attempts": item["attempts"]} for item in failed_items]

        # Reset attempts to give fresh retry budget
        for item in failed_items:
            item["attempts"] = 0

        print(f"🔄 Manual regeneration triggered for {len(failed_items)} failed summaries in course {course_id}")

        # Trigger immediate retry in background
        asyncio.create_task(_retry_failed_summaries_once(course_id))

        return {
            "success": True,
            "message": f"Retrying {len(failed_items)} failed summaries",
            "retry_count": len(failed_items),
            "files_retrying": files_retrying
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def _retry_failed_summaries_once(course_id: str):
    """Helper to retry failed summaries immediately (called by manual regeneration)"""
    try:
        if course_id not in failed_summary_queue:
            return

        failed_items = failed_summary_queue[course_id]
        print(f"   Retrying {len(failed_items)} summaries for course {course_id}")

        from utils.file_upload_manager import FileUploadManager
        api_key = os.getenv("GOOGLE_API_KEY")
        file_upload_client = genai.Client(api_key=api_key)
        file_uploader = FileUploadManager(
            file_upload_client,
            cache_duration_hours=48,
            storage_manager=storage_manager,
            chat_storage=chat_storage
        )

        for item in failed_items[:]:  # Copy to allow modification
            try:
                result = await _generate_single_summary(
                    item["file_info"], course_id, file_uploader, file_summarizer, chat_storage, None,
                    document_manager=document_manager, storage_manager=storage_manager
                )

                if result.get("status") in ["success", "cached"]:
                    print(f"   ✅ Retry successful for {item['filename']}")
                    failed_items.remove(item)
                elif result.get("status") == "removed":
                    # File was removed due to validation failure - don't retry
                    print(f"   🗑️  Removed invalid file from queue: {item['filename']}")
                    failed_items.remove(item)
                else:
                    print(f"   ⚠️  Retry failed for {item['filename']}: {result.get('error', 'Unknown')}")
                    item["last_error"] = result.get('error', 'Unknown')[:200]
                    item["attempts"] += 1

            except Exception as e:
                print(f"   ❌ Retry exception for {item['filename']}: {e}")
                item["last_error"] = str(e)[:200]
                item["attempts"] += 1

        # Clean up if no items left
        if not failed_items:
            del failed_summary_queue[course_id]

    except Exception as e:
        print(f"❌ Error in manual retry: {e}")
        import traceback
        traceback.print_exc()


@app.get("/cache-stats")
async def get_global_cache_stats():
    """
    Get global Gemini cache statistics across all courses

    Returns:
        JSON with overall cache statistics:
        - total_files: Total files cached
        - total_bytes: Total bytes cached
        - total_mb: Total MB cached
        - courses_count: Number of courses with cached files
        - expiring_soon_count: Files expiring within 6 hours
        - expired_count: Expired cache entries (need cleanup)
    """
    try:
        if not chat_storage:
            return {
                "success": False,
                "error": "Chat storage not available"
            }

        stats = chat_storage.get_cache_stats()

        return {
            "success": True,
            "total_files": stats.get('total_files', 0),
            "total_bytes": stats.get('total_bytes', 0),
            "total_mb": round(stats.get('total_bytes', 0) / (1024 * 1024), 2),
            "courses_count": stats.get('courses_count', 0),
            "expiring_soon_count": stats.get('expiring_soon_count', 0),
            "expired_count": stats.get('expired_count', 0),
            "cache_health": "excellent" if stats.get('expiring_soon_count', 0) == 0 else "good"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.delete("/courses/{course_id}/materials/{file_id}")
async def soft_delete_material(course_id: str, file_id: str, permanent: bool = False):
    """
    Soft delete or permanently delete a course material file

    Args:
        course_id: Course identifier
        file_id: File/document identifier (doc_id from file_summaries)
        permanent: If True, permanently delete from GCS and database

    Returns:
        Success confirmation or error
    """
    try:
        if not chat_storage:
            raise HTTPException(status_code=500, detail="Chat storage not initialized")

        if permanent:
            # Hard delete: Remove from GCS and database
            print(f"🗑️  Permanently deleting file {file_id} from course {course_id}")

            # Get file info before deletion
            file_info = chat_storage.get_file_summary(file_id)
            if not file_info:
                raise HTTPException(status_code=404, detail="File not found")

            # Delete from GCS if storage_manager available
            if storage_manager:
                try:
                    # Extract filename from metadata or use filename field
                    metadata = file_info.get('metadata', {})
                    gcs_path = f"{course_id}/{file_info['filename']}"
                    storage_manager.delete_file(gcs_path)
                    print(f"✅ Deleted from GCS: {gcs_path}")
                except Exception as gcs_error:
                    print(f"⚠️  Failed to delete from GCS: {gcs_error}")

            # Hard delete from database
            success = chat_storage.hard_delete_file(file_id)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to delete file from database")

            return {
                "success": True,
                "message": "File permanently deleted",
                "file_id": file_id,
                "permanent": True
            }
        else:
            # Soft delete: Set deleted_at timestamp
            print(f"🗑️  Soft deleting file {file_id} from course {course_id}")
            success = chat_storage.soft_delete_file(file_id)

            if not success:
                raise HTTPException(status_code=500, detail="Failed to soft delete file")

            return {
                "success": True,
                "message": "File removed from AI memory (can be restored from Settings)",
                "file_id": file_id,
                "permanent": False
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting material: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting material: {str(e)}")


@app.get("/courses/{course_id}/deleted-materials")
async def get_deleted_materials(course_id: str):
    """
    Get all soft-deleted materials for a course

    Args:
        course_id: Course identifier

    Returns:
        List of deleted files with metadata
    """
    try:
        if not chat_storage:
            raise HTTPException(status_code=500, detail="Chat storage not initialized")

        deleted_files = chat_storage.get_deleted_files(course_id)

        return {
            "success": True,
            "course_id": course_id,
            "deleted_files": deleted_files,
            "count": len(deleted_files)
        }

    except Exception as e:
        print(f"❌ Error fetching deleted materials: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching deleted materials: {str(e)}")


@app.put("/courses/{course_id}/materials/{file_id}/restore")
async def restore_material(course_id: str, file_id: str):
    """
    Restore a soft-deleted material

    Args:
        course_id: Course identifier
        file_id: File/document identifier

    Returns:
        Success confirmation or error
    """
    try:
        if not chat_storage:
            raise HTTPException(status_code=500, detail="Chat storage not initialized")

        print(f"♻️  Restoring file {file_id} for course {course_id}")
        success = chat_storage.restore_file(file_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to restore file")

        return {
            "success": True,
            "message": "File restored successfully",
            "file_id": file_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error restoring material: {e}")
        raise HTTPException(status_code=500, detail=f"Error restoring material: {str(e)}")


@app.delete("/users/{canvas_user_id}/data")
async def delete_user_data(canvas_user_id: str):
    """
    Delete all data for a specific Canvas user.

    This permanently deletes:
    - All chat sessions and messages
    - All uploaded files from Google Cloud Storage
    - All file summaries from the database
    - All Gemini cache entries

    Args:
        canvas_user_id: Canvas user ID whose data should be deleted

    Returns:
        Deletion statistics
    """
    try:
        if not chat_storage:
            raise HTTPException(status_code=500, detail="Chat storage not initialized")

        print(f"\n{'='*80}")
        print(f"🗑️  DELETE USER DATA REQUEST")
        print(f"   Canvas User ID: {canvas_user_id}")
        print(f"{'='*80}")

        # Step 1: Get list of files to delete from GCS before deleting from database
        file_paths = chat_storage.get_user_file_paths(canvas_user_id)
        print(f"📁 Found {len(file_paths)} files to delete from GCS")

        # Step 2: Get Gemini cache entries (for potential future cleanup)
        gemini_entries = chat_storage.get_user_gemini_cache_entries(canvas_user_id)
        print(f"🔥 Found {len(gemini_entries)} Gemini cache entries")

        # Step 3: Delete from GCS
        gcs_deleted = 0
        gcs_errors = []
        if storage_manager and file_paths:
            for file_path in file_paths:
                try:
                    storage_manager.delete_file(file_path)
                    gcs_deleted += 1
                except Exception as e:
                    print(f"⚠️  Failed to delete GCS file {file_path}: {e}")
                    gcs_errors.append({"path": file_path, "error": str(e)})

        print(f"✅ Deleted {gcs_deleted}/{len(file_paths)} files from GCS")

        # Step 4: Delete from database (chat sessions, messages, file summaries, gemini cache)
        db_stats = chat_storage.delete_user_data(canvas_user_id)

        print(f"✅ Database cleanup complete:")
        print(f"   - Chat sessions deleted: {db_stats['chat_sessions_deleted']}")
        print(f"   - Chat messages deleted: {db_stats['chat_messages_deleted']}")
        print(f"   - File summaries deleted: {db_stats['file_summaries_deleted']}")
        print(f"   - Gemini cache entries deleted: {db_stats['gemini_cache_deleted']}")

        # Step 5: Clear in-memory caches
        if document_manager:
            # Clear document catalog entries for this user's files
            # Note: This clears all courses for simplicity - catalog will rebuild on next access
            document_manager.catalog.clear()
            print(f"✅ In-memory document catalog cleared")

        result = {
            "success": True,
            "canvas_user_id": canvas_user_id,
            "message": "All user data has been permanently deleted",
            "statistics": {
                "gcs_files_deleted": gcs_deleted,
                "gcs_files_attempted": len(file_paths),
                "gcs_errors": len(gcs_errors),
                "chat_sessions_deleted": db_stats['chat_sessions_deleted'],
                "chat_messages_deleted": db_stats['chat_messages_deleted'],
                "file_summaries_deleted": db_stats['file_summaries_deleted'],
                "gemini_cache_deleted": db_stats['gemini_cache_deleted']
            }
        }

        if gcs_errors:
            result["gcs_errors"] = gcs_errors

        print(f"\n✅ USER DATA DELETION COMPLETE for {canvas_user_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting user data: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting user data: {str(e)}")


@app.post("/courses/{course_id}/update_canvas_ids")
async def update_canvas_ids(course_id: str, updates: Dict):
    """
    Update canvas_id for existing files in the database

    Request format:
    {
        "updates": [
            {"doc_id": "1425905_abc123...", "canvas_id": "15061355"},
            ...
        ]
    }

    Returns:
        {
            "success": True,
            "updated_count": int
        }
    """
    try:
        if not chat_storage:
            raise HTTPException(status_code=500, detail="Chat storage not initialized")

        updates_list = updates.get("updates", [])
        if not updates_list:
            raise HTTPException(status_code=400, detail="No updates provided")

        print(f"🔄 Updating canvas_ids for {len(updates_list)} files in course {course_id}")

        updated_count = 0
        for update in updates_list:
            doc_id = update.get("doc_id")
            canvas_id = update.get("canvas_id")

            if not doc_id:
                continue

            # Get existing file summary
            file_summary = chat_storage.get_file_summary(doc_id)
            if file_summary:
                # Update with canvas_id
                chat_storage.save_file_summary(
                    doc_id=doc_id,
                    course_id=course_id,
                    filename=file_summary["filename"],
                    summary=file_summary["summary"],
                    topics=file_summary.get("topics"),
                    metadata=file_summary.get("metadata"),
                    content_hash=file_summary.get("content_hash"),
                    canvas_id=canvas_id
                )
                updated_count += 1
                print(f"   ✅ Updated {doc_id[:24]}... with canvas_id: {canvas_id}")

        print(f"✅ Updated {updated_count} files with canvas_ids")

        return {
            "success": True,
            "updated_count": updated_count
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating canvas_ids: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating canvas_ids: {str(e)}")


@app.post("/admin/cleanup_bad_filenames/{course_id}")
async def cleanup_bad_filenames(course_id: str, dry_run: bool = True, remove_duplicates: bool = True):
    """
    Clean up files with problematic filenames and duplicates

    Args:
        course_id: Course identifier
        dry_run: If True, only report what would be deleted (default: True)
        remove_duplicates: If True, also remove duplicate files (default: True)

    Returns:
        {
            "dry_run": bool,
            "bad_files_found": int,
            "duplicates_found": int,
            "deleted_count": int,
            "deleted_files": [...],
            "errors": [...]
        }
    """
    try:
        if not storage_manager:
            raise HTTPException(status_code=503, detail="Storage service not available")

        print(f"\n{'='*80}")
        print(f"🧹 CLEANUP BAD FILENAMES & DUPLICATES REQUEST:")
        print(f"   Course ID: {course_id}")
        print(f"   Dry Run: {dry_run}")
        print(f"   Remove Duplicates: {remove_duplicates}")
        print(f"{'='*80}")

        # Get all files for the course
        all_files = storage_manager.list_files(course_id=course_id)
        print(f"📂 Found {len(all_files)} total files in GCS")

        # Find files with "/" in the filename part (after course_id/)
        bad_files = []
        for blob_name in all_files:
            # Split on first "/" to separate course_id from filename
            parts = blob_name.split('/', 1)
            if len(parts) == 2:
                filename = parts[1]
                # Check if filename itself contains "/" (indicates bad filename)
                if '/' in filename:
                    bad_files.append(blob_name)
                    print(f"   ⚠️  Bad file found (contains '/'): {blob_name}")

        print(f"🔍 Found {len(bad_files)} files with problematic names")

        # Find duplicates (keep newest, delete older versions)
        duplicates_to_delete = []
        if remove_duplicates:
            from collections import defaultdict
            files_by_name = defaultdict(list)

            # Group files by filename
            for blob_name in all_files:
                parts = blob_name.split('/', 1)
                if len(parts) == 2:
                    filename = parts[1]
                    files_by_name[filename].append(blob_name)

            # Find duplicates (same filename appears multiple times)
            for filename, blob_names in files_by_name.items():
                if len(blob_names) > 1:
                    # Get metadata for each file to determine which is newest
                    file_metadata = []
                    for blob_name in blob_names:
                        try:
                            blob = storage_manager.bucket.blob(blob_name)
                            blob.reload()
                            file_metadata.append({
                                'blob_name': blob_name,
                                'updated': blob.updated,
                                'size': blob.size
                            })
                        except Exception as e:
                            print(f"   ⚠️  Error getting metadata for {blob_name}: {e}")

                    if len(file_metadata) > 1:
                        # Sort by updated time (newest first)
                        file_metadata.sort(key=lambda x: x['updated'], reverse=True)

                        # Keep the newest, mark others for deletion
                        newest = file_metadata[0]
                        for old_file in file_metadata[1:]:
                            duplicates_to_delete.append(old_file['blob_name'])
                            print(f"   🔄 Duplicate found: {old_file['blob_name']}")
                            print(f"      Keeping newer version from {newest['updated']}")

            print(f"🔍 Found {len(duplicates_to_delete)} duplicate files to remove")

        # Combine bad files and duplicates
        files_to_delete = list(set(bad_files + duplicates_to_delete))
        print(f"🗑️  Total files to delete: {len(files_to_delete)}")

        if len(files_to_delete) == 0:
            return {
                "dry_run": dry_run,
                "bad_files_found": 0,
                "duplicates_found": 0,
                "deleted_count": 0,
                "deleted_files": [],
                "errors": [],
                "message": "No bad filenames or duplicates found! All files are clean."
            }

        if dry_run:
            print(f"🔍 DRY RUN - No files will be deleted")
            print(f"   Would delete {len(files_to_delete)} files:")
            print(f"   - Bad filenames (with '/'): {len(bad_files)}")
            print(f"   - Duplicates: {len(duplicates_to_delete)}")
            for blob_name in files_to_delete:
                reason = "bad filename" if blob_name in bad_files else "duplicate"
                print(f"     - {blob_name} ({reason})")
            return {
                "dry_run": True,
                "bad_files_found": len(bad_files),
                "duplicates_found": len(duplicates_to_delete),
                "deleted_count": 0,
                "deleted_files": files_to_delete,
                "errors": [],
                "message": f"Dry run complete. Found {len(bad_files)} bad filenames and {len(duplicates_to_delete)} duplicates to delete."
            }

        # Actually delete the files
        deleted_files = []
        errors = []

        for blob_name in files_to_delete:
            try:
                # Delete from GCS
                success = storage_manager.delete_file(blob_name)

                if success:
                    print(f"   ✅ Deleted from GCS: {blob_name}")

                    # Delete from Gemini cache if exists
                    if chat_storage:
                        cache_deleted = chat_storage.delete_gemini_cache_entry(blob_name)
                        if cache_deleted:
                            print(f"   ✅ Deleted from Gemini cache: {blob_name}")

                    deleted_files.append(blob_name)
                else:
                    errors.append({
                        "file": blob_name,
                        "error": "File not found or delete failed"
                    })
                    print(f"   ❌ Failed to delete: {blob_name}")

            except Exception as e:
                errors.append({
                    "file": blob_name,
                    "error": str(e)
                })
                print(f"   ❌ Error deleting {blob_name}: {e}")

        print(f"✅ Cleanup complete: {len(deleted_files)} deleted, {len(errors)} errors")

        return {
            "dry_run": False,
            "bad_files_found": len(bad_files),
            "duplicates_found": len(duplicates_to_delete),
            "deleted_count": len(deleted_files),
            "deleted_files": deleted_files,
            "errors": errors,
            "message": f"Successfully deleted {len(deleted_files)} files ({len(bad_files)} bad filenames, {len(duplicates_to_delete)} duplicates)."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in cleanup: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/cleanup_duplicate_summaries/{course_id}")
async def cleanup_duplicate_summaries(course_id: str, dry_run: bool = True):
    """
    Clean up duplicate file summary entries in the catalog

    Finds files with same course_id + filename and keeps only the newest entry.
    This fixes duplicate entries in the document catalog/AI memory.

    Args:
        course_id: Course identifier
        dry_run: If True, only report duplicates without deleting (default: True)

    Returns:
        {
            "dry_run": bool,
            "duplicates_found": int,
            "deleted_count": int,
            "deleted_entries": [...],
            "errors": [...]
        }
    """
    try:
        if not chat_storage:
            raise HTTPException(status_code=503, detail="Chat storage not available")

        print(f"\n{'='*80}")
        print(f"🧹 CLEANUP DUPLICATE FILE SUMMARIES REQUEST:")
        print(f"   Course ID: {course_id}")
        print(f"   Dry Run: {dry_run}")
        print(f"{'='*80}")

        result = chat_storage.cleanup_duplicate_file_summaries(course_id, dry_run=dry_run)

        duplicates_found = result.get('duplicates_found', 0)
        deleted_count = result.get('deleted_count', 0)

        if duplicates_found == 0:
            message = "No duplicate file summaries found! Catalog is clean."
        elif dry_run:
            message = f"Dry run complete. Found {duplicates_found} duplicate entries to remove."
        else:
            message = f"Successfully removed {deleted_count} duplicate entries from catalog."

        print(f"✅ {message}")

        return {
            "dry_run": dry_run,
            "duplicates_found": duplicates_found,
            "deleted_count": deleted_count,
            "deleted_entries": result.get('deleted_entries', []),
            "errors": result.get('errors', []),
            "message": message
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in cleanup duplicate summaries: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdfs/{course_id}/{doc_id:path}")
async def serve_pdf(course_id: str, doc_id: str, page: Optional[int] = None):
    """
    Serve files from GCS using hash-based doc_id

    TODO: Update frontend to use hash-based doc_ids in URLs
    Currently expects doc_id format: {course_id}_{hash}
    But path parameter receives just the hash portion

    Args:
        course_id: Course identifier
        doc_id: Document hash or full doc_id

    Returns:
        Redirect to GCS signed URL

    Example:
        GET /pdfs/12345/abc123def456...#page=5
        Opens file with hash abc123def456... at page 5
    """
    try:
        # Decode URL-encoded doc_id
        doc_id = unquote(doc_id)

        # Try to serve from GCS
        if storage_manager:
            try:
                # HASH-BASED: doc_id should be the content hash
                # GCS blob name format: {course_id}/{hash}.pdf
                blob_name = f"{course_id}/{doc_id}.pdf"

                # Check if file exists
                file_exists = storage_manager.file_exists(blob_name)

                if not file_exists:
                    logger.warning(f"File not found in GCS: {blob_name}")
                    raise HTTPException(status_code=404, detail=f"File not found: {doc_id}")

                # Generate signed URL that's valid for 1 hour
                signed_url = storage_manager.get_signed_url(blob_name, expiration_minutes=60)

                if not signed_url:
                    raise HTTPException(status_code=500, detail="Failed to generate signed URL")

                # Append page anchor if provided
                if page:
                    signed_url = f"{signed_url}#page={page}"

                # Redirect to GCS signed URL for browser to open
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=signed_url, status_code=302)

            except HTTPException:
                raise
            except Exception as gcs_error:
                logger.error(f"GCS error serving file: {gcs_error}")
                raise HTTPException(status_code=500, detail=f"GCS error: {str(gcs_error)}")

        # No storage manager available
        raise HTTPException(status_code=503, detail="Storage service not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in serve_pdf: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )
