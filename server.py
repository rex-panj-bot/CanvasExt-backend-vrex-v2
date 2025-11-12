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


def predict_stored_filename(original_filename: str) -> List[str]:
    """
    Predict ALL possible filenames that could be in GCS after sanitization and conversion.

    This is critical because Canvas files may not have extensions, and we detect them
    from Content-Type during download. We need to check for ALL possibilities.

    Args:
        original_filename: Original filename from Canvas (e.g., "Lecture 1/2.pptx" or "Darwin PPT")

    Returns:
        List of possible GCS filenames to check (ordered by likelihood)

    Examples:
        "Lecture 1/2.pptx" → ["Lecture 1-2.pdf", "Lecture 1-2.pptx"]
        "Darwin PPT" → ["Darwin PPT", "Darwin PPT.pdf", "Darwin PPT.pptx", "Darwin PPT.docx", ...]
        "Assignment.pdf" → ["Assignment.pdf"]
    """
    # Step 1: Sanitize filename (replace / with -)
    sanitized = original_filename.replace('/', '-')

    # Step 2: Check if it already has an extension
    if '.' in sanitized:
        # Has extension - predict the converted name
        if sanitized.endswith(('.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls', '.rtf')):
            # Office files get converted to PDF
            base = sanitized.rsplit('.', 1)[0]
            return [base + '.pdf', sanitized]  # Try converted first, then original

        if sanitized.endswith(('.html', '.htm', '.xml', '.json')):
            # Web formats get converted to .txt
            base = sanitized.rsplit('.', 1)[0]
            return [base + '.txt', sanitized]

        # Everything else stays as-is (PDFs, images, text files)
        return [sanitized]

    # Step 3: No extension - could be anything!
    # Try common possibilities (ordered by likelihood in course materials)
    return [
        sanitized,  # Might not have extension even in GCS
        f"{sanitized}.pdf",  # Most common after conversion
        f"{sanitized}.pptx",  # Office files
        f"{sanitized}.docx",
        f"{sanitized}.xlsx",
        f"{sanitized}.txt",  # Text files
    ]


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

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# In-memory cache for filename mappings: (course_id, original_filename) -> actual_gcs_filename
# This eliminates the need to guess/check multiple filenames when serving files
# Populated during check_files_exist and uploads
filename_cache: Dict[tuple, str] = {}

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
        # Use actual_filename (post-conversion) to match catalog IDs
        # This ensures doc_id format matches what document_manager uses
        filename = file_info.get("actual_filename") or file_info["filename"]
        file_path = file_info["path"]

        # Create doc_id to match document_manager format (includes extension)
        doc_id = f"{course_id}_{filename}"

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

        # PHASE 4: Priority upload queue - Higher concurrency for new uploads (20 vs 10)
        # This makes new study bots ready faster
        semaphore = asyncio.Semaphore(20)

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

        print(f"🔥 Pre-warming Gemini cache for {len(successful_uploads)} files (priority queue, 20 concurrent)...")
        tasks = [upload_single_file(file_info) for file_info in successful_uploads]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        print(f"✅ Gemini pre-warm complete: {success_count}/{len(successful_uploads)} files cached (ready for fast queries!)")

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

        # Semaphore to limit concurrent processing (max 50 at a time)
        # Summaries are I/O bound (Gemini API calls), can handle higher concurrency
        semaphore = asyncio.Semaphore(50)

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


async def _process_uploads_background(course_id: str, files_in_memory: List[Dict]):
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
        print(f"\n🔄 BACKGROUND PROCESSING STARTED: {len(files_in_memory)} files for course {course_id}")

        # Create UploadFile-like objects from memory
        from fastapi import UploadFile
        from io import BytesIO

        upload_files = []
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
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(upload_files) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"   Batch {batch_num}/{total_batches}: Processing {len(batch)} files...")

            upload_tasks = [_process_single_upload(course_id, file) for file in batch]
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

        # Count results
        successful = [r for r in processed_results if r["status"] == "uploaded"]
        failed = [r for r in processed_results if r["status"] == "failed"]
        skipped = [r for r in processed_results if r["status"] == "skipped"]

        print(f"✅ Background upload complete: {len(successful)} succeeded, {len(failed)} failed, {len(skipped)} skipped")

        # PHASE 2: Update catalog
        if document_manager and successful:
            print(f"📚 Adding {len(successful)} files to catalog...")
            new_paths = [r["path"] for r in successful]
            document_manager.add_files_to_catalog(new_paths)
            print(f"✅ Catalog updated")

        # PHASE 3: Pre-warm Gemini cache (background within background!)
        if successful and chat_storage:
            print(f"🔥 Starting Gemini pre-warm for {len(successful)} files...")
            asyncio.create_task(_upload_to_gemini_background(course_id, successful))

        # PHASE 4: Generate summaries
        if file_summarizer and chat_storage and successful:
            print(f"📝 Generating summaries for {len(successful)} files...")
            asyncio.create_task(_generate_summaries_background(course_id, successful))

        print(f"✅ BACKGROUND PROCESSING COMPLETE for course {course_id}")
        print(f"   Files are now available for queries!")

    except Exception as e:
        print(f"❌ Critical error in background upload processing: {e}")
        import traceback
        traceback.print_exc()


@app.post("/upload_pdfs")
async def upload_pdfs(
    course_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload files for a course (supports PDFs, documents, images, etc.)

    INSTANT RESPONSE MODE: Returns immediately after accepting files,
    processes everything in background for instant study bot creation.

    Args:
        course_id: Course identifier (used as filename prefix)
        files: List of files to upload (PDF, DOCX, TXT, images, etc.)

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
        print(f"   Number of files: {len(files)}")
        print(f"   File names received:")
        for f in files[:10]:  # Show first 10
            print(f"      - \"{f.filename}\"")
        print(f"   Storage manager available: {storage_manager is not None}")
        print(f"{'='*80}")

        # Read all files into memory IMMEDIATELY (before background processing)
        # This is fast (<100ms) and allows us to return response instantly
        files_in_memory = []
        for file in files:
            content = await file.read()
            files_in_memory.append({
                'filename': file.filename,
                'content': content,
                'content_type': file.content_type
            })

        print(f"✅ Files accepted ({len(files_in_memory)} files, {sum(len(f['content']) for f in files_in_memory) / 1024 / 1024:.1f} MB)")

        # Start background processing (non-blocking)
        asyncio.create_task(_process_uploads_background(course_id, files_in_memory))

        # INSTANT RESPONSE - User sees chat immediately!
        return {
            "success": True,
            "message": f"Processing {len(files)} files in background",
            "files": [{"filename": f['filename'], "status": "processing"} for f in files_in_memory],
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
    Check which files exist in GCS and return signed URLs for existing files

    PHASE 2 OPTIMIZATION: Allows frontend to skip downloading from Canvas
    if files already exist in GCS (much faster to download from GCS)

    Args:
        request: {
            "course_id": "123456",
            "files": [{"name": "file.pdf", "url": "..."}, ...]
        }

    Returns:
        {
            "exists": [{"name": "file.pdf", "url": "https://...", "size": 123}],
            "missing": ["other_file.pdf"]
        }

    Performance: Checks 100 files in ~500ms
    """
    try:
        course_id = request.get("course_id")
        files = request.get("files", [])

        if not course_id or not files:
            raise HTTPException(status_code=400, detail="course_id and files required")

        print(f"📋 Check files: {len(files)} files for course {course_id}")

        if not storage_manager:
            # If no GCS, return all as missing
            return {
                "exists": [],
                "missing": [f["name"] for f in files]
            }

        # Get all files from GCS once (single API call)
        all_gcs_files = storage_manager.list_files(course_id=course_id)
        gcs_filenames = {blob.split('/', 1)[1] for blob in all_gcs_files if '/' in blob}

        # Build reverse mapping cache: actual_gcs_filename -> possible_original_names
        # This helps us populate the filename_cache for faster lookups later
        gcs_to_originals = {}
        for gcs_filename in gcs_filenames:
            # Reverse the sanitization: "Lecture 1-2.pdf" -> "Lecture 1/2.pdf"
            # Try to match it back to original Canvas names
            gcs_to_originals[gcs_filename] = gcs_filename

        exists = []
        missing = []

        for file_info in files:
            file_name = file_info.get("name")
            if not file_name:
                continue

            # Predict what the filename will be after sanitization and conversion
            possible_filenames = predict_stored_filename(file_name)

            # Check if any of the predicted filenames exist in GCS
            found_blob_name = None
            found_filename = None
            for predicted_filename in possible_filenames:
                if predicted_filename in gcs_filenames:
                    found_blob_name = f"{course_id}/{predicted_filename}"
                    found_filename = predicted_filename
                    filename_cache[(course_id, file_name)] = found_filename
                    break

            if found_blob_name:
                # File exists - generate signed URL (valid for 1 hour)
                signed_url = storage_manager.get_signed_url(found_blob_name, expiration_minutes=60)

                if signed_url:
                    exists.append({
                        "name": file_name,
                        "actual_name": found_blob_name.split('/')[-1],
                        "url": signed_url,
                        "from_gcs": True
                    })
                else:
                    missing.append(file_name)
            else:
                missing.append(file_name)

        print(f"✅ {len(exists)} in GCS, {len(missing)} missing")

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
async def process_canvas_files(request: Dict):
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

            if not file_name or not file_url:
                return {"status": "skipped", "reason": "missing name or url"}

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

                # Download file from Canvas
                headers = {}
                if canvas_cookies:
                    headers['Cookie'] = canvas_cookies

                async with session.get(download_url, headers=headers, allow_redirects=True) as response:
                    if response.status != 200:
                        print(f"❌ {file_name}: HTTP {response.status}")
                        failed += 1
                        return {"status": "failed", "error": f"HTTP {response.status}", "filename": file_name}

                    file_content = await response.read()

                    content_type_header = response.headers.get('Content-Type', '').split(';')[0].strip()

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

                if needs_conversion(safe_filename):
                    from utils.file_converter import convert_to_text
                    web_formats = ['html', 'htm', 'xml', 'json']
                    is_web_format = ext.lower() in web_formats

                    if is_web_format:
                        text_bytes = convert_to_text(file_content, safe_filename)
                        if text_bytes:
                            file_content = text_bytes
                            actual_filename = safe_filename.rsplit('.', 1)[0] + '.txt'
                            mime_type = 'text/plain'
                    else:
                        pdf_bytes = convert_office_to_pdf(file_content, safe_filename)
                        if pdf_bytes:
                            file_content = pdf_bytes
                            actual_filename = safe_filename.rsplit('.', 1)[0] + '.pdf'
                            mime_type = 'application/pdf'

                if storage_manager:
                    blob_name = storage_manager.upload_pdf(
                        course_id,
                        actual_filename,
                        file_content,
                        mime_type
                    )
                    processed += 1
                    print(f"✅ {actual_filename}")

                    # Cache filename mapping for instant lookup later
                    original_canvas_name = file_info.get("name")
                    filename_cache[(course_id, original_canvas_name)] = actual_filename

                    # Return both original and actual filenames for frontend mapping
                    # Frontend needs this to update stored_name in materials
                    return {
                        "status": "uploaded",
                        "original_name": original_canvas_name,
                        "filename": actual_filename,
                        "path": blob_name
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
        semaphore = asyncio.Semaphore(8)

        async def process_with_semaphore(file_info, session):
            async with semaphore:
                return await process_single_file(file_info, session)

        # Create single aiohttp session for all requests
        async with aiohttp.ClientSession() as session:
            # Create tasks for all files
            tasks = [process_with_semaphore(file_info, session) for file_info in files]

            # Execute all tasks in parallel (limited to 8 concurrent)
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count results and collect file mappings
        uploaded_files = []
        for result in results:
            if isinstance(result, Exception):
                failed += 1
            elif isinstance(result, dict):
                if result.get("status") == "uploaded":
                    uploaded_files.append({
                        "original_name": result.get("original_name"),
                        "stored_name": result.get("filename"),
                        "path": result.get("path")
                    })

        print(f"✅ Complete: {processed} uploaded, {skipped} skipped, {failed} failed")

        # Add newly uploaded files to catalog (incremental, not full refresh)
        if processed > 0 and document_manager:
            try:
                print(f"📚 Adding {len(uploaded_files)} files to catalog...")
                new_paths = [upload_result["path"] for upload_result in uploaded_files]
                document_manager.add_files_to_catalog(new_paths)
                print(f"✅ Catalog updated")
            except Exception as e:
                print(f"⚠️ Catalog update failed: {e}")

        # Generate summaries for uploaded files (same as /upload_pdfs)
        if file_summarizer and chat_storage and processed > 0:
            print(f"📝 Generating summaries for {processed} uploaded files...")
            # Transform uploaded_files into format expected by _generate_summaries_background
            summary_files = []
            for upload_result in uploaded_files:
                summary_files.append({
                    "filename": upload_result.get("stored_name"),  # Use stored_name (post-conversion)
                    "actual_filename": upload_result.get("stored_name"),
                    "path": upload_result.get("path"),
                    "status": "uploaded",
                    "mime_type": "application/pdf"  # All files converted to PDF
                })
            asyncio.create_task(_generate_summaries_background(course_id, summary_files))

        return {
            "success": True,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "total": len(files),
            "uploaded_files": uploaded_files  # Include file mappings for frontend
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

            # Call Gemini Flash (fast and cheap)
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
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


@app.get("/pdfs/{course_id}/{filename:path}")
async def serve_pdf(course_id: str, filename: str, page: Optional[int] = None):
    """
    Serve files from GCS for viewing in browser

    Args:
        course_id: Course identifier
        filename: Filename (original name from Canvas)

    Returns:
        Redirect to GCS signed URL or local file

    Example:
        GET /pdfs/12345/Lecture_Notes.pdf#page=5
        Opens Lecture_Notes.pdf at page 5
    """
    try:
        # Decode URL-encoded filename
        filename = unquote(filename)

        print(f"📄 Serving file: {filename}")

        # Try to serve from GCS first
        if storage_manager:
            try:
                # OPTIMIZATION: Check cache first for instant lookup (no guessing needed)
                cache_key = (course_id, filename)
                cached_filename = filename_cache.get(cache_key)

                if cached_filename:
                    print(f"   ⚡ Cache hit! '{filename}' → '{cached_filename}'")
                    blob_name = f"{course_id}/{cached_filename}"
                    found_filename = cached_filename
                else:
                    # Cache miss - need to guess possible filenames
                    print(f"   ⚠️  Cache miss for '{filename}', checking GCS...")

                    # Predict the actual stored filename (sanitized/converted)
                    # Frontend may send original Canvas name, but GCS has sanitized name
                    # Example: "Lecture 1/2.pptx" → "Lecture 1-2.pdf"
                    # For files without extensions, try multiple possibilities
                    possible_filenames = predict_stored_filename(filename)

                    if len(possible_filenames) > 1:
                        print(f"   🔮 Trying {len(possible_filenames)} possibilities: {possible_filenames}")

                    # Try each possible filename until we find one that exists
                    blob_name = None
                    found_filename = None
                    for predicted_filename in possible_filenames:
                        test_blob_name = f"{course_id}/{predicted_filename}"
                        if storage_manager.file_exists(test_blob_name):
                            blob_name = test_blob_name
                            found_filename = predicted_filename
                            print(f"   ✅ Found in GCS: {predicted_filename}")

                            # Cache this mapping for next time
                            filename_cache[cache_key] = found_filename
                            break

                    if not blob_name:
                        print(f"   ❌ File not found in GCS. Tried: {possible_filenames}")
                        raise HTTPException(status_code=404, detail=f"File not found in GCS: {filename}")

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
                print(f"⚠️  GCS error: {gcs_error}")
                raise HTTPException(status_code=500, detail=f"GCS error: {str(gcs_error)}")

        # No storage manager available
        raise HTTPException(status_code=503, detail="Storage service not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )
