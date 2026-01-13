"""
File Upload Manager for Gemini File API
Handles file uploads to Gemini (PDFs, documents, images, etc.)
"""

import os
from google import genai
from pathlib import Path
from typing import Dict, Optional, List
import time
import asyncio
from .mime_types import get_mime_type, get_file_extension, is_supported_by_gemini
from .file_converter import convert_office_to_pdf, needs_conversion
from .file_validator import FileValidator

# Production mode - suppress verbose logging
PRODUCTION_MODE = os.getenv('PRODUCTION', 'true').lower() == 'true'

def debug_debug_print(*args, **kwargs):
    """Print only in development mode"""
    if not PRODUCTION_MODE:
        debug_print(*args, **kwargs)


class FileUploadManager:
    def __init__(self, client: genai.Client, cache_duration_hours: int = 48, storage_manager=None, chat_storage=None):
        """
        Initialize File Upload Manager

        Args:
            client: Gemini client instance
            cache_duration_hours: How long to cache file URIs (default: 48 hours, max for File API)
            storage_manager: Optional StorageManager for GCS file access
            chat_storage: Optional ChatStorage for database-backed URI caching (PHASE 3)
        """
        self.client = client
        self.cache_duration_hours = cache_duration_hours
        self.storage_manager = storage_manager
        self.chat_storage = chat_storage

        # In-memory cache: {file_path: {uri, upload_time, file_object}}
        # Note: Database cache (chat_storage) takes precedence if available
        self._file_cache = {}

    def upload_pdf(self, file_path: str, display_name: Optional[str] = None, mime_type: Optional[str] = None, canvas_user_id: Optional[str] = None) -> Dict:
        """
        Upload a file to Gemini File API (supports PDFs, documents, images, etc.)

        Args:
            file_path: Path to file (local path or GCS blob name)
            display_name: Optional display name for the file
            mime_type: Optional MIME type (auto-detected from filename if not provided)
            canvas_user_id: Optional Canvas user ID for ownership tracking

        Returns:
            Dict with file object and metadata
        """
        # Extract filename for logging
        filename = file_path.split('/')[-1] if '/' in file_path else Path(file_path).name

        # Auto-detect MIME type if not provided
        if not mime_type:
            mime_type = get_mime_type(filename)
            if not mime_type:
                # Unknown file type - skip it
                ext = get_file_extension(filename) or 'unknown'
                debug_print(f"⚠️  Unknown MIME type for {filename} ({ext.upper()}), skipping")
                return {
                    'error': f'Unknown file type ({ext.upper()})',
                    'filename': filename,
                    'skipped': True,
                    'validation_failed': True
                }

        # Filter out audio files - NOT supported by Gemini
        if mime_type.startswith('audio/'):
            ext = get_file_extension(filename) or 'audio'
            debug_print(f"⚠️  Audio file not supported: {filename} ({ext.upper()}), skipping")
            return {
                'error': f'Audio files not supported ({ext.upper()})',
                'filename': filename,
                'skipped': True,
                'validation_failed': True
            }

        # PHASE 3: Check database cache first (persistent across server restarts)
        if self.chat_storage:
            db_cached = self.chat_storage.get_gemini_uri(file_path)
            if db_cached:
                # Reconstruct file object from cached data
                try:
                    file_obj = self.client.files.get(name=db_cached['gemini_name'])
                    # Use Gemini's actual MIME type from the file object
                    actual_mime_type = file_obj.mime_type or mime_type
                    result = {
                        'file': file_obj,
                        'uri': db_cached['gemini_uri'],
                        'name': db_cached['gemini_name'],
                        'display_name': db_cached['filename'],
                        'size_bytes': db_cached.get('size_bytes', 0),
                        'mime_type': actual_mime_type,  # Use actual MIME type from Gemini
                        'upload_time': time.time(),
                        'from_cache': True
                    }
                    # Also cache in memory for this session
                    self._file_cache[file_path] = result
                    return result
                except Exception as e:
                    # Cache miss - will re-upload silently
                    pass

        # Check in-memory cache
        if file_path in self._file_cache:
            cached = self._file_cache[file_path]
            age_hours = (time.time() - cached['upload_time']) / 3600

            # If cache still valid (under 48 hours), return cached
            if age_hours < self.cache_duration_hours:
                # Silently use cache
                return cached

            # Cache expired, remove it
            del self._file_cache[file_path]

        # Determine if this is a GCS blob or local file
        is_gcs = '/' in file_path and not Path(file_path).exists()

        try:
            # Get file bytes
            file_bytes = None
            if is_gcs and self.storage_manager:
                # Download from GCS
                file_bytes = self.storage_manager.download_pdf(file_path)
            else:
                # Local file
                path = Path(file_path)
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()

            # Validate file is not empty
            if not file_bytes or len(file_bytes) == 0:
                debug_print(f"⚠️  File is empty (0 bytes): {filename}")
                return {
                    'error': f'File is empty (0 bytes)',
                    'filename': filename,
                    'skipped': True
                }

            # Convert Office files to PDF if needed
            original_filename = filename
            conversion_attempted = False
            if needs_conversion(filename):
                ext = get_file_extension(filename) or 'unknown'
                conversion_attempted = True
                try:
                    pdf_bytes = convert_office_to_pdf(file_bytes, filename)
                    if pdf_bytes:
                        file_bytes = pdf_bytes
                        filename = filename.rsplit('.', 1)[0] + '.pdf'
                        mime_type = 'application/pdf'
                    else:
                        debug_print(f"⚠️  Conversion failed for {filename}, will try uploading original format")
                except Exception as conv_error:
                    debug_print(f"⚠️  Conversion error for {filename}: {conv_error}")
                    debug_print(f"    Will try uploading original format")

            ext = get_file_extension(filename) or 'file'

            # Proactive validation for PDFs to prevent 400 errors
            if mime_type == 'application/pdf':
                is_valid, error_reason = FileValidator.validate_pdf(file_bytes, filename)
                if not is_valid:
                    debug_print(f"❌ [VALIDATION] PDF validation failed for {filename}: {error_reason}")
                    return {
                        'error': error_reason,
                        'filename': filename,
                        'skipped': True,
                        'validation_failed': True  # Flag for special handling (don't retry)
                    }

            # Proactive validation for videos to prevent 400 errors
            if mime_type and mime_type.startswith('video/'):
                is_valid, error_reason = FileValidator.validate_video(file_bytes, filename)
                if not is_valid:
                    debug_print(f"❌ [VALIDATION] Video validation failed for {filename}: {error_reason}")
                    return {
                        'error': error_reason,
                        'filename': filename,
                        'skipped': True,
                        'validation_failed': True  # Flag for special handling (don't retry)
                    }

            # SPECIAL HANDLING: Text files (assignments/pages) should NOT be uploaded as files
            # Gemini File API expects documents with pages (PDFs, images), not plain text
            # Instead, return the text content directly for inline use
            if mime_type == 'text/plain' or filename.endswith('.txt'):
                debug_print(f"📝 [TXT FILE] Skipping Gemini upload for {filename} - will use text content directly")
                try:
                    text_content = file_bytes.decode('utf-8')

                    # Proactive validation for text files
                    is_valid, error_reason = FileValidator.validate_text(text_content, filename)
                    if not is_valid:
                        debug_print(f"❌ [VALIDATION] Text validation failed for {filename}: {error_reason}")
                        return {
                            'error': error_reason,
                            'filename': filename,
                            'skipped': True,
                            'validation_failed': True  # Flag for special handling (don't retry)
                        }

                    result = {
                        'file': None,
                        'uri': None,
                        'text_content': text_content,  # Return text instead of file URI
                        'name': filename,
                        'display_name': display_name or original_filename,
                        'size_bytes': len(file_bytes),
                        'mime_type': mime_type,
                        'upload_time': time.time(),
                        'is_text': True  # Flag to indicate this is text content, not a file
                    }
                    # Cache in memory for this session
                    self._file_cache[file_path] = result
                    return result
                except UnicodeDecodeError as e:
                    debug_print(f"❌ Failed to decode text file {filename}: {e}")
                    # Fall through to try file upload anyway

            # Silently upload - batch summary will show totals

            # Upload from bytes
            import io
            try:
                file_obj = self.client.files.upload(
                    file=io.BytesIO(file_bytes),
                    config={
                        'mime_type': mime_type,
                        'display_name': display_name or original_filename
                    }
                )

                # Wait for file to be in ACTIVE state
                # Files are uploaded but not immediately available for use
                max_wait = 10  # seconds
                wait_interval = 0.5  # check every 500ms
                waited = 0
                while file_obj.state.name != 'ACTIVE' and waited < max_wait:
                    time.sleep(wait_interval)
                    waited += wait_interval
                    file_obj = self.client.files.get(name=file_obj.name)

                if file_obj.state.name != 'ACTIVE':
                    debug_print(f"⚠️  Warning: {filename} uploaded but not ACTIVE after {max_wait}s (state: {file_obj.state.name})")

                # Cache the result
                result = {
                    'file': file_obj,
                    'uri': file_obj.uri,
                    'name': file_obj.name,
                    'display_name': file_obj.display_name,
                    'size_bytes': file_obj.size_bytes,
                    'mime_type': file_obj.mime_type or mime_type,  # Use Gemini's actual MIME type
                    'upload_time': time.time()
                }

                # Cache in memory
                self._file_cache[file_path] = result

                # PHASE 3: Cache in database for persistence across server restarts
                if self.chat_storage:
                    # Extract course_id from file_path (format: "course_id/filename")
                    course_id = file_path.split('/')[0] if '/' in file_path else 'unknown'
                    self.chat_storage.save_gemini_uri(
                        file_path=file_path,
                        course_id=course_id,
                        filename=original_filename,
                        gemini_uri=file_obj.uri,
                        gemini_name=file_obj.name,
                        mime_type=mime_type,
                        size_bytes=file_obj.size_bytes,
                        expires_hours=self.cache_duration_hours,
                        canvas_user_id=canvas_user_id
                    )

                # Silently uploaded - batch summary will show totals

                return result
            except Exception as upload_error:
                # Gemini rejected the file - provide clear error message
                ext = get_file_extension(filename) or 'file'
                error_msg = str(upload_error)
                if 'not supported' in error_msg.lower() or 'mime' in error_msg.lower():
                    debug_print(f"❌ Gemini rejected {filename}: {ext.upper()} format not supported by Gemini API")
                    debug_print(f"   Error: {error_msg}")
                    return {
                        "error": f"File type {ext.upper()} not supported by Gemini API. Supported: PDF, TXT, MD, CSV, PNG, JPG, JPEG, GIF, WEBP, MOV, MP4, AVI, WEBM, WMV, MPEG, MPG, FLV, 3GP",
                        "filename": filename,
                        "rejected": True
                    }
                else:
                    raise upload_error

        except Exception as e:
            error_msg = f"Failed to upload {filename}: {str(e)}"
            debug_print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg}

    async def _upload_single_pdf_async(self, file_path: str, display_name: Optional[str] = None) -> Dict:
        """Upload a single PDF asynchronously"""
        # Run synchronous upload in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.upload_pdf, file_path, display_name)
        return result

    async def upload_multiple_pdfs_async(self, file_info_list) -> Dict:
        """
        Upload multiple files in batches (async version)
        Supports PDFs, documents, images, etc.

        Args:
            file_info_list: List of (file_path, display_name) tuples or just file paths

        Returns:
            Dict with uploaded files and stats
        """
        # Normalize input: accept both tuples and plain strings
        normalized_list = []
        for item in file_info_list:
            if isinstance(item, tuple):
                normalized_list.append(item)
            else:
                # Plain string path - use None for display_name (will use filename)
                normalized_list.append((item, None))

        # Process in batches to avoid memory issues with large file sets
        BATCH_SIZE = 100  # Allow full parallel uploads for speed
        debug_print(f"Uploading {len(normalized_list)} files to Gemini in batches of {BATCH_SIZE}...")

        uploaded = []
        failed = []
        total_bytes = 0

        for i in range(0, len(normalized_list), BATCH_SIZE):
            batch = normalized_list[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(normalized_list) + BATCH_SIZE - 1) // BATCH_SIZE
            debug_print(f"   Batch {batch_num}/{total_batches}: Processing {len(batch)} files...")

            # Upload batch in parallel - pass both path and display_name
            upload_tasks = [self._upload_single_pdf_async(fp, dn) for fp, dn in batch]
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            # Process batch results
            for j, result in enumerate(results):
                file_path = batch[j][0]
                if isinstance(result, Exception):
                    failed.append({
                        'path': file_path,
                        'error': str(result)
                    })
                elif 'error' in result:
                    failed.append({
                        'path': file_path,
                        'error': result['error']
                    })
                else:
                    uploaded.append(result)
                    total_bytes += result.get('size_bytes', 0)

        # Log summary
        debug_print(f"Batch upload complete: {len(uploaded)} succeeded, {len(failed)} failed")

        if uploaded:
            # Group by file type
            type_counts = {}
            for file_info in uploaded:
                display_name = file_info.get('display_name', '')
                ext = display_name.split('.')[-1].lower() if '.' in display_name else 'unknown'
                type_counts[ext] = type_counts.get(ext, 0) + 1
            debug_print(f"📊 Uploaded file types: {type_counts}")
            debug_print(f"📊 Total size: {total_bytes:,} bytes ({total_bytes / (1024*1024):.1f} MB)")

            # Show sample files
            sample_files = [f.get('display_name', 'unknown') for f in uploaded[:5]]
            debug_print(f"📄 Sample files: {sample_files}")

        if failed:
            debug_print(f"❌ Failed uploads:")
            for fail in failed[:3]:  # Show first 3 failures
                debug_print(f"   - {fail.get('path', 'unknown')}: {fail.get('error', 'unknown error')}")

        return {
            'success': True,
            'uploaded_count': len(uploaded),
            'failed_count': len(failed),
            'total_bytes': total_bytes,
            'files': uploaded,
            'failed': failed if failed else None
        }

    def upload_multiple_pdfs(self, file_paths: list, progress_callback=None) -> Dict:
        """
        Upload multiple files with optional progress callback (sync version)
        Supports PDFs, documents, images, etc.

        Args:
            file_paths: List of file paths
            progress_callback: Optional callback(current, total, file_name) for progress updates

        Returns:
            Dict with uploaded files and stats
        """
        uploaded = []
        failed = []
        total_bytes = 0
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths):
            # Call progress callback if provided
            if progress_callback:
                file_name = Path(file_path).name
                progress_callback(i + 1, total_files, file_name)

            result = self.upload_pdf(file_path)

            if 'error' in result:
                failed.append({
                    'path': file_path,
                    'error': result['error']
                })
            else:
                uploaded.append(result)
                total_bytes += result.get('size_bytes', 0)

        return {
            'success': True,
            'uploaded_count': len(uploaded),
            'failed_count': len(failed),
            'total_bytes': total_bytes,
            'files': uploaded,
            'failed': failed if failed else None
        }

    def clear_cache(self, file_path: Optional[str] = None):
        """
        Clear file cache

        Args:
            file_path: Optional specific file to clear. If None, clears all.
        """
        if file_path:
            self._file_cache.pop(file_path, None)
            debug_print(f"🗑️  Cleared cache for {Path(file_path).name}")
        else:
            self._file_cache.clear()
            debug_print(f"🗑️  Cleared all file upload cache")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_bytes = sum(entry.get('size_bytes', 0) for entry in self._file_cache.values())
        total_mb = total_bytes / (1024 * 1024)

        return {
            'cached_files': len(self._file_cache),
            'total_bytes': total_bytes,
            'total_mb': round(total_mb, 2),
            'cache_duration_hours': self.cache_duration_hours
        }
