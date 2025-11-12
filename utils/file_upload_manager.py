"""
File Upload Manager for Gemini File API
Handles file uploads to Gemini (PDFs, documents, images, etc.)
"""

from google import genai
from pathlib import Path
from typing import Dict, Optional, List
import time
import asyncio
from .mime_types import get_mime_type, get_file_extension, is_supported_by_gemini
from .file_converter import convert_office_to_pdf, needs_conversion


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

    def upload_pdf(self, file_path: str, display_name: Optional[str] = None, mime_type: Optional[str] = None) -> Dict:
        """
        Upload a file to Gemini File API (supports PDFs, documents, images, etc.)

        Args:
            file_path: Path to file (local path or GCS blob name)
            display_name: Optional display name for the file
            mime_type: Optional MIME type (auto-detected from filename if not provided)

        Returns:
            Dict with file object and metadata
        """
        # Extract filename for logging
        filename = file_path.split('/')[-1] if '/' in file_path else Path(file_path).name

        # Auto-detect MIME type if not provided
        if not mime_type:
            mime_type = get_mime_type(filename)
            if not mime_type:
                # Try to upload anyway - let Gemini reject it if unsupported
                ext = get_file_extension(filename) or 'unknown'
                print(f"⚠️  Unknown MIME type for {filename} ({ext.upper()}), will attempt upload anyway")
                # Use a generic MIME type as fallback
                mime_type = 'application/octet-stream'

        # PHASE 3: Check database cache first (persistent across server restarts)
        if self.chat_storage:
            db_cached = self.chat_storage.get_gemini_uri(file_path)
            if db_cached:
                print(f"✅ [CACHE HIT] Database: {filename} (saved ~5-10s upload time)")
                # Reconstruct file object from cached data
                try:
                    file_obj = self.client.files.get(name=db_cached['gemini_name'])
                    result = {
                        'file': file_obj,
                        'uri': db_cached['gemini_uri'],
                        'name': db_cached['gemini_name'],
                        'display_name': db_cached['filename'],
                        'size_bytes': db_cached.get('size_bytes', 0),
                        'mime_type': db_cached.get('mime_type', mime_type),
                        'upload_time': time.time(),
                        'from_cache': True
                    }
                    # Also cache in memory for this session
                    self._file_cache[file_path] = result
                    return result
                except Exception as e:
                    print(f"⚠️  [CACHE MISS] Cached file not accessible in Gemini, will re-upload: {e}")

        # Check in-memory cache
        if file_path in self._file_cache:
            cached = self._file_cache[file_path]
            age_hours = (time.time() - cached['upload_time']) / 3600

            # If cache still valid (under 48 hours), return cached
            if age_hours < self.cache_duration_hours:
                print(f"✅ [CACHE HIT] Memory: {filename} (age: {age_hours:.1f}h, saved ~5-10s)")
                return cached

            # Cache expired, remove it
            print(f"⚠️  [CACHE MISS] Expired: {filename} (age: {age_hours:.1f}h), re-uploading...")
            del self._file_cache[file_path]

        # Determine if this is a GCS blob or local file
        is_gcs = '/' in file_path and not Path(file_path).exists()

        try:
            # Get file bytes
            file_bytes = None
            if is_gcs and self.storage_manager:
                # Download from GCS
                print(f"📤 Downloading {filename} from GCS...")
                file_bytes = self.storage_manager.download_pdf(file_path)
            else:
                # Local file
                path = Path(file_path)
                if not path.exists():
                    return {"error": f"File not found: {file_path}"}
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()

            # Convert Office files to PDF if needed
            original_filename = filename
            conversion_attempted = False
            if needs_conversion(filename):
                ext = get_file_extension(filename) or 'unknown'
                print(f"🔄 Converting {filename} ({ext.upper()}) to PDF...")
                conversion_attempted = True
                try:
                    pdf_bytes = convert_office_to_pdf(file_bytes, filename)
                    if pdf_bytes:
                        file_bytes = pdf_bytes
                        filename = filename.rsplit('.', 1)[0] + '.pdf'
                        mime_type = 'application/pdf'
                        print(f"✅ Converted {original_filename} to PDF ({len(pdf_bytes):,} bytes)")
                    else:
                        print(f"⚠️  Conversion failed for {filename}, will try uploading original format")
                except Exception as conv_error:
                    print(f"⚠️  Conversion error for {filename}: {conv_error}")
                    print(f"    Will try uploading original format")

            ext = get_file_extension(filename) or 'file'
            print(f"📤 [CACHE MISS] Uploading {filename} ({ext.upper()}, {mime_type}) to Gemini File API (~5-10s)...")
            if conversion_attempted and filename.endswith('.pdf'):
                print(f"    (Converted from {original_filename})")

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
                    print(f"⚠️  Warning: {filename} uploaded but not ACTIVE after {max_wait}s (state: {file_obj.state.name})")

                # Cache the result
                result = {
                    'file': file_obj,
                    'uri': file_obj.uri,
                    'name': file_obj.name,
                    'display_name': file_obj.display_name,
                    'size_bytes': file_obj.size_bytes,
                    'mime_type': mime_type,
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
                        expires_hours=self.cache_duration_hours
                    )

                ext = get_file_extension(filename) or 'file'
                print(f"✅ Uploaded {filename} ({ext.upper()}, {file_obj.size_bytes:,} bytes) - URI: {file_obj.uri[:60]}...")

                return result
            except Exception as upload_error:
                # Gemini rejected the file - provide clear error message
                ext = get_file_extension(filename) or 'file'
                error_msg = str(upload_error)
                if 'not supported' in error_msg.lower() or 'mime' in error_msg.lower():
                    print(f"❌ Gemini rejected {filename}: {ext.upper()} format not supported by Gemini API")
                    print(f"   Error: {error_msg}")
                    return {
                        "error": f"File type {ext.upper()} not supported by Gemini API. Supported: PDF, TXT, MD, CSV, PNG, JPG, JPEG, GIF, WEBP",
                        "filename": filename,
                        "rejected": True
                    }
                else:
                    raise upload_error

        except Exception as e:
            error_msg = f"Failed to upload {filename}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg}

    async def _upload_single_pdf_async(self, file_path: str) -> Dict:
        """Upload a single PDF asynchronously"""
        # Run synchronous upload in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.upload_pdf, file_path)
        return result

    async def upload_multiple_pdfs_async(self, file_paths: List[str]) -> Dict:
        """
        Upload multiple files in batches (async version)
        Supports PDFs, documents, images, etc.

        Args:
            file_paths: List of file paths

        Returns:
            Dict with uploaded files and stats
        """
        # Process in batches to avoid memory issues with large file sets
        BATCH_SIZE = 100  # Allow full parallel uploads for speed
        print(f"📤 Uploading {len(file_paths)} files to Gemini in batches of {BATCH_SIZE}...")

        uploaded = []
        failed = []
        total_bytes = 0

        for i in range(0, len(file_paths), BATCH_SIZE):
            batch = file_paths[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(file_paths) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"   Batch {batch_num}/{total_batches}: Processing {len(batch)} files...")

            # Upload batch in parallel
            upload_tasks = [self._upload_single_pdf_async(fp) for fp in batch]
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            # Process batch results
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    failed.append({
                        'path': batch[j],
                        'error': str(result)
                    })
                elif 'error' in result:
                    failed.append({
                        'path': batch[j],
                        'error': result['error']
                    })
                else:
                    uploaded.append(result)
                    total_bytes += result.get('size_bytes', 0)

        # Log summary
        print(f"✅ Batch upload complete: {len(uploaded)} succeeded, {len(failed)} failed")

        if uploaded:
            # Group by file type
            type_counts = {}
            for file_info in uploaded:
                display_name = file_info.get('display_name', '')
                ext = display_name.split('.')[-1].lower() if '.' in display_name else 'unknown'
                type_counts[ext] = type_counts.get(ext, 0) + 1
            print(f"📊 Uploaded file types: {type_counts}")
            print(f"📊 Total size: {total_bytes:,} bytes ({total_bytes / (1024*1024):.1f} MB)")

            # Show sample files
            sample_files = [f.get('display_name', 'unknown') for f in uploaded[:5]]
            print(f"📄 Sample files: {sample_files}")

        if failed:
            print(f"❌ Failed uploads:")
            for fail in failed[:3]:  # Show first 3 failures
                print(f"   - {fail.get('path', 'unknown')}: {fail.get('error', 'unknown error')}")

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
            print(f"🗑️  Cleared cache for {Path(file_path).name}")
        else:
            self._file_cache.clear()
            print(f"🗑️  Cleared all file upload cache")

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
