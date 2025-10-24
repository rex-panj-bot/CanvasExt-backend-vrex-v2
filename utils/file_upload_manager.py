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
    def __init__(self, client: genai.Client, cache_duration_hours: int = 48, storage_manager=None):
        """
        Initialize File Upload Manager

        Args:
            client: Gemini client instance
            cache_duration_hours: How long to cache file URIs (default: 48 hours, max for File API)
            storage_manager: Optional StorageManager for GCS file access
        """
        self.client = client
        self.cache_duration_hours = cache_duration_hours
        self.storage_manager = storage_manager

        # Cache: {file_path: {uri, upload_time, file_object}}
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
                return {"error": f"Unsupported file type for {filename}"}

        # Check if file type is supported by Gemini File API
        if not is_supported_by_gemini(filename):
            ext = get_file_extension(filename)
            print(f"⚠️  Skipping {filename}: {ext.upper()} files are not supported by Gemini File API")
            print(f"    Supported formats: PDF, TXT, MD, CSV, PNG, JPG, JPEG, GIF, WEBP")
            return {"error": f"File type not supported: {ext.upper()} files cannot be processed by Gemini. Please convert to PDF first.", "skipped": True}

        # Check cache first
        if file_path in self._file_cache:
            cached = self._file_cache[file_path]
            age_hours = (time.time() - cached['upload_time']) / 3600

            # If cache still valid (under 48 hours), return cached
            if age_hours < self.cache_duration_hours:
                print(f"✅ Cache hit: {filename} (uploaded {age_hours:.1f}h ago)")
                return cached

            # Cache expired, remove it
            print(f"⚠️  Cache expired for {filename}, re-uploading...")
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
            if needs_conversion(filename):
                ext = get_file_extension(filename)
                print(f"🔄 Converting {filename} ({ext.upper()}) to PDF...")
                pdf_bytes = convert_office_to_pdf(file_bytes, filename)
                if pdf_bytes:
                    file_bytes = pdf_bytes
                    filename = filename.rsplit('.', 1)[0] + '.pdf'
                    mime_type = 'application/pdf'
                    print(f"✅ Converted {original_filename} to PDF ({len(pdf_bytes):,} bytes)")
                else:
                    print(f"⚠️  Conversion failed for {filename}, uploading original")

            ext = get_file_extension(filename) or 'file'
            print(f"📤 Uploading {filename} ({ext.upper()}, {mime_type}) to Gemini File API...")

            # Upload from bytes
            import io
            file_obj = self.client.files.upload(
                file=io.BytesIO(file_bytes),
                config={
                    'mime_type': mime_type,
                    'display_name': display_name or original_filename
                }
            )

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

            self._file_cache[file_path] = result
            ext = get_file_extension(filename) or 'file'
            print(f"✅ Uploaded {filename} ({ext.upper()}, {file_obj.size_bytes:,} bytes) - URI: {file_obj.uri[:60]}...")

            return result

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
        Upload multiple files in parallel (async version)
        Supports PDFs, documents, images, etc.

        Args:
            file_paths: List of file paths

        Returns:
            Dict with uploaded files and stats
        """
        print(f"📤 Uploading {len(file_paths)} files to Gemini...")

        # Upload all files in parallel
        upload_tasks = [self._upload_single_pdf_async(fp) for fp in file_paths]
        results = await asyncio.gather(*upload_tasks, return_exceptions=True)

        # Process results
        uploaded = []
        failed = []
        total_bytes = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({
                    'path': file_paths[i],
                    'error': str(result)
                })
            elif 'error' in result:
                failed.append({
                    'path': file_paths[i],
                    'error': result['error']
                })
            else:
                uploaded.append(result)
                total_bytes += result.get('size_bytes', 0)

        print(f"✅ Parallel upload complete: {len(uploaded)} succeeded, {len(failed)} failed")

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
