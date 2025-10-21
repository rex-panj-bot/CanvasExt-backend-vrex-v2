"""
File Upload Manager for Gemini File API
Handles direct PDF uploads to Gemini without text extraction
"""

from google import genai
from pathlib import Path
from typing import Dict, Optional, List
import time
import asyncio


class FileUploadManager:
    def __init__(self, client: genai.Client, cache_duration_hours: int = 48):
        """
        Initialize File Upload Manager

        Args:
            client: Gemini client instance
            cache_duration_hours: How long to cache file URIs (default: 48 hours, max for File API)
        """
        self.client = client
        self.cache_duration_hours = cache_duration_hours

        # Cache: {file_path: {uri, upload_time, file_object}}
        self._file_cache = {}

    def upload_pdf(self, file_path: str, display_name: Optional[str] = None) -> Dict:
        """
        Upload a PDF file to Gemini File API

        Args:
            file_path: Path to PDF file
            display_name: Optional display name for the file

        Returns:
            Dict with file object and metadata
        """
        # Check cache first
        if file_path in self._file_cache:
            cached = self._file_cache[file_path]
            age_hours = (time.time() - cached['upload_time']) / 3600

            # If cache still valid (under 48 hours), return cached
            if age_hours < self.cache_duration_hours:
                print(f"✅ Cache hit: {Path(file_path).name} (uploaded {age_hours:.1f}h ago)")
                return cached

            # Cache expired, remove it
            print(f"⚠️  Cache expired for {Path(file_path).name}, re-uploading...")
            del self._file_cache[file_path]

        # Upload file
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        try:
            print(f"📤 Uploading {path.name} to Gemini File API...")

            with open(file_path, 'rb') as f:
                file_obj = self.client.files.upload(
                    file=f,
                    config={
                        'mime_type': 'application/pdf',
                        'display_name': display_name or path.name
                    }
                )

            # Cache the result
            result = {
                'file': file_obj,
                'uri': file_obj.uri,
                'name': file_obj.name,
                'display_name': file_obj.display_name,
                'size_bytes': file_obj.size_bytes,
                'upload_time': time.time()
            }

            self._file_cache[file_path] = result
            print(f"✅ Uploaded {path.name} ({file_obj.size_bytes:,} bytes)")

            return result

        except Exception as e:
            error_msg = f"Failed to upload {path.name}: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

    async def _upload_single_pdf_async(self, file_path: str) -> Dict:
        """Upload a single PDF asynchronously"""
        # Run synchronous upload in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.upload_pdf, file_path)
        return result

    async def upload_multiple_pdfs_async(self, file_paths: List[str]) -> Dict:
        """
        Upload multiple PDF files in parallel (async version)

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
        Upload multiple PDF files with optional progress callback (sync version)

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
