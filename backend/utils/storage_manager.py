"""
Storage Manager - Google Cloud Storage Integration
Handles file storage and retrieval from GCS bucket
Supports multiple file types (PDFs, documents, images, etc.)
"""

from google.cloud import storage
from google.oauth2 import service_account
import os
from pathlib import Path
from typing import List, Optional, BinaryIO
import logging
from io import BytesIO
from .mime_types import get_mime_type, get_file_extension

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file storage in Google Cloud Storage (PDFs, documents, images, etc.)"""

    def __init__(self, bucket_name: str, project_id: str, credentials_path: Optional[str] = None):
        """
        Initialize Storage Manager

        Args:
            bucket_name: GCS bucket name (e.g., 'canvas-extension-pdfs')
            project_id: GCP project ID
            credentials_path: Path to service account JSON (optional, uses GOOGLE_APPLICATION_CREDENTIALS env var if not provided)
        """
        self.bucket_name = bucket_name
        self.project_id = project_id

        # Initialize GCS client
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = storage.Client(project=project_id, credentials=credentials)
        else:
            # Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
            self.client = storage.Client(project=project_id)

        self.bucket = self.client.bucket(bucket_name)

        logger.info(f"StorageManager initialized with bucket: {bucket_name}")

    def upload_pdf(self, course_id: str, filename: str, file_content: bytes, content_type: Optional[str] = None) -> str:
        """
        Upload a file to GCS (supports PDFs and other file types)

        Args:
            course_id: Course identifier
            filename: Original filename
            file_content: File content as bytes
            content_type: MIME type (auto-detected from filename if not provided)

        Returns:
            GCS blob name (path in bucket)
        """
        try:
            # Sanitize filename: replace forward slashes (GCS path separator) with dashes
            # Forward slashes in filenames cause issues with GCS blob paths
            sanitized_filename = filename.replace('/', '-')
            if sanitized_filename != filename:
                print(f"   ⚠️  Sanitized filename: '{filename}' → '{sanitized_filename}'")

            # Create blob name: {course_id}/{filename}
            blob_name = f"{course_id}/{sanitized_filename}"
            blob = self.bucket.blob(blob_name)

            # OPTIMIZATION: Check if file already exists in GCS to avoid re-upload
            if blob.exists():
                blob.reload()  # Load metadata to check size
                if blob.size == len(file_content):
                    ext = get_file_extension(filename) or 'file'
                    logger.info(f"⚡ File already in GCS, skipping upload: {blob_name} ({blob.size} bytes, {ext.upper()})")
                    return blob_name
                else:
                    logger.info(f"⚠️  File exists but size mismatch ({blob.size} != {len(file_content)}), re-uploading: {filename}")

            # Auto-detect MIME type if not provided
            if not content_type:
                content_type = get_mime_type(filename)
                if not content_type:
                    # Default to binary if unknown
                    content_type = 'application/octet-stream'
                    logger.warning(f"Unknown file type for {filename}, using {content_type}")

            # Upload with metadata
            blob.upload_from_string(
                file_content,
                content_type=content_type,
                timeout=300  # 5 minute timeout for large files
            )

            # Make publicly readable (optional - remove if you want private)
            # blob.make_public()

            ext = get_file_extension(filename) or 'file'
            logger.info(f"✅ Uploaded {blob_name} ({len(file_content)} bytes, {ext.upper()}, {content_type})")
            return blob_name

        except Exception as e:
            logger.error(f"❌ Failed to upload {filename}: {e}")
            raise

    def download_pdf(self, blob_name: str) -> bytes:
        """
        Download a PDF from GCS

        Args:
            blob_name: Full path in bucket (e.g., '12345/lecture1.pdf')

        Returns:
            PDF content as bytes
        """
        try:
            blob = self.bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(f"Blob {blob_name} not found in bucket {self.bucket_name}")

            content = blob.download_as_bytes()
            logger.info(f"Downloaded {blob_name} ({len(content)} bytes)")
            return content

        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {e}")
            raise

    def download_pdf_to_file(self, blob_name: str, local_path: str):
        """
        Download a PDF from GCS to a local file

        Args:
            blob_name: Full path in bucket
            local_path: Local filesystem path to save to
        """
        try:
            blob = self.bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(f"Blob {blob_name} not found in bucket {self.bucket_name}")

            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {blob_name} to {local_path}")

        except Exception as e:
            logger.error(f"Failed to download {blob_name} to file: {e}")
            raise

    def list_files(self, course_id: Optional[str] = None, prefix: Optional[str] = None, file_extension: Optional[str] = None) -> List[str]:
        """
        List all files in bucket, optionally filtered by course and file type

        Args:
            course_id: Filter by course ID (optional)
            prefix: Custom prefix to filter by (optional, overrides course_id)
            file_extension: Filter by file extension (e.g., 'pdf', 'docx') - if None, returns all files

        Returns:
            List of blob names
        """
        try:
            # Use custom prefix or course_id
            search_prefix = prefix if prefix else (f"{course_id}/" if course_id else None)

            blobs = self.client.list_blobs(self.bucket_name, prefix=search_prefix)

            # Filter by extension if specified
            if file_extension:
                blob_names = [blob.name for blob in blobs if blob.name.endswith(f'.{file_extension}')]
                logger.info(f"Found {len(blob_names)} {file_extension.upper()} files" + (f" for course {course_id}" if course_id else ""))
            else:
                blob_names = [blob.name for blob in blobs]
                logger.info(f"Found {len(blob_names)} files" + (f" for course {course_id}" if course_id else ""))

            return blob_names

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise

    def file_exists(self, blob_name: str) -> bool:
        """
        Check if a file exists in GCS

        Args:
            blob_name: Full path in bucket (e.g., '12345/lecture1.pdf')

        Returns:
            True if file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if {blob_name} exists: {e}")
            return False

    def get_signed_url(self, blob_name: str, expires_in_seconds: int = 3600) -> Optional[str]:
        """
        Generate a signed URL for downloading a file from GCS

        Args:
            blob_name: Full path in bucket (e.g., '12345/lecture1.pdf')
            expires_in_seconds: URL expiration time in seconds (default: 1 hour)

        Returns:
            Signed URL string, or None if file doesn't exist
        """
        try:
            from datetime import timedelta

            blob = self.bucket.blob(blob_name)

            if not blob.exists():
                logger.warning(f"Blob {blob_name} not found, cannot generate signed URL")
                return None

            # Generate signed URL (valid for specified duration)
            url = blob.generate_signed_url(
                expiration=timedelta(seconds=expires_in_seconds),
                method='GET',
                version='v4'
            )

            logger.info(f"Generated signed URL for {blob_name} (expires in {expires_in_seconds}s)")
            return url

        except Exception as e:
            logger.error(f"Failed to generate signed URL for {blob_name}: {e}")
            return None

    def delete_file(self, blob_name: str) -> bool:
        """
        Delete a PDF from GCS

        Args:
            blob_name: Full path in bucket

        Returns:
            True if deleted successfully
        """
        try:
            blob = self.bucket.blob(blob_name)

            if not blob.exists():
                logger.warning(f"Blob {blob_name} not found, cannot delete")
                return False

            blob.delete()
            logger.info(f"Deleted {blob_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete {blob_name}: {e}")
            raise

    def file_exists(self, blob_name: str) -> bool:
        """
        Check if a file exists in GCS

        Args:
            blob_name: Full path in bucket

        Returns:
            True if file exists
        """
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()

        except Exception as e:
            logger.error(f"Failed to check if {blob_name} exists: {e}")
            return False

    def get_file_metadata(self, blob_name: str) -> dict:
        """
        Get metadata for a file

        Args:
            blob_name: Full path in bucket

        Returns:
            Dictionary with metadata (size, created, updated, content_type, etc.)
        """
        try:
            blob = self.bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(f"Blob {blob_name} not found")

            blob.reload()  # Fetch metadata from GCS

            return {
                'name': blob.name,
                'size': blob.size,
                'content_type': blob.content_type,
                'created': blob.time_created,
                'updated': blob.updated,
                'md5_hash': blob.md5_hash,
                'public_url': blob.public_url if blob.public_url else None
            }

        except Exception as e:
            logger.error(f"Failed to get metadata for {blob_name}: {e}")
            raise

    def get_signed_url(self, blob_name: str, expiration_minutes: int = 60) -> str:
        """
        Generate a signed URL for temporary access to a private file

        Args:
            blob_name: Full path in bucket
            expiration_minutes: URL expiration time in minutes (default 60)

        Returns:
            Signed URL string
        """
        try:
            blob = self.bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(f"Blob {blob_name} not found")

            # Generate signed URL (valid for specified duration)
            from datetime import timedelta
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET"
            )

            logger.info(f"Generated signed URL for {blob_name} (expires in {expiration_minutes}m)")
            return url

        except Exception as e:
            logger.error(f"Failed to generate signed URL for {blob_name}: {e}")
            raise

    def bulk_upload(self, course_id: str, files: List[tuple]) -> dict:
        """
        Upload multiple files in bulk

        Args:
            course_id: Course identifier
            files: List of tuples (filename, content_bytes)

        Returns:
            Dictionary with success/failure counts and details
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(files)
        }

        for filename, content in files:
            try:
                blob_name = self.upload_pdf(course_id, filename, content)
                results['successful'].append({
                    'filename': filename,
                    'blob_name': blob_name,
                    'size': len(content)
                })
            except Exception as e:
                results['failed'].append({
                    'filename': filename,
                    'error': str(e)
                })

        logger.info(f"Bulk upload: {len(results['successful'])}/{results['total']} successful")
        return results
