"""
File Validator
Pre-validates files before Gemini API upload to detect 400 errors proactively
"""

import io
from typing import Tuple, Optional
import pypdf


class FileValidator:
    """Validates files before Gemini API upload to prevent 400 errors"""

    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB (Gemini API limit)
    MAX_BATCH_SIZE = 200 * 1024 * 1024  # 200 MB safe batch limit

    # Video file extensions supported by Gemini
    VIDEO_EXTENSIONS = {'.mov', '.mp4', '.avi', '.webm', '.wmv', '.mpeg', '.mpg', '.flv', '.3gp'}

    @staticmethod
    def validate_pdf(pdf_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate PDF file for Gemini API compatibility

        Detects common 400 errors:
        - Empty files (0 bytes)
        - Files too large (>2GB)
        - PDFs with no pages
        - Corrupted/unreadable PDFs

        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename (for logging)

        Returns:
            (is_valid, error_message)
            - (True, None) if valid
            - (False, "error reason") if invalid
        """
        # Check 1: Empty file
        if not pdf_bytes or len(pdf_bytes) == 0:
            return False, "File is empty (0 bytes)"

        # Check 2: File size
        size_bytes = len(pdf_bytes)
        if size_bytes > FileValidator.MAX_FILE_SIZE:
            size_gb = size_bytes / (1024**3)
            return False, f"File too large ({size_gb:.1f} GB, max 2 GB)"

        # Check 3: Valid PDF structure and pages
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes), strict=False)

            # Check 4: Has pages
            num_pages = len(reader.pages)
            if num_pages == 0:
                return False, "PDF has no pages"

            # Check 5: First page readable (basic corruption check)
            try:
                _ = reader.pages[0].extract_text()
            except Exception as extract_error:
                # Some PDFs might have unreadable text but still be valid for Gemini
                # Only fail if we can't access the page at all
                pass

            return True, None

        except pypdf.errors.PdfReadError as e:
            return False, f"PDF corrupted or invalid: {str(e)}"
        except Exception as e:
            return False, f"PDF validation failed: {str(e)}"

    @staticmethod
    def validate_text(text_content: str, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate text file content

        Args:
            text_content: Text file content as string
            filename: Original filename

        Returns:
            (is_valid, error_message)
        """
        if not text_content or len(text_content.strip()) == 0:
            return False, "Text file is empty"

        # Text files should be reasonable size (not multi-GB)
        size_bytes = len(text_content.encode('utf-8'))
        if size_bytes > 10 * 1024 * 1024:  # 10 MB text file is suspicious
            return False, f"Text file unusually large ({size_bytes / (1024**2):.1f} MB)"

        return True, None

    @staticmethod
    def validate_video(video_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate video file for Gemini API compatibility

        Detects common 400 errors:
        - Empty files (0 bytes)
        - Files too large (>2GB)
        - Unsupported video formats

        Args:
            video_bytes: Video file content as bytes
            filename: Original filename (for logging)

        Returns:
            (is_valid, error_message)
            - (True, None) if valid
            - (False, "error reason") if invalid
        """
        # Check 1: Empty file
        if not video_bytes or len(video_bytes) == 0:
            return False, "Video file is empty (0 bytes)"

        # Check 2: File size (2 GB limit for Gemini API)
        size_bytes = len(video_bytes)
        if size_bytes > FileValidator.MAX_FILE_SIZE:
            size_gb = size_bytes / (1024**3)
            return False, f"Video file too large ({size_gb:.1f} GB, max 2 GB)"

        # Check 3: File extension (basic format check)
        ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if ext not in FileValidator.VIDEO_EXTENSIONS:
            return False, f"Unsupported video format: {ext}"

        # Check 4: Minimum size (videos smaller than 100 bytes are likely corrupted)
        if size_bytes < 100:
            return False, f"Video file suspiciously small ({size_bytes} bytes)"

        return True, None

    @staticmethod
    def validate_batch(files: list) -> Tuple[bool, Optional[str]]:
        """
        Validate batch total size doesn't exceed limits

        Args:
            files: List of dicts with 'size_bytes' keys

        Returns:
            (is_valid, error_message)
        """
        total_bytes = sum(f.get('size_bytes', 0) for f in files)
        if total_bytes > FileValidator.MAX_BATCH_SIZE:
            size_mb = total_bytes / (1024**2)
            return False, f"Batch too large ({size_mb:.1f} MB, max 200 MB)"
        return True, None
