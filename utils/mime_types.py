"""
MIME Type Utilities
Handles file type detection and MIME type mapping for multi-format support
"""

from typing import Optional

# Comprehensive MIME type mapping for supported file formats
MIME_TYPE_MAP = {
    # Documents
    'pdf': 'application/pdf',
    'txt': 'text/plain',
    'md': 'text/markdown',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'rtf': 'application/rtf',

    # Spreadsheets
    'xls': 'application/vnd.ms-excel',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'csv': 'text/csv',

    # Presentations
    'ppt': 'application/vnd.ms-powerpoint',
    'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',

    # Images
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'webp': 'image/webp',
}

# Gemini File API supported file types (verified working MIME types only)
GEMINI_SUPPORTED = {
    # Text files - directly supported
    'txt', 'md', 'csv',

    # Documents - PDF has best support
    'pdf',

    # Images - supported for vision capabilities
    'png', 'jpg', 'jpeg', 'gif', 'webp',

    # Note: Office formats (docx, xlsx, pptx) are NOT supported by Gemini File API
    # They need to be converted to PDF or extracted as text first
}

# File types that need conversion before Gemini upload (NOT directly supported)
NEEDS_CONVERSION = {
    # Microsoft Office → PDF
    'doc',   # Convert to PDF
    'docx',  # Convert to PDF
    'xls',   # Convert to PDF
    'xlsx',  # Convert to PDF
    'ppt',   # Convert to PDF
    'pptx',  # Convert to PDF
    'rtf',   # Convert to PDF or TXT

    # OpenDocument formats → PDF
    'odt',   # OpenDocument Text → PDF
    'ods',   # OpenDocument Spreadsheet → PDF
    'odp',   # OpenDocument Presentation → PDF

    # Web/data formats → TXT (Gemini supports text/plain)
    'html',  # HTML → TXT
    'htm',   # HTML → TXT
    'xml',   # XML → TXT
    'json',  # JSON → TXT
}

def get_mime_type(filename: str) -> Optional[str]:
    """
    Get MIME type from filename extension

    Args:
        filename: Name of the file

    Returns:
        MIME type string or None if not supported
    """
    if not filename:
        return None

    # Extract extension (case-insensitive)
    ext = filename.split('.')[-1].lower()

    return MIME_TYPE_MAP.get(ext)

def get_file_extension(filename: str) -> Optional[str]:
    """
    Extract file extension from filename

    Args:
        filename: Name of the file

    Returns:
        File extension (lowercase) or None
    """
    if not filename or '.' not in filename:
        return None

    return filename.split('.')[-1].lower()

def is_supported_by_gemini(filename: str) -> bool:
    """
    Check if file type is supported by Gemini API

    Args:
        filename: Name of the file

    Returns:
        True if supported, False otherwise
    """
    ext = get_file_extension(filename)
    return ext in GEMINI_SUPPORTED if ext else False

def needs_conversion(filename: str) -> bool:
    """
    Check if file needs conversion before Gemini upload

    Args:
        filename: Name of the file

    Returns:
        True if needs conversion, False otherwise
    """
    ext = get_file_extension(filename)
    return ext in NEEDS_CONVERSION if ext else False

def get_display_type(filename: str) -> str:
    """
    Get human-readable file type for display

    Args:
        filename: Name of the file

    Returns:
        Display type string (e.g., "PDF Document", "Excel Spreadsheet")
    """
    ext = get_file_extension(filename)

    type_names = {
        'pdf': 'PDF Document',
        'txt': 'Text File',
        'md': 'Markdown File',
        'doc': 'Word Document (Legacy)',
        'docx': 'Word Document',
        'rtf': 'Rich Text File',
        'xls': 'Excel Spreadsheet (Legacy)',
        'xlsx': 'Excel Spreadsheet',
        'csv': 'CSV Spreadsheet',
        'ppt': 'PowerPoint (Legacy)',
        'pptx': 'PowerPoint Presentation',
        'png': 'PNG Image',
        'jpg': 'JPEG Image',
        'jpeg': 'JPEG Image',
        'gif': 'GIF Image',
        'webp': 'WebP Image',
    }

    return type_names.get(ext, f'{ext.upper()} File' if ext else 'Unknown File')
