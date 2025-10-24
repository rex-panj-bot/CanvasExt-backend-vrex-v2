"""
File Converter - Convert Office files to PDF
Handles PPTX, DOCX, XLSX conversion to PDF format for Gemini compatibility
"""

import io
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, BinaryIO
import logging

logger = logging.getLogger(__name__)


def convert_office_to_pdf(file_bytes: bytes, filename: str) -> Optional[bytes]:
    """
    Convert Office document (PPTX, DOCX, XLSX) to PDF

    Args:
        file_bytes: Original file content as bytes
        filename: Original filename (used to determine conversion method)

    Returns:
        PDF bytes if successful, None if conversion failed
    """
    ext = filename.split('.')[-1].lower() if '.' in filename else ''

    if ext not in ['pptx', 'docx', 'xlsx', 'ppt', 'doc', 'xls']:
        logger.warning(f"File {filename} doesn't need conversion")
        return None

    try:
        # Try LibreOffice conversion (most reliable)
        pdf_bytes = _convert_with_libreoffice(file_bytes, filename, ext)
        if pdf_bytes:
            return pdf_bytes

        # Fallback: Try python-pptx/docx for text extraction as PDF
        logger.info(f"LibreOffice not available, trying Python library extraction for {filename}")
        pdf_bytes = _extract_text_as_pdf(file_bytes, ext)
        return pdf_bytes

    except Exception as e:
        logger.error(f"Failed to convert {filename} to PDF: {e}")
        return None


def _convert_with_libreoffice(file_bytes: bytes, filename: str, ext: str) -> Optional[bytes]:
    """
    Convert Office file to PDF using LibreOffice headless mode
    """
    # Check if LibreOffice is available
    libreoffice_paths = [
        '/usr/bin/libreoffice',  # Linux
        '/usr/bin/soffice',       # Linux alternative
        '/Applications/LibreOffice.app/Contents/MacOS/soffice',  # macOS
        'C:\\Program Files\\LibreOffice\\program\\soffice.exe',   # Windows
    ]

    soffice_path = None
    for path in libreoffice_paths:
        if os.path.exists(path):
            soffice_path = path
            break

    if not soffice_path:
        logger.warning("LibreOffice not found on system")
        return None

    # Create temp directory for conversion
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, filename)

        # Write input file
        with open(input_path, 'wb') as f:
            f.write(file_bytes)

        # Convert to PDF using LibreOffice
        try:
            result = subprocess.run([
                soffice_path,
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', tmpdir,
                input_path
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"LibreOffice conversion failed: {result.stderr}")
                return None

            # Read converted PDF
            pdf_filename = filename.rsplit('.', 1)[0] + '.pdf'
            pdf_path = os.path.join(tmpdir, pdf_filename)

            if not os.path.exists(pdf_path):
                logger.error(f"PDF output not found: {pdf_path}")
                return None

            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            logger.info(f"✅ Converted {filename} to PDF using LibreOffice ({len(pdf_bytes)} bytes)")
            return pdf_bytes

        except subprocess.TimeoutExpired:
            logger.error(f"LibreOffice conversion timeout for {filename}")
            return None
        except Exception as e:
            logger.error(f"LibreOffice conversion error: {e}")
            return None


def _extract_text_as_pdf(file_bytes: bytes, ext: str) -> Optional[bytes]:
    """
    Extract text from Office file and create simple PDF
    Fallback method when LibreOffice is not available
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        text_content = None

        # Extract text based on file type
        if ext in ['pptx', 'ppt']:
            try:
                from pptx import Presentation
                prs = Presentation(io.BytesIO(file_bytes))
                slides_text = []
                for i, slide in enumerate(prs.slides):
                    slide_text = f"Slide {i+1}:\n"
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            slide_text += shape.text + "\n"
                    slides_text.append(slide_text)
                text_content = "\n\n".join(slides_text)
            except ImportError:
                logger.error("python-pptx not installed")
                return None

        elif ext in ['docx', 'doc']:
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_bytes))
                paragraphs = [para.text for para in doc.paragraphs]
                text_content = "\n\n".join(paragraphs)
            except ImportError:
                logger.error("python-docx not installed")
                return None

        elif ext in ['xlsx', 'xls']:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(file_bytes))
                sheets_text = []
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    sheet_text = f"Sheet: {sheet_name}\n"
                    for row in sheet.iter_rows(values_only=True):
                        row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
                        sheet_text += row_text + "\n"
                    sheets_text.append(sheet_text)
                text_content = "\n\n".join(sheets_text)
            except ImportError:
                logger.error("openpyxl not installed")
                return None

        if not text_content:
            return None

        # Create PDF from text
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Split text into paragraphs and add to PDF
        for para_text in text_content.split('\n\n'):
            if para_text.strip():
                para = Paragraph(para_text.replace('\n', '<br/>'), styles['Normal'])
                story.append(para)
                story.append(Spacer(1, 0.2*inch))

        doc.build(story)
        pdf_bytes = pdf_buffer.getvalue()

        logger.info(f"✅ Extracted text and created PDF ({len(pdf_bytes)} bytes)")
        return pdf_bytes

    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return None


def needs_conversion(filename: str) -> bool:
    """Check if file needs PDF conversion"""
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    return ext in ['pptx', 'docx', 'xlsx', 'ppt', 'doc', 'xls', 'rtf']
