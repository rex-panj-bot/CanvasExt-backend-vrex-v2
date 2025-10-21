"""
Document Manager
Handles full document text extraction and material catalog management
Provides tools for the agentic system to load complete documents
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from PyPDF2 import PdfReader
import re


class DocumentManager:
    def __init__(self, upload_dir: str = "./uploads"):
        """
        Initialize Document Manager

        Args:
            upload_dir: Directory where PDFs are stored
        """
        self.upload_dir = Path(upload_dir)

        # Build material catalog on init
        self.catalog = self._build_catalog()

    def _build_catalog(self, quick_mode: bool = False) -> Dict[str, List[Dict]]:
        """
        Build a catalog of all available materials

        Args:
            quick_mode: If True, skip page counting for faster catalog build

        Returns: {course_id: [material_metadata]}
        """
        catalog = {}

        if not self.upload_dir.exists():
            return catalog

        # Group files by course_id (extracted from filename prefix)
        for pdf_path in self.upload_dir.glob("*.pdf"):
            filename = pdf_path.name

            # Extract course_id from filename (format: {course_id}_{original_name}.pdf)
            parts = filename.split('_', 1)
            if len(parts) < 2:
                continue

            course_id = parts[0]
            original_name = parts[1].replace('.pdf', '')

            if course_id not in catalog:
                catalog[course_id] = []

            # Get file size
            size_mb = pdf_path.stat().st_size / (1024 * 1024)

            # Try to count pages (skip in quick mode for faster catalog build)
            num_pages = None
            if not quick_mode:
                try:
                    reader = PdfReader(str(pdf_path))
                    num_pages = len(reader.pages)
                except:
                    num_pages = None

            catalog[course_id].append({
                "id": filename.replace('.pdf', ''),
                "name": original_name,
                "filename": filename,
                "path": str(pdf_path),
                "size_mb": round(size_mb, 2),
                "num_pages": num_pages,
                "type": self._infer_type(original_name)
            })

        print(f"📚 Built catalog: {sum(len(docs) for docs in catalog.values())} documents across {len(catalog)} courses")
        return catalog

    def add_files_to_catalog(self, file_paths: List[str]):
        """
        Incrementally add new files to catalog without full rescan

        Args:
            file_paths: List of new file paths to add
        """
        for file_path in file_paths:
            pdf_path = Path(file_path)
            filename = pdf_path.name

            # Extract course_id from filename
            parts = filename.split('_', 1)
            if len(parts) < 2:
                continue

            course_id = parts[0]
            original_name = parts[1].replace('.pdf', '')

            if course_id not in self.catalog:
                self.catalog[course_id] = []

            # Check if already in catalog
            if any(doc['id'] == filename.replace('.pdf', '') for doc in self.catalog[course_id]):
                continue  # Skip duplicates

            # Get file size
            size_mb = pdf_path.stat().st_size / (1024 * 1024)

            # Add to catalog (no page counting for speed)
            self.catalog[course_id].append({
                "id": filename.replace('.pdf', ''),
                "name": original_name,
                "filename": filename,
                "path": str(pdf_path),
                "size_mb": round(size_mb, 2),
                "num_pages": None,  # Skip for speed
                "type": self._infer_type(original_name)
            })

        print(f"➕ Added {len(file_paths)} files to catalog")

    def _infer_type(self, name: str) -> str:
        """Infer document type from filename"""
        name_lower = name.lower()

        if 'hw' in name_lower or 'homework' in name_lower:
            return 'homework'
        elif 'lecture' in name_lower or 'notes' in name_lower or 'pg' in name_lower:
            return 'lecture_notes'
        elif 'syllabus' in name_lower:
            return 'syllabus'
        elif 'exam' in name_lower or 'test' in name_lower:
            return 'exam'
        elif 'solution' in name_lower:
            return 'solutions'
        else:
            return 'document'

    def get_material_catalog(self, course_id: str) -> Dict:
        """
        Get catalog of all materials for a course

        Args:
            course_id: Course identifier

        Returns:
            Dict with materials list and summary stats
        """
        materials = self.catalog.get(course_id, [])

        # Calculate stats
        total_size = sum(m['size_mb'] for m in materials)
        total_pages = sum(m['num_pages'] for m in materials if m['num_pages'])

        # Group by type
        by_type = {}
        for material in materials:
            mat_type = material['type']
            if mat_type not in by_type:
                by_type[mat_type] = []
            by_type[mat_type].append(material)

        return {
            "course_id": course_id,
            "total_documents": len(materials),
            "total_size_mb": round(total_size, 2),
            "total_pages": total_pages,
            "by_type": {k: len(v) for k, v in by_type.items()},
            "materials": materials
        }

    def search_materials(self, course_id: str, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search materials by name/metadata

        Args:
            course_id: Course identifier
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of matching materials with relevance scores
        """
        materials = self.catalog.get(course_id, [])
        query_lower = query.lower()

        # Score materials
        scored = []
        for material in materials:
            score = 0
            name_lower = material['name'].lower()

            # Exact match
            if query_lower == name_lower:
                score += 20
            # Contains query
            elif query_lower in name_lower:
                score += 10
            # Word match
            elif any(word == query_lower for word in name_lower.split()):
                score += 5
            # Type match
            if query_lower in material['type']:
                score += 3

            if score > 0:
                scored.append({
                    **material,
                    "relevance_score": score
                })

        # Sort by score and limit results
        scored.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored[:max_results]


    def get_full_document_text(self, course_id: str, doc_id: str) -> Optional[Dict]:
        """
        Extract full text from a PDF document

        Args:
            course_id: Course identifier
            doc_id: Document ID (filename without .pdf)

        Returns:
            Dict with full_text, metadata, and extraction method
        """
        # Find the document
        materials = self.catalog.get(course_id, [])
        document = None
        for mat in materials:
            if mat['id'] == doc_id:
                document = mat
                break

        if not document:
            return {
                "error": f"Document not found: {doc_id}",
                "course_id": course_id,
                "doc_id": doc_id
            }

        pdf_path = Path(document['path'])
        if not pdf_path.exists():
            return {
                "error": f"File not found: {pdf_path}",
                "course_id": course_id,
                "doc_id": doc_id
            }

        # Extract text using PyPDF2
        extraction_method = "PyPDF2"
        full_text = ""

        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    full_text += f"\n[Page {page_num + 1}]\n"
                    full_text += page_text

        except Exception as e:
            return {
                "error": f"Failed to extract text: {str(e)}",
                "course_id": course_id,
                "doc_id": doc_id
            }

        if not full_text.strip():
            return {
                "error": "No text could be extracted (may be image-only PDF)",
                "course_id": course_id,
                "doc_id": doc_id
            }

        # Clean and format text
        full_text = self._clean_text(full_text)

        return {
            "success": True,
            "doc_id": doc_id,
            "name": document['name'],
            "full_text": full_text,
            "char_count": len(full_text),
            "word_count": len(full_text.split()),
            "num_pages": document['num_pages'],
            "extraction_method": extraction_method,
            "metadata": {
                "filename": document['filename'],
                "type": document['type'],
                "size_mb": document['size_mb']
            }
        }

    def get_multiple_documents(self, course_id: str, doc_ids: List[str]) -> Dict:
        """
        Get full text for multiple documents (batched)

        Args:
            course_id: Course identifier
            doc_ids: List of document IDs

        Returns:
            Dict with documents list and combined stats
        """
        documents = []
        total_chars = 0
        failed = []

        for doc_id in doc_ids:
            result = self.get_full_document_text(course_id, doc_id)

            if result.get("success"):
                documents.append(result)
                total_chars += result['char_count']
            else:
                failed.append({
                    "doc_id": doc_id,
                    "error": result.get("error", "Unknown error")
                })

        # Estimate size in MB (rough)
        estimated_mb = total_chars / (1024 * 1024)

        return {
            "success": True,
            "course_id": course_id,
            "loaded_count": len(documents),
            "failed_count": len(failed),
            "total_chars": total_chars,
            "estimated_mb": round(estimated_mb, 2),
            "documents": documents,
            "failed": failed if failed else None
        }

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Remove common OCR artifacts
        text = text.replace('\x00', '')

        return text.strip()

    def get_document_summary(self, course_id: str, doc_id: str) -> Optional[Dict]:
        """
        Get summary/metadata for a document without loading full text

        Args:
            course_id: Course identifier
            doc_id: Document ID

        Returns:
            Document metadata
        """
        materials = self.catalog.get(course_id, [])

        for material in materials:
            if material['id'] == doc_id:
                return material

        return None

    def find_syllabus(self, course_id: str) -> Optional[str]:
        """
        Auto-detect syllabus document for a course

        Args:
            course_id: Course identifier

        Returns:
            Document ID of syllabus if found, None otherwise
        """
        materials = self.catalog.get(course_id, [])

        # Check for exact type match first
        for material in materials:
            if material.get('type') == 'syllabus':
                return material['id']

        # Check for keyword in name (case-insensitive)
        for material in materials:
            name_lower = material['name'].lower()
            if 'syllabus' in name_lower or 'syllabi' in name_lower:
                return material['id']

        return None
