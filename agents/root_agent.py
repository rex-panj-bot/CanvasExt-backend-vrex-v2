"""
Root Agent - Query Processing for Study Assistant

Uses Gemini 2.5 Pro with native PDF processing for optimal performance.
"""

from google import genai
from google.genai import types
from typing import List, Dict, AsyncGenerator
from utils.file_upload_manager import FileUploadManager


class RootAgent:
    def __init__(self, document_manager, google_api_key: str):
        """
        Initialize Root Agent with Gemini 2.5 Pro

        Args:
            document_manager: DocumentManager instance for document catalog
            google_api_key: Google API key for Gemini
        """
        self.document_manager = document_manager

        # Initialize Gemini client
        self.client = genai.Client(api_key=google_api_key)
        self.model_id = "gemini-2.5-flash"

        # Initialize File Upload Manager
        self.file_upload_manager = FileUploadManager(self.client, cache_duration_hours=48)

        # Session tracking: {session_id: {doc_ids: set(), file_uris: list()}}
        self.session_uploads = {}

    def clear_session(self, session_id: str):
        """Clear uploaded files for a session"""
        if session_id in self.session_uploads:
            del self.session_uploads[session_id]
            print(f"🗑️  Cleared session: {session_id}")

    async def process_query_stream(
        self,
        course_id: str,
        user_message: str,
        conversation_history: List[Dict],
        selected_docs: List[str] = None,
        syllabus_id: str = None,
        session_id: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Process user query with streaming response

        Args:
            course_id: Course identifier
            user_message: User's question
            conversation_history: Previous conversation messages
            selected_docs: List of selected document IDs to process
            syllabus_id: Optional syllabus document ID (always included)
            session_id: WebSocket session ID for tracking uploads

        Yields:
            Response chunks
        """
        try:
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: Starting query processing")
            print(f"   Course ID: {course_id}")
            print(f"   Session ID: {session_id}")
            print(f"   Conversation history length: {len(conversation_history)}")

            # Step 1: Get all available documents
            catalog = self.document_manager.get_material_catalog(course_id)
            all_materials = catalog.get("materials", [])
            print(f"   📂 Total materials in catalog: {len(all_materials)}")

            if not all_materials:
                yield "No course materials found. Please upload PDFs first."
                return

            # Step 2: Filter based on selection
            materials_to_use = []

            print(f"   📋 Selected docs from client: {selected_docs}")
            print(f"   📋 Number of selected docs: {len(selected_docs) if selected_docs else 0}")

            if selected_docs:
                # Debug: Show all material IDs available
                available_ids = [m["id"] for m in all_materials]
                print(f"   🔑 Available material IDs: {available_ids[:5]}..." if len(available_ids) > 5 else f"   🔑 Available material IDs: {available_ids}")

                materials_to_use = [m for m in all_materials if m["id"] in selected_docs]
                print(f"   ✅ Matched {len(materials_to_use)} materials from selection")

                if len(materials_to_use) < len(selected_docs):
                    print(f"   ⚠️  WARNING: {len(selected_docs) - len(materials_to_use)} selected docs not found in catalog!")
                    missing = set(selected_docs) - set(m["id"] for m in materials_to_use)
                    print(f"   ⚠️  Missing IDs: {list(missing)[:3]}...")

                # Always include syllabus if provided and not already selected
                if syllabus_id and syllabus_id not in selected_docs:
                    syllabus = next((m for m in all_materials if m["id"] == syllabus_id), None)
                    if syllabus:
                        materials_to_use.append(syllabus)
                        print(f"   ⭐ Including syllabus: {syllabus['name']}")
            else:
                print(f"   ⚠️  No docs selected - using ALL materials")
                materials_to_use = all_materials

            if not materials_to_use:
                print(f"   ❌ No materials to use after filtering!")
                yield "No documents selected. Please select at least one document."
                return

            print(f"   📚 Final materials to use: {len(materials_to_use)} PDF files")
            print(f"   📚 Material names: {[m['name'][:30] for m in materials_to_use[:3]]}...")

            # Step 3: Check if files already uploaded in this session
            selected_doc_ids = set([m["id"] for m in materials_to_use])
            session_cache = self.session_uploads.get(session_id, {})
            cached_doc_ids = session_cache.get("doc_ids", set())

            print(f"   🔍 Session check:")
            print(f"      Session ID: {session_id}")
            print(f"      Selected doc IDs: {list(selected_doc_ids)[:3]}...")
            print(f"      Cached doc IDs: {list(cached_doc_ids)[:3] if cached_doc_ids else 'None'}...")
            print(f"      IDs match: {cached_doc_ids == selected_doc_ids}")

            need_upload = False
            uploaded_files = []

            if session_id and cached_doc_ids == selected_doc_ids:
                # Same files already uploaded in this session - reuse
                print(f"   ✅ Reusing {len(materials_to_use)} files from session cache")
                uploaded_files = session_cache.get("file_info", [])
                print(f"   ✅ Retrieved {len(uploaded_files)} file URIs from cache")
            else:
                # Need to upload (new session or different file selection)
                need_upload = True
                print(f"   📤 Uploading {len(materials_to_use)} PDFs to Gemini...")

                file_paths = [mat["path"] for mat in materials_to_use]
                print(f"   📂 File paths to upload: {file_paths[:2]}...")
                upload_result = await self.file_upload_manager.upload_multiple_pdfs_async(file_paths)

                if not upload_result.get('success'):
                    error_msg = upload_result.get('error', 'Unknown error')
                    print(f"   ❌ Upload failed: {error_msg}")
                    yield f"❌ Error uploading files: {error_msg}"
                    return

                uploaded_files = upload_result.get('files', [])
                total_mb = upload_result['total_bytes'] / (1024 * 1024)

                print(f"   ✅ Uploaded {len(uploaded_files)} PDFs (~{total_mb:.1f}MB)")
                print(f"   ✅ File URIs: {[f['uri'][:50] + '...' for f in uploaded_files[:2]]}")

                # Cache for this session
                if session_id:
                    self.session_uploads[session_id] = {
                        "doc_ids": selected_doc_ids,
                        "file_info": uploaded_files
                    }
                    print(f"   💾 Cached {len(uploaded_files)} files for session")

                # Yield upload status to user
                yield f"📤 Loaded {len(uploaded_files)} PDFs (~{total_mb:.1f}MB)\n\n"

            # Step 4: Build system instruction
            file_names = [mat['name'] for mat in materials_to_use]
            system_instruction = f"""You are an AI study assistant with access to {len(uploaded_files)} course PDF documents.

Available materials: {', '.join(file_names[:10])}{"..." if len(file_names) > 10 else ""}

Read the PDFs directly to answer questions.

IMPORTANT CITATION FORMAT:
When referencing information from documents, use this exact format:
[Source: DocumentName, Page X]

Examples:
- According to the lecture notes [Source: Lecture_3_Algorithms, Page 12], sorting algorithms...
- The syllabus states [Source: CS101_Syllabus, Page 3] that exams are worth 40%.

Always cite specific pages when making factual claims from the documents. Use the exact document name without .pdf extension."""

            # Step 5: Build conversation with file references
            contents = []

            # Add conversation history (text only, no files)
            for msg in conversation_history[-4:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

            # Add current message
            parts = []

            # Always attach files when documents are selected (Gemini requires file references on every call)
            print(f"   📎 Attaching files to API call:")
            print(f"      uploaded_files count: {len(uploaded_files)}")
            if uploaded_files:
                # Attach PDF file URIs
                for file_info in uploaded_files:
                    parts.append(types.Part(file_data=types.FileData(file_uri=file_info['uri'])))
                print(f"      ✅ Attached {len(uploaded_files)} file URIs to message")
                print(f"      Sample URIs: {[f['uri'][:60] + '...' for f in uploaded_files[:2]]}")
            else:
                print(f"      ⚠️  WARNING: No uploaded_files to attach!")

            # Add user question
            parts.append(types.Part(text=user_message))
            contents.append(types.Content(role="user", parts=parts))

            print(f"   📨 Message parts: {len(parts)} parts total (files + text)")
            print(f"   🤖 Calling Gemini API now...")
            print(f"      Model: {self.model_id}")
            print(f"      Contents length: {len(contents)}")
            print(f"      History messages: {len(conversation_history)}")

            # Step 6: Stream response from Gemini
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=8192,
                )
            )

            # Step 7: Stream response chunks
            total_generated = 0
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    total_generated += len(chunk.text)

            print(f"   ✅ Complete ({total_generated} chars generated)")

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            yield f"\n\n*{error_msg}*"
