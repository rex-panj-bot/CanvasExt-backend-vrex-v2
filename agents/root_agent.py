"""
Root Agent - Query Processing for Study Assistant

Uses Gemini 2.5 Flash with native PDF processing for optimal performance.
"""

from google import genai
from google.genai import types
from typing import List, Dict, AsyncGenerator, Optional
from utils.file_upload_manager import FileUploadManager
from agents.file_selector_agent import FileSelectorAgent
import asyncio


class RootAgent:
    # REMOVED: strip_extension_for_matching() - no longer needed with hash-based IDs
    # Hash-based doc_ids are immutable and identical between frontend and backend

    def __init__(self, document_manager, google_api_key: str, storage_manager=None, chat_storage=None):
        """
        Initialize Root Agent with Gemini 2.5 Flash

        Args:
            document_manager: DocumentManager instance for document catalog
            google_api_key: Google API key for Gemini
            storage_manager: Optional StorageManager for GCS file access
            chat_storage: Optional ChatStorage for file summaries access
        """
        self.document_manager = document_manager
        self.storage_manager = storage_manager
        self.chat_storage = chat_storage

        # Initialize Gemini client
        self.client = genai.Client(api_key=google_api_key)
        self.model_id = "gemini-2.5-flash"
        self.fallback_model = "gemini-2.0-flash"  # Fallback for rate limits/503
        self.fallback_model_2 = "gemini-2.0-flash-lite"  # Second fallback tier

        # Initialize File Upload Manager
        self.file_upload_manager = FileUploadManager(
            self.client,
            cache_duration_hours=48,
            storage_manager=storage_manager
        )

        # Initialize File Selector Agent
        self.file_selector_agent = FileSelectorAgent(google_api_key=google_api_key)

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
        session_id: str = None,
        enable_web_search: bool = False,
        user_api_key: str = None,
        use_smart_selection: bool = False,
        stop_check_callback = None
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
            enable_web_search: Whether to enable Google Search grounding
            user_api_key: Optional user-provided Gemini API key (overrides default)
            use_smart_selection: Whether to use AI-powered intelligent file selection

        Yields:
            Response chunks
        """
        try:
            # Use user's API key if provided, otherwise use default
            api_client = self.client
            file_upload_manager = self.file_upload_manager

            if user_api_key:
                api_client = genai.Client(api_key=user_api_key)
                # Create new FileUploadManager with user's API key
                # Files must be uploaded with same key used for generation
                file_upload_manager = FileUploadManager(
                    api_client,
                    cache_duration_hours=48,
                    storage_manager=self.storage_manager
                )
                print(f"🔑 Using user-provided API key")

            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: Starting query processing")
            print(f"   Course ID: {course_id}")
            print(f"   Session ID: {session_id}")
            print(f"   Web Search: {'Enabled' if enable_web_search else 'Disabled'}")
            print(f"   Smart Selection: {'Enabled' if use_smart_selection else 'Disabled'}")
            print(f"   Conversation history length: {len(conversation_history)}")

            # Step 1: Get all available documents
            catalog = self.document_manager.get_material_catalog(course_id)
            all_materials = catalog.get("materials", [])
            print(f"   📂 Total materials in catalog: {len(all_materials)}")

            # Enrich materials with canvas_id from database (for fallback matching)
            # Priority: catalog canvas_id > summary canvas_id
            if self.chat_storage:
                for material in all_materials:
                    # Skip if material already has canvas_id from catalog
                    if material.get("canvas_id"):
                        continue

                    doc_id = material.get("id")
                    if doc_id:
                        file_summary = self.chat_storage.get_file_summary(doc_id)
                        if file_summary and file_summary.get("canvas_id"):
                            material["canvas_id"] = file_summary["canvas_id"]

            if not all_materials:
                yield "No course materials found. Please upload PDFs first."
                return

            # Step 2: Filter based on selection (manual or AI-powered)
            materials_to_use = []

            # ANCHOR CLUSTER: Smart selection with Global Discovery or Scoped Refinement
            if use_smart_selection and self.chat_storage:
                print(f"\n   🤖 ANCHOR CLUSTER SMART SELECTION ENABLED")
                print(f"   ❓ User query: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
                print(f"   📋 Manual selection provided: {len(selected_docs) if selected_docs else 0} files")

                # Get file summaries from database
                file_summaries = self.chat_storage.get_all_summaries_for_course(course_id)
                print(f"   📚 Found {len(file_summaries)} file summaries")

                if not file_summaries:
                    print(f"   ⚠️  No summaries available, falling back to manual selection")
                    yield "\n⚠️ No file summaries available. Using all materials.\n"
                    materials_to_use = all_materials
                else:
                    # ALWAYS fetch syllabus for ground truth (needed for both scenarios)
                    # Try to get stored syllabus_id from database first
                    if not syllabus_id and self.chat_storage:
                        syllabus_id = self.chat_storage.get_course_syllabus(course_id)
                        if syllabus_id:
                            print(f"      ✅ Found stored syllabus_id in database: {syllabus_id}")

                    syllabus_summary = None
                    if syllabus_id:
                        print(f"      Fetching syllabus for ground truth: {syllabus_id}")
                        syllabus_summary = await self.file_selector_agent.get_syllabus_summary(
                            syllabus_id, file_summaries
                        )
                        if syllabus_summary:
                            print(f"   ✅ Syllabus found ({len(syllabus_summary)} chars)")
                        else:
                            print(f"   ⚠️  Syllabus not found")
                    else:
                        print(f"      No syllabus_id stored, searching for 'syllabus' in filenames...")
                        syllabus_summary = await self.file_selector_agent.get_syllabus_summary(
                            None, file_summaries
                        )

                    # Determine max_files based on user query or use default
                    import re
                    max_files = 15  # Default

                    # Check if user specifies a number in their query
                    number_patterns = [
                        r'at least (\d+)',
                        r'give me (\d+)',
                        r'(\d+) files',
                        r'(\d+) sources'
                    ]
                    for pattern in number_patterns:
                        match = re.search(pattern, user_message.lower())
                        if match:
                            requested = int(match.group(1))
                            max_files = min(requested, 30)  # Cap at 30
                            print(f"   📊 User requested {requested} files, using max_files={max_files}")
                            break

                    # Route to Anchor Cluster file selector
                    # Pass selected_docs if user made manual selection (Scoped Refinement)
                    # Pass None if no selection (Global Discovery)
                    yield "📋 Analyzing your question and selecting relevant materials..."
                    selected_files = await self.file_selector_agent.select_relevant_files(
                        user_query=user_message,
                        file_summaries=file_summaries,
                        syllabus_summary=syllabus_summary,
                        syllabus_doc_id=syllabus_id,
                        selected_docs=selected_docs,  # NEW: Pass user's manual selection
                        max_files=max_files
                    )

                    if not selected_files:
                        print(f"   ⚠️  File selector returned no files, using all materials")
                        yield "\n⚠️ Could not determine relevant files. Using all materials.\n"
                        materials_to_use = all_materials
                    else:
                        # Get the actual materials based on selected doc_ids
                        selected_doc_ids = [f.get("doc_id") for f in selected_files]
                        print(f"   🔍 Selected doc IDs from Anchor Cluster: {selected_doc_ids[:5]}...")
                        materials_to_use = [m for m in all_materials if m["id"] in selected_doc_ids]
                        print(f"   🔍 Matched {len(materials_to_use)} materials from {len(selected_doc_ids)} selected IDs")

                        # Always include syllabus if available (anchor doc)
                        if syllabus_id and syllabus_id not in selected_doc_ids:
                            syllabus = next((m for m in all_materials if m["id"] == syllabus_id), None)
                            if syllabus:
                                materials_to_use.append(syllabus)
                                print(f"   📌 Added syllabus as anchor doc")

                        # Show selected files to user
                        file_names = [f.get("filename", "unknown") for f in selected_files[:3]]
                        if selected_docs:
                            # Scoped Refinement - show pruning
                            yield f"\n✅ Pruned selection: {len(selected_docs)} → {len(materials_to_use)} relevant files\n"
                            yield f"   Files: {', '.join(file_names)}{'...' if len(file_names) > 3 else ''}\n\n"
                        else:
                            # Global Discovery - show selection
                            yield f"\n✅ Selected {len(materials_to_use)} relevant files: {', '.join(file_names)}{'...' if len(file_names) > 3 else ''}\n\n"

                        print(f"   ✅ Anchor Cluster selected {len(materials_to_use)} files:")
                        for file in selected_files[:5]:
                            print(f"      📄 {file.get('filename')}")

            # Manual selection (original behavior)
            elif selected_docs:
                print(f"   📋 Using manual selection")
                print(f"   📋 Selected docs from client: {selected_docs}")
                print(f"   📋 Number of selected docs: {len(selected_docs)}")

                # Debug: Show all material IDs available
                available_ids = [m["id"] for m in all_materials]
                print(f"   🔑 Available material IDs ({len(available_ids)}):")
                for avail_id in available_ids[:10]:  # Show first 10
                    print(f"      - {avail_id}")

                print(f"\n   📋 Selected doc IDs from frontend ({len(selected_docs)}):")
                for sel_id in selected_docs[:10]:  # Show first 10
                    print(f"      - {sel_id}")

                # HASH-BASED: Direct ID matching - no fuzzy logic needed
                # Frontend sends exact doc_id ({course_id}_{hash}) which matches catalog
                # Build a set of available IDs for O(1) lookup
                available_id_set = {material["id"] for material in all_materials}

                # Build canvas_id lookup map for fallback matching
                # Key by BOTH string and numeric versions for flexible matching
                canvas_id_map = {}
                for material in all_materials:
                    if material.get("canvas_id"):
                        # Store as string key
                        canvas_id_map[str(material["canvas_id"])] = material
                        # Also store numeric version if it's a number
                        try:
                            numeric_id = int(material["canvas_id"])
                            canvas_id_map[numeric_id] = material
                        except (ValueError, TypeError):
                            pass

                # Match selected docs using exact ID comparison
                materials_to_use = []
                matched_by_canvas_id = []
                for selected_id in selected_docs:
                    if selected_id in available_id_set:
                        # Direct match by doc_id - find and add the material
                        matched_material = next(m for m in all_materials if m["id"] == selected_id)
                        materials_to_use.append(matched_material)
                    elif selected_id in canvas_id_map:
                        # Fallback: match by Canvas ID (for legacy materials)
                        matched_material = canvas_id_map[selected_id]
                        materials_to_use.append(matched_material)
                        matched_by_canvas_id.append(selected_id)
                        print(f"   🔄 Matched by Canvas ID: {selected_id} → {matched_material['name']}")
                    elif str(selected_id) in canvas_id_map:
                        # Try string version if numeric didn't match
                        matched_material = canvas_id_map[str(selected_id)]
                        materials_to_use.append(matched_material)
                        matched_by_canvas_id.append(selected_id)
                        print(f"   🔄 Matched by Canvas ID (str): {selected_id} → {matched_material['name']}")

                print(f"\n   ✅ Matched {len(materials_to_use)} materials from selection")
                if matched_by_canvas_id:
                    print(f"   📌 {len(matched_by_canvas_id)} matched using fallback Canvas ID matching")

                if len(materials_to_use) < len(selected_docs):
                    print(f"   ⚠️  WARNING: {len(selected_docs) - len(materials_to_use)} selected docs not found in catalog!")
                    matched_ids = {m["id"] for m in materials_to_use}
                    matched_canvas_ids = {m.get("canvas_id") for m in materials_to_use if m.get("canvas_id")}
                    missing = [sid for sid in selected_docs if sid not in available_id_set and sid not in canvas_id_map]
                    print(f"   ⚠️  Missing IDs:")
                    for miss_id in missing[:10]:  # Show first 10
                        print(f"      - {miss_id}")

                # Always include syllabus if provided and not already selected
                if syllabus_id and syllabus_id not in selected_docs:
                    syllabus = next((m for m in all_materials if m["id"] == syllabus_id), None)
                    if syllabus:
                        materials_to_use.append(syllabus)
                        print(f"   ⭐ Including syllabus: {syllabus['name']}")

            # No selection - act like regular Gemini (no files)
            else:
                print(f"   ℹ️  No docs selected and smart selection off - using no materials (regular Gemini mode)")
                materials_to_use = []

            # Allow empty materials when user wants regular Gemini mode (no selection + no smart select)
            if not materials_to_use:
                # Check if this was intentional (no selection + smart select off)
                if not use_smart_selection and not selected_docs:
                    print(f"   ℹ️  Using Gemini in chat-only mode (no files)")
                    # Skip to chat generation without files
                else:
                    print(f"   ❌ No materials to use after filtering!")
                    yield "No documents selected. Please select at least one document or enable Smart Selection."
                    return

            print(f"   📚 Final materials to use: {len(materials_to_use)} PDF files")
            if materials_to_use:
                print(f"   📚 Material names: {[m['name'][:30] for m in materials_to_use[:3]]}...")

            # Step 3: Upload files to Gemini (skip if no materials selected)
            need_upload = False
            uploaded_files = []

            if materials_to_use:
                selected_doc_ids = set([m["id"] for m in materials_to_use])
                session_cache = self.session_uploads.get(session_id, {})
                cached_doc_ids = session_cache.get("doc_ids", set())

                print(f"   🔍 Session check:")
                print(f"      Session ID: {session_id}")
                print(f"      Selected doc IDs ({len(selected_doc_ids)}): {list(selected_doc_ids)[:3]}...")
                print(f"      Cached doc IDs ({len(cached_doc_ids)}): {list(cached_doc_ids)[:3] if cached_doc_ids else 'None'}...")
                print(f"      IDs match: {cached_doc_ids == selected_doc_ids}")

                # Verify all materials have paths
                materials_with_paths = [m for m in materials_to_use if m.get("path")]
                if len(materials_with_paths) < len(materials_to_use):
                    print(f"   ⚠️  WARNING: {len(materials_to_use) - len(materials_with_paths)} materials missing paths!")
                    missing_path_materials = [m for m in materials_to_use if not m.get("path")]
                    for mat in missing_path_materials[:3]:
                        print(f"      - {mat.get('name', 'unknown')}: id={mat.get('id', 'unknown')}")

                print(f"   📂 Materials to upload: {len(materials_with_paths)} with valid paths")

                if session_id and cached_doc_ids == selected_doc_ids:
                    # Same files already uploaded in this session - reuse
                    print(f"   ✅ Reusing {len(materials_to_use)} files from session cache")
                    uploaded_files = session_cache.get("file_info", [])
                    print(f"   ✅ Retrieved {len(uploaded_files)} file URIs from cache")
                else:
                    # Need to upload (new session or different file selection)
                    need_upload = True
                    print(f"   📤 Uploading {len(materials_to_use)} files to Gemini...")

                    file_paths = [mat["path"] for mat in materials_to_use if mat.get("path")]
                    print(f"   📂 Files with paths: {len(file_paths)}/{len(materials_to_use)}")
                    print(f"   📂 Sample paths: {file_paths[:2]}...")

                    if not file_paths:
                        print(f"   ❌ ERROR: No valid file paths found!")
                        yield "❌ Error: No valid file paths found in materials. Please re-scan course materials.\n\n"
                        return

                    upload_result = await file_upload_manager.upload_multiple_pdfs_async(file_paths)

                    if not upload_result.get('success'):
                        error_msg = upload_result.get('error', 'Unknown error')
                        print(f"   ❌ Upload failed: {error_msg}")
                        yield f"❌ Error uploading files: {error_msg}"
                        return

                    uploaded_files = upload_result.get('files', [])
                    failed_files = upload_result.get('failed', [])
                    total_mb = upload_result['total_bytes'] / (1024 * 1024)

                    print(f"   ✅ Uploaded {len(uploaded_files)} files (~{total_mb:.1f}MB)")
                    if failed_files:
                        print(f"   ⚠️  Failed to upload {len(failed_files)} files:")
                        for failed in failed_files:
                            print(f"      - {failed.get('path', 'unknown')}: {failed.get('error', 'unknown error')}")
                    print(f"   ✅ File URIs: {[f['uri'][:50] + '...' for f in uploaded_files[:2] if f.get('uri')]}")

                    # Inform user if no files could be uploaded
                    if len(uploaded_files) == 0:
                        yield "⚠️ **No files could be uploaded to Gemini.**\n\n"
                        if failed_files:
                            yield f"**Failed uploads ({len(failed_files)} files):**\n"
                            for failed in failed_files[:5]:  # Show first 5
                                path = failed.get('path', 'unknown')
                                filename = path.split('/')[-1] if '/' in path else path
                                error = failed.get('error', 'unknown error')
                                yield f"- {filename}: {error}\n"
                            if len(failed_files) > 5:
                                yield f"- ... and {len(failed_files) - 5} more\n"
                        yield "\n**Supported formats**: PDF, TXT, MD, CSV, PNG, JPG, JPEG, GIF, WEBP\n"
                        yield "**Tip**: Convert PPTX, DOCX, XLSX files to PDF for best results\n\n"
                        return

                    # Cache for this session
                    if session_id:
                        self.session_uploads[session_id] = {
                            "doc_ids": selected_doc_ids,
                            "file_info": uploaded_files
                        }
                        print(f"   💾 Cached {len(uploaded_files)} files for session")

                    # Yield upload status to user
                    status_msg = f"📤 **Loaded {len(uploaded_files)} of {len(materials_to_use)} files** (~{total_mb:.1f}MB)"
                    if failed_files:
                        status_msg += f"\n⚠️ **{len(failed_files)} files could not be uploaded:**\n"
                        for failed in failed_files[:3]:  # Show first 3
                            path = failed.get('path', 'unknown')
                            filename = path.split('/')[-1] if '/' in path else path
                            error = failed.get('error', 'unknown error')
                            # Show short error message
                            short_error = error.split('.')[0] if '.' in error else error
                            status_msg += f"  - {filename}: {short_error}\n"
                        if len(failed_files) > 3:
                            status_msg += f"  - ... and {len(failed_files) - 3} more\n"
                    yield status_msg + "\n"

            # Step 4: Build system instruction
            file_names = [mat['name'] for mat in materials_to_use]

            # Build system instruction based on whether files are loaded
            if materials_to_use:
                # Build capabilities description
                capabilities_text = "1. Read uploaded documents (PDFs, Word docs, images, etc.) directly"
                if enable_web_search:
                    capabilities_text += "\n2. Perform Google searches for current information, news, or topics not in course materials"

                system_instruction = f"""You are an AI study assistant with access to {len(uploaded_files)} course documents{"and real-time web search" if enable_web_search else ""}.

Available course materials: {', '.join(file_names[:10])}{"..." if len(file_names) > 10 else ""}

CAPABILITIES:
{capabilities_text}

CITATION FORMAT - CRITICAL:
You MUST cite sources using this EXACT format (including brackets, comma, and capitalized "Page"):
[Source: DocumentName, Page X]

RULES:
- Always use square brackets: [ ]
- Always include comma after document name
- Always capitalize "Page" or "Pages"
- Page numbers must be digits only
- NO variations allowed (no "p.", "pg.", dashes instead of commas, etc.)

CORRECT examples:
✓ [Source: Lecture_3_Algorithms, Page 12]
✓ [Source: CS101_Syllabus, Page 3]
✓ [Source: Homework_2, Pages 5-7]

INCORRECT examples (DO NOT USE):
✗ [Source: Lecture_3_Algorithms Page 12] (missing comma)
✗ [Source: Lecture_3_Algorithms - Page 12] (dash instead of comma)
✗ [Source: Lecture_3_Algorithms, p. 12] (lowercase "p.")
✗ (Lecture_3_Algorithms, Page 12) (wrong brackets)
{"When using web search results, the sources will be automatically cited below your response." if enable_web_search else ""}

LATEX FORMATTING - CRITICAL:
When writing mathematical expressions, you MUST follow these rules:
1. ALWAYS wrap inline math in single dollar signs: $expression$
2. ALWAYS wrap display math (equations on their own line) in double dollar signs: $$expression$$
3. Use proper delimiter pairs with \\left and \\right:
   - \\left( ... \\right) for parentheses
   - \\left[ ... \\right] for brackets
   - \\left\\{{ ... \\right\\}} for braces
   - NEVER use \\left or \\right without a proper matching delimiter
4. Always balance all braces {{{{ }}}}
5. Common LaTeX commands: \\frac{{numerator}}{{denominator}}, \\sqrt{{x}}, \\mathcal{{L}}

CORRECT examples:
✓ The inverse Laplace transform is $\\mathcal{{L}}^{{-1}}\\left(\\frac{{6}}{{s+2}}\\right) = 6e^{{-2t}}$.
✓ Display equation: $$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}$$
✓ Inline fraction: The derivative is $\\frac{{dy}}{{dt}} = 3t^2$.

INCORRECT examples (DO NOT USE):
✗ \\mathcal{{L}}^{{-1}}\\left\\frac{{6}}{{s+2}}\\right}} (no delimiters, mismatched \\left/\\right)
✗ The answer is \\frac{{1}}{{2}} (no dollar signs)
✗ $$\\left\\frac{{x}}{{y}}\\right$$ (\\left without proper delimiter)

{"PRIORITY: Always prioritize course materials first. Use web search only when information is not available in course materials or when user asks about current events." if enable_web_search else "Focus on providing accurate information from the course materials."}"""
            else:
                # No files - regular Gemini mode
                system_instruction = f"""You are a helpful AI assistant{"with access to real-time web search" if enable_web_search else ""}.

{"CAPABILITIES:\n1. Answer questions on any topic\n2. Perform Google searches for current information, news, or topics when needed\n\nWhen using web search results, the sources will be automatically cited below your response." if enable_web_search else "Provide helpful, accurate, and conversational responses to user questions."}

LATEX FORMATTING - CRITICAL:
When writing mathematical expressions, you MUST follow these rules:
1. ALWAYS wrap inline math in single dollar signs: $expression$
2. ALWAYS wrap display math (equations on their own line) in double dollar signs: $$expression$$
3. Use proper delimiter pairs with \\left and \\right:
   - \\left( ... \\right) for parentheses
   - \\left[ ... \\right] for brackets
   - \\left\\{{ ... \\right\\}} for braces
   - NEVER use \\left or \\right without a proper matching delimiter
4. Always balance all braces {{{{ }}}}
5. Common LaTeX commands: \\frac{{numerator}}{{denominator}}, \\sqrt{{x}}, \\mathcal{{L}}

CORRECT examples:
✓ The inverse Laplace transform is $\\mathcal{{L}}^{{-1}}\\left(\\frac{{6}}{{s+2}}\\right) = 6e^{{-2t}}$.
✓ Display equation: $$x = \\frac{{-b \\pm \\sqrt{{b^2 - 4ac}}}}{{2a}}$$
✓ Inline fraction: The derivative is $\\frac{{dy}}{{dt}} = 3t^2$.

INCORRECT examples (DO NOT USE):
✗ \\mathcal{{L}}^{{-1}}\\left\\frac{{6}}{{s+2}}\\right}} (no delimiters, mismatched \\left/\\right)
✗ The answer is \\frac{{1}}{{2}} (no dollar signs)
✗ $$\\left\\frac{{x}}{{y}}\\right$$ (\\left without proper delimiter)"""

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
            print(f"      materials_to_use: {len(materials_to_use)} files")
            print(f"      uploaded_files: {len(uploaded_files)} files successfully uploaded")

            # Separate text files from regular files
            text_files = []
            file_uri_files = []

            if uploaded_files:
                for file_info in uploaded_files:
                    if file_info.get('is_text'):
                        text_files.append(file_info)
                    else:
                        file_uri_files.append(file_info)

                # Attach file URIs (PDFs, images, etc.)
                if file_uri_files:
                    for file_info in file_uri_files:
                        if file_info.get('uri'):  # Safety check: ensure URI exists
                            parts.append(types.Part(file_data=types.FileData(file_uri=file_info['uri'])))
                        else:
                            print(f"      ⚠️  Skipping file without URI: {file_info.get('display_name', 'unknown')}")
                    print(f"      ✅ Attached {len(file_uri_files)} file URIs (PDFs, images, etc.)")

                # Attach text content inline (assignments, pages)
                if text_files:
                    text_content_parts = []
                    for file_info in text_files:
                        display_name = file_info.get('display_name', file_info.get('name', 'unknown'))
                        text_content = file_info.get('text_content', '')
                        text_content_parts.append(f"**{display_name}:**\n\n{text_content}")

                    # Combine all text files into one text part
                    combined_text = "\n\n---\n\n".join(text_content_parts)
                    parts.append(types.Part(text=f"**Course Materials (Text Files):**\n\n{combined_text}\n\n---\n\n"))
                    print(f"      ✅ Attached {len(text_files)} text files inline (assignments, pages)")

                print(f"      Sample materials:")
                for f in uploaded_files[:3]:
                    display_name = f.get('display_name', 'unknown')
                    if f.get('is_text'):
                        text_len = len(f.get('text_content', ''))
                        print(f"         - {display_name}: [TEXT, {text_len} chars]")
                    else:
                        uri = (f['uri'][:60] + '...') if f.get('uri') else 'None'
                        print(f"         - {display_name}: {uri}")
                if len(uploaded_files) > 3:
                    print(f"         ... and {len(uploaded_files) - 3} more")
            else:
                print(f"      ⚠️  WARNING: No uploaded_files to attach!")
                print(f"      This means NO FILES will be sent to the LLM!")

            # Add user question
            parts.append(types.Part(text=user_message))
            contents.append(types.Content(role="user", parts=parts))

            print(f"   📨 Message parts: {len(parts)} parts total (files + text)")
            print(f"   🤖 Calling Gemini API now...")
            print(f"      Model: {self.model_id}")
            print(f"      Contents length: {len(contents)}")
            print(f"      History messages: {len(conversation_history)}")

            # Step 6: Stream response from Gemini
            # Conditionally enable Google Search based on user preference
            config_params = {
                "system_instruction": system_instruction,
                "temperature": 0.7,
                "max_output_tokens": 16000,  # Increased from 8192 for more detailed answers
            }

            # Only add Google Search tool if enabled
            if enable_web_search:
                config_params["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                print(f"   🌐 Google Search enabled for this query")

            config = types.GenerateContentConfig(**config_params)

            # Try with primary model, fallback to preview if rate limited or 503
            model_to_use = self.model_id
            try:
                response_stream = await api_client.aio.models.generate_content_stream(
                    model=model_to_use,
                    contents=contents,
                    config=config
                )
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit (429), quota, or 503 (model overloaded)
                if ('429' in error_str or 'quota' in error_str or 'rate' in error_str or
                    '503' in error_str or 'overload' in error_str or 'resource_exhausted' in error_str):
                    print(f"⚠️ Rate limited/overloaded on {model_to_use}, falling back to {self.fallback_model}")
                    model_to_use = self.fallback_model
                    # Small delay before retry
                    await asyncio.sleep(2)
                    try:
                        # Retry with fallback model
                        response_stream = await api_client.aio.models.generate_content_stream(
                            model=model_to_use,
                            contents=contents,
                            config=config
                        )
                    except Exception as e2:
                        error_str2 = str(e2).lower()
                        # If fallback also rate limited, try second fallback
                        if ('429' in error_str2 or 'quota' in error_str2 or 'rate' in error_str2 or
                            '503' in error_str2 or 'overload' in error_str2 or 'resource_exhausted' in error_str2):
                            print(f"⚠️ Rate limited/overloaded on {model_to_use}, falling back to {self.fallback_model_2}")
                            model_to_use = self.fallback_model_2
                            await asyncio.sleep(2)
                            # Final retry with second fallback
                            response_stream = await api_client.aio.models.generate_content_stream(
                                model=model_to_use,
                                contents=contents,
                                config=config
                            )
                        else:
                            raise  # Re-raise if not a rate limit error
                else:
                    raise  # Re-raise if not a rate limit/overload error

            # Step 7: Stream response chunks with grounding metadata
            total_generated = 0
            search_results_shown = False

            chunk_num = 0
            async for chunk in response_stream:
                chunk_num += 1
                # Check if user requested to stop
                if stop_check_callback:
                    should_stop = stop_check_callback()
                    if chunk_num % 5 == 0:  # Log every 5 chunks to avoid spam
                        print(f"   📊 Chunk {chunk_num}: stop_check = {should_stop}")
                    if should_stop:
                        print(f"   🛑 Stop requested at chunk {chunk_num}, breaking Gemini stream early")
                        break

                # Stream text response
                if chunk.text:
                    yield chunk.text
                    total_generated += len(chunk.text)

                # Handle grounding metadata (web search results)
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        metadata = candidate.grounding_metadata

                        # Display search results if available
                        if not search_results_shown and hasattr(metadata, 'search_entry_point'):
                            search_entry_point = metadata.search_entry_point
                            if hasattr(search_entry_point, 'rendered_content'):
                                print(f"   🔍 Web search performed")
                                search_results_shown = True

                        # Extract and yield grounding chunks (web sources)
                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                            sources = []
                            for gc in metadata.grounding_chunks:
                                if hasattr(gc, 'web') and gc.web:
                                    web = gc.web
                                    if hasattr(web, 'uri') and hasattr(web, 'title'):
                                        sources.append(f"- [{web.title}]({web.uri})")

                            if sources and not search_results_shown:
                                yield "\n\n**Web Sources:**\n" + "\n".join(sources[:5]) + "\n"
                                print(f"   🌐 Included {len(sources)} web sources")

            print(f"   ✅ Complete ({total_generated} chars generated)")

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            yield f"\n\n*{error_msg}*"
