"""
File Selector Agent
Intelligently selects the most relevant course materials for a given query
"""

from google import genai
from google.genai import types
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class FileSelectorAgent:
    """Selects relevant files based on query and file summaries"""

    def __init__(self, google_api_key: str):
        """
        Initialize File Selector Agent

        Args:
            google_api_key: Google API key for Gemini
        """
        self.client = genai.Client(api_key=google_api_key)
        self.model_id = "gemini-2.5-flash-lite"  # 3-6x cheaper, same 65K token limit

    async def select_relevant_files(
        self,
        user_query: str,
        file_summaries: List[Dict],
        syllabus_summary: Optional[str] = None,
        max_files: int = 5
    ) -> List[Dict]:
        """
        Select the most relevant files for a given query

        Args:
            user_query: The user's question
            file_summaries: List of dicts with doc_id, filename, summary, topics
            syllabus_summary: Optional course syllabus summary for context
            max_files: Maximum number of files to select (default 5)

        Returns:
            List of selected file dicts with relevance scores
        """
        try:
            print(f"\n   🎯 FILE SELECTOR AGENT CALLED")
            print(f"      Max files requested: {max_files}")
            print(f"      Available summaries: {len(file_summaries)}")

            if not file_summaries:
                print(f"      ❌ No file summaries available")
                return []

            print(f"      Query: {user_query[:150]}...")

            # Limit file summaries to prevent prompt overflow
            # With 16K output tokens and Flash-Lite, we can handle 100+ files
            MAX_FILES_FOR_SELECTION = 100
            if len(file_summaries) > MAX_FILES_FOR_SELECTION:
                logger.warning(f"⚠️  Limiting file summaries from {len(file_summaries)} to {MAX_FILES_FOR_SELECTION} (prompt size)")
                file_summaries = file_summaries[:MAX_FILES_FOR_SELECTION]

            # Build context for the selection prompt
            files_context = self._build_files_context(file_summaries)

            # Detect if this is an exam/assessment-related query
            exam_keywords = ['exam', 'test', 'midterm', 'final', 'quiz', 'assessment']
            is_exam_query = any(keyword in user_query.lower() for keyword in exam_keywords)

            # Craft selection prompt
            syllabus_context = ""
            if syllabus_summary:
                syllabus_context = f"""
**Course Syllabus Information:**
{syllabus_summary[:800]}
{"..." if len(syllabus_summary) > 800 else ""}

Use this syllabus to understand course structure, exam coverage, and topic organization."""
            else:
                syllabus_context = "**Course Context:** No syllabus available"

            exam_instruction = ""
            if is_exam_query and syllabus_summary:
                exam_instruction = "\n⚠️ EXAM-RELATED QUERY: Consult the syllabus to identify which topics/weeks are covered in the specified exam, then select materials from those relevant periods."

            prompt = f"""You are an intelligent file selection assistant for a study assistant AI. Your task is to select the most relevant course materials to answer a student's question.

**Student Question:**
{user_query}

{syllabus_context}

**Available Course Materials:**
{files_context}

**Task:**
Select up to {max_files} most relevant files based on topic relevance. Prioritize files that directly address the question.{exam_instruction}
IMPORTANT: If the question asks for a specific number of files (e.g., "give me 7 files"), return exactly that many files if available. Otherwise, return as many relevant files as needed, up to the {max_files} limit.

**Response Format:**
Return a JSON object with selected files and brief reasoning (for debugging):

{{
  "selected_files": ["doc_id_1", "doc_id_2", "doc_id_3", ...],
  "reasoning": "Brief 1-2 sentence explanation of why these files were chosen"
}}

Example:
{{
  "selected_files": ["1432009_Lecture_14", "1432009_Sociobiology_Wilson"],
  "reasoning": "Selected files covering sociobiology theory and human behavior evolution, which directly address the paper topic."
}}

Return empty array [] with reasoning if no files are relevant."""

            # Generate selection using Gemini with error handling
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,  # Low temperature for consistent selection
                        max_output_tokens=16000,  # Increased from 3000 - allows 100+ files selection
                    )
                )

                # Check for MAX_TOKENS finish reason (incomplete response)
                if response and hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason if hasattr(response.candidates[0], 'finish_reason') else None
                    if finish_reason and str(finish_reason) == 'FinishReason.MAX_TOKENS':
                        logger.error("❌ Response truncated due to MAX_TOKENS limit")
                        logger.error(f"   Prompt length: {len(prompt)} chars")
                        logger.error(f"   File summaries: {len(file_summaries)}")
                        logger.error(f"   Consider reducing file count or summary length")
                        return []  # Fallback to manual selection

                # Validate response
                if not response or not response.text:
                    logger.error("❌ Empty/null response from Gemini API")
                    logger.error(f"   Prompt length: {len(prompt)} chars")
                    logger.error(f"   File summaries: {len(file_summaries)}")
                    logger.error(f"   Response object: {response}")
                    if response:
                        logger.error(f"   Response candidates: {response.candidates if hasattr(response, 'candidates') else 'N/A'}")
                        logger.error(f"   Prompt feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                    return []  # Fallback to manual selection

                response_text = response.text.strip()
                print(f"   🤖 RAW API RESPONSE:")
                print(f"      {response_text[:1000]}{'...' if len(response_text) > 1000 else ''}")

            except Exception as api_error:
                logger.error(f"❌ Gemini API error during file selection: {api_error}")
                import traceback
                traceback.print_exc()
                return []  # Graceful fallback to manual selection

            # Parse JSON response
            try:
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1]
                    response_text = response_text.split("```")[0]
                elif response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    response_text = response_text.split("```")[0]

                response_text = response_text.strip()
                parsed = json.loads(response_text)

                selected_doc_ids = parsed.get("selected_files", [])
                reasoning = parsed.get("reasoning", "No reasoning provided")

                print(f"   📋 PARSED RESPONSE:")
                print(f"      Selected doc_ids: {selected_doc_ids}")
                print(f"      💡 AI Reasoning: {reasoning}")

                # Convert doc_ids to file info objects for compatibility
                selected = []
                matched_ids = []
                unmatched_ids = []

                for doc_id in selected_doc_ids:
                    # Find the corresponding file info from file_summaries
                    found = False
                    for file_info in file_summaries:
                        if file_info.get("doc_id") == doc_id:
                            selected.append(file_info)
                            matched_ids.append(doc_id)
                            found = True
                            break
                    if not found:
                        unmatched_ids.append(doc_id)

                print(f"   ✅ Matched {len(matched_ids)}/{len(selected_doc_ids)} doc_ids to file summaries")
                if unmatched_ids:
                    print(f"   ⚠️  Could not find file info for {len(unmatched_ids)} doc_ids:")
                    for unmatched_id in unmatched_ids[:3]:
                        print(f"      - {unmatched_id}")

                # Log selected files
                for file_info in selected[:5]:  # Show first 5
                    logger.info(f"   📄 {file_info.get('filename')}")

                return selected

            except json.JSONDecodeError as e:
                logger.error(f"Could not parse file selection response: {e}")
                logger.error(f"Raw response: {response_text[:500]}")
                # Fallback: return all files (let LLM decide)
                return []

        except Exception as e:
            logger.error(f"Error in file selection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _build_files_context(self, file_summaries: List[Dict]) -> str:
        """Build a formatted string of file summaries for the prompt"""
        context_lines = []

        for idx, file_info in enumerate(file_summaries, 1):
            doc_id = file_info.get("doc_id", "unknown")
            filename = file_info.get("filename", "unknown")
            summary = file_info.get("summary", "No summary available")
            topics = file_info.get("topics", [])
            metadata = file_info.get("metadata", {})

            # Parse topics if it's a JSON string
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics)
                except:
                    topics = []

            # Parse metadata if it's a JSON string
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            doc_type = metadata.get("doc_type", "document")
            time_refs = metadata.get("time_references", "")

            topics_str = ", ".join(topics[:5]) if topics else "N/A"

            # Truncate summary to 150 chars to keep prompt compact
            truncated_summary = summary[:150] + '...' if len(summary) > 150 else summary

            context_lines.append(
                f"{idx}. **{filename}**\n"
                f"   - ID: {doc_id}\n"
                f"   - Type: {doc_type}\n"
                f"   - Topics: {topics_str}\n"
                f"   - Time: {time_refs if time_refs else 'N/A'}\n"
                f"   - Summary: {truncated_summary}\n"
            )

        return "\n".join(context_lines)

    async def get_syllabus_summary(
        self,
        syllabus_doc_id: str,
        file_summaries: List[Dict]
    ) -> Optional[str]:
        """
        Extract syllabus summary if available

        Args:
            syllabus_doc_id: Document ID of the syllabus (optional - can be None)
            file_summaries: List of all file summaries

        Returns:
            Syllabus summary text or None
        """
        # First, try exact match by doc_id if provided
        if syllabus_doc_id:
            for file_info in file_summaries:
                if file_info.get("doc_id") == syllabus_doc_id:
                    summary = file_info.get("summary", "")
                    print(f"      ✅ Found syllabus by ID: {file_info.get('filename')}")
                    return summary

        # If not found by exact match, search for "syllabus" in filename
        print(f"      🔍 Searching for syllabus in {len(file_summaries)} files...")
        for file_info in file_summaries:
            filename = file_info.get("filename", "").lower()
            if "syllabus" in filename:
                summary = file_info.get("summary", "")
                print(f"      ✅ Found syllabus by filename: {file_info.get('filename')}")
                return summary

        print(f"      ❌ No syllabus found")
        return None

    async def should_use_syllabus(self, user_query: str) -> bool:
        """
        Use AI to determine if the query requires syllabus/course structure context

        Args:
            user_query: The student's question

        Returns:
            True if syllabus would be helpful, False otherwise
        """
        try:
            prompt = f"""Analyze this student question and determine if it requires course structure context (syllabus, schedule, exam coverage).

Question: "{user_query}"

Answer with ONLY "yes" or "no":
- "yes" if the question is about: exams, tests, quizzes, specific weeks/chapters/modules, what's covered when, course timeline, assessment schedule
- "no" if the question is about: general topics, concepts, definitions, paper sources, understanding material

Your answer (just "yes" or "no"):"""

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=10,  # Just need "yes" or "no"
                )
            )

            answer = response.text.strip().lower()
            needs_syllabus = answer == "yes"

            print(f"      AI decision: {answer} ({'needs syllabus' if needs_syllabus else 'topic-based'})")
            return needs_syllabus

        except Exception as e:
            print(f"      ⚠️  Error determining syllabus need: {e}")
            print(f"      Defaulting to: No (topic-based)")
            return False  # Default to not using syllabus on error
