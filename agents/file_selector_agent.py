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
        self.model_id = "gemini-2.5-flash"

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
            if not file_summaries:
                logger.warning("No file summaries available for selection")
                return []

            logger.info(f"🔍 Selecting relevant files from {len(file_summaries)} available materials")
            logger.info(f"   Query: {user_query[:100]}...")

            # Build context for the selection prompt
            files_context = self._build_files_context(file_summaries)

            # Craft selection prompt
            prompt = f"""You are an intelligent file selection assistant for a study assistant AI. Your task is to select the most relevant course materials to answer a student's question.

**Student Question:**
{user_query}

**Course Context:**
{syllabus_summary if syllabus_summary else "No syllabus available"}

**Available Course Materials:**
{files_context}

**Task:**
Analyze the student's question and select the {max_files} most relevant files that would help answer it. Consider:
1. Direct topic relevance (does the file cover the topics mentioned in the question?)
2. Temporal relevance (if the question mentions specific weeks/dates, prioritize files from that period)
3. Question type (conceptual vs. procedural questions may need different materials)
4. Complementary content (files that together provide complete context)

**Response Format:**
Return a JSON array with up to {max_files} most relevant files, ordered by relevance (most relevant first):

{{
  "selected_files": [
    {{
      "doc_id": "course_123_filename",
      "filename": "Week 3 Lecture Notes.pdf",
      "relevance_score": 95,
      "reason": "Direct coverage of the question topic"
    }},
    ...
  ],
  "reasoning": "Brief explanation of selection strategy"
}}

If the question is too general or you need more files to answer comprehensively, you may select up to {max_files} files.
If no files are relevant, return an empty array with an explanation."""

            # Generate selection using Gemini
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Low temperature for consistent selection
                    max_output_tokens=1500,
                )
            )

            response_text = response.text.strip()

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

                selected = parsed.get("selected_files", [])
                reasoning = parsed.get("reasoning", "")

                logger.info(f"✅ Selected {len(selected)} relevant files")
                logger.info(f"   Reasoning: {reasoning}")

                # Log selected files
                for file_info in selected[:3]:  # Show first 3
                    logger.info(f"   📄 {file_info.get('filename')} (score: {file_info.get('relevance_score', 0)})")

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

            context_lines.append(
                f"{idx}. **{filename}**\n"
                f"   - ID: {doc_id}\n"
                f"   - Type: {doc_type}\n"
                f"   - Topics: {topics_str}\n"
                f"   - Time: {time_refs if time_refs else 'N/A'}\n"
                f"   - Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}\n"
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
            syllabus_doc_id: Document ID of the syllabus
            file_summaries: List of all file summaries

        Returns:
            Syllabus summary text or None
        """
        for file_info in file_summaries:
            if file_info.get("doc_id") == syllabus_doc_id:
                return file_info.get("summary", "")

        # If not found by exact match, search for "syllabus" in filename
        for file_info in file_summaries:
            filename = file_info.get("filename", "").lower()
            if "syllabus" in filename:
                return file_info.get("summary", "")

        return None
