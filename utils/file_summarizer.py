"""
File Summarizer
Generates concise summaries of course materials for intelligent file selection
"""

from google import genai
from google.genai import types
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class FileSummarizer:
    """Generates summaries of uploaded files using Gemini"""

    def __init__(self, google_api_key: str):
        """
        Initialize File Summarizer

        Args:
            google_api_key: Google API key for Gemini
        """
        self.client = genai.Client(api_key=google_api_key)
        self.model_id = "gemini-2.5-flash-lite"  # 3-6x cheaper, same 65K token limit

    async def summarize_file(
        self,
        file_uri: str,
        filename: str,
        mime_type: str
    ) -> Tuple[str, List[str], Dict]:
        """
        Generate a summary of a file

        Args:
            file_uri: Gemini File API URI
            filename: Original filename
            mime_type: MIME type of the file

        Returns:
            Tuple of (summary, topics_list, metadata_dict)
        """
        try:
            logger.info(f"Generating summary for: {filename}")

            # Craft prompt for summarization (optimized for speed)
            prompt = f"""Briefly summarize this document in 40-50 words and list the 3 most important topics.

Document: {filename}

Format as JSON:
{{"summary": "...", "topics": ["topic1", "topic2", "topic3"]}}"""

            # Generate summary using Gemini (optimized config for speed)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Lower = faster, more deterministic
                    max_output_tokens=500,  # Increased from 200 for more detailed topics
                    top_p=0.8,  # Reduce token sampling
                    top_k=20    # Reduce token candidates
                )
            )

            response_text = response.text.strip()

            # Try to parse JSON response
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

                summary = parsed.get("summary", "")
                topics = parsed.get("topics", [])
                # Simplified metadata (no doc_type or time_references in new prompt)
                metadata = {
                    "doc_type": "document",
                    "optimized": True  # Flag to indicate optimized summarization
                }

                logger.info(f"✅ Generated summary for {filename}: {len(summary)} chars, {len(topics)} topics")
                return summary, topics, metadata

            except json.JSONDecodeError as e:
                # Fallback: use raw response as summary
                logger.warning(f"Could not parse JSON response for {filename}, using raw text: {e}")
                return response_text, [], {"doc_type": "unknown", "time_references": ""}

        except Exception as e:
            logger.error(f"Error generating summary for {filename}: {e}")
            # Return empty summary on error
            return f"Error generating summary: {str(e)}", [], {"doc_type": "unknown", "error": str(e)}

    async def summarize_text_content(
        self,
        content: str,
        filename: str
    ) -> Tuple[str, List[str], Dict]:
        """
        Generate summary from raw text content (for pages, assignments, etc.)

        Args:
            content: Text content to summarize
            filename: Name/title of the content

        Returns:
            Tuple of (summary, topics_list, metadata_dict)
        """
        try:
            logger.info(f"Generating summary for text content: {filename}")

            # Truncate content if too long (keep first 8000 chars for context)
            if len(content) > 8000:
                content = content[:8000] + "...[truncated]"

            prompt = f"""Analyze this content and provide a structured analysis:

Title: {filename}

Content:
{content}

Please provide:
1. **Summary**: A concise 150-200 word summary capturing the main content and purpose
2. **Topics**: 5-10 key topics, concepts, or keywords (comma-separated list)
3. **Content Type**: (e.g., assignment description, course page, syllabus, schedule, etc.)
4. **Time/Date References**: Any specific dates, weeks, or time periods mentioned

Format your response as JSON:
{{
  "summary": "...",
  "topics": ["topic1", "topic2", ...],
  "doc_type": "...",
  "time_references": "..."
}}"""

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2000,  # Increased from 800 for richer summaries
                )
            )

            response_text = response.text.strip()

            # Parse JSON response
            try:
                if response_text.startswith("```json"):
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif response_text.startswith("```"):
                    response_text = response_text.split("```")[1].split("```")[0]

                response_text = response_text.strip()
                parsed = json.loads(response_text)

                summary = parsed.get("summary", "")
                topics = parsed.get("topics", [])
                metadata = {
                    "doc_type": parsed.get("doc_type", "page"),
                    "time_references": parsed.get("time_references", "")
                }

                logger.info(f"✅ Generated summary for {filename}: {len(summary)} chars, {len(topics)} topics")
                return summary, topics, metadata

            except json.JSONDecodeError:
                # Fallback
                logger.warning(f"Could not parse JSON for {filename}, using raw text")
                return response_text, [], {"doc_type": "page", "time_references": ""}

        except Exception as e:
            logger.error(f"Error generating summary for {filename}: {e}")
            return f"Error: {str(e)}", [], {"doc_type": "page", "error": str(e)}
