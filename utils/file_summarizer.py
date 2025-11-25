"""
File Summarizer
Generates concise summaries of course materials for intelligent file selection
"""

from google import genai
from google.genai import types
from typing import Dict, List, Optional, Tuple
import json
import logging
import asyncio
import random

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
        self.model_id = "gemini-2.0-flash-lite"  # Higher RPM (30) for batch summarization
        self.fallback_model = "gemini-2.0-flash"  # Valid fallback model

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable (rate limit, server error, timeout)"""
        error_str = str(error).lower()

        # Retryable errors: rate limits, server errors, timeouts
        retryable_indicators = [
            '429',  # Rate limit
            'quota',  # Quota exceeded
            'rate limit',  # Rate limit text
            '503',  # Service unavailable
            '500',  # Internal server error
            'overloaded',  # Model overloaded
            'timeout',  # Timeout
            'deadline',  # Deadline exceeded
        ]

        return any(indicator in error_str for indicator in retryable_indicators)

    def _get_error_type(self, error: Exception) -> str:
        """Get human-readable error type"""
        error_str = str(error).lower()

        if '429' in error_str or 'quota' in error_str or 'rate limit' in error_str:
            return 'RATE_LIMIT'
        elif '503' in error_str:
            return 'SERVICE_UNAVAILABLE'
        elif '500' in error_str:
            return 'INTERNAL_ERROR'
        elif 'overloaded' in error_str:
            return 'MODEL_OVERLOADED'
        elif 'timeout' in error_str or 'deadline' in error_str:
            return 'TIMEOUT'
        else:
            return 'UNKNOWN_ERROR'

    async def summarize_file(
        self,
        file_uri: str,
        filename: str,
        mime_type: str
    ) -> Tuple[str, List[str], Dict]:
        """
        Generate a summary of a file with exponential backoff retry logic

        Args:
            file_uri: Gemini File API URI
            filename: Original filename
            mime_type: MIME type of the file

        Returns:
            Tuple of (summary, topics_list, metadata_dict)

        Raises:
            Exception: After 5 retry attempts fail
        """
        logger.info(f"Generating summary for: {filename}")

        # Craft prompt for summarization (optimized for speed)
        prompt = f"""Briefly summarize this document in 40-50 words and list the 3 most important topics.

Document: {filename}

Return ONLY valid JSON with no explanatory text."""

        # Retry configuration
        max_attempts = 5
        base_delay = 2.0  # Start with 2 seconds
        model_to_use = self.model_id

        for attempt in range(max_attempts):
            try:
                # Generate summary using Gemini
                response = self.client.models.generate_content(
                    model=model_to_use,
                    contents=[
                        types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Lower = faster, more deterministic
                        max_output_tokens=1500,
                        top_p=0.8,  # Reduce token sampling
                        top_k=20,    # Reduce token candidates
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": "A concise 40-50 word summary"
                                },
                                "topics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "3 most important topics"
                                }
                            },
                            "required": ["summary", "topics"]
                        }
                    )
                )

                # Success! Parse and return response
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
                    # Invalid JSON response - raise error instead of saving raw text
                    logger.error(f"Could not parse JSON response for {filename}: {e}")
                    logger.error(f"Raw response: {response_text[:200]}")
                    raise ValueError(f"Invalid JSON response from Gemini API: {response_text[:200]}")

            except Exception as e:
                error_type = self._get_error_type(e)
                is_retryable = self._is_retryable_error(e)

                # Log detailed error information
                logger.warning(f"⚠️ Attempt {attempt + 1}/{max_attempts} failed for {filename}: {error_type} - {str(e)[:100]}")

                # If not retryable, raise immediately
                if not is_retryable:
                    logger.error(f"❌ Non-retryable error for {filename}: {error_type}")
                    raise

                # If this was the last attempt, raise
                if attempt == max_attempts - 1:
                    logger.error(f"❌ All {max_attempts} attempts failed for {filename}")
                    raise

                # Calculate exponential backoff with jitter
                # Delays: 2s, 4s, 8s, 16s (with ±25% jitter)
                exp_delay = base_delay * (2 ** attempt)
                jitter = random.uniform(0.75, 1.25)  # ±25% jitter
                wait_time = exp_delay * jitter

                # Try fallback model on first rate limit
                if error_type == 'RATE_LIMIT' and attempt == 0:
                    model_to_use = self.fallback_model
                    logger.info(f"   → Switching to fallback model: {model_to_use}")

                logger.info(f"   → Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

        # Should never reach here
        raise Exception(f"Failed to generate summary for {filename} after {max_attempts} attempts")

    async def summarize_files_batch(
        self,
        files: List[Dict]
    ) -> Dict:
        """
        Generate summaries for multiple files in a single API request

        Gemini supports up to 3,000 files per request. We'll batch 10-15 files at a time
        for optimal performance while staying within context limits.

        Args:
            files: List of dicts with keys:
                - uri: Gemini File API URI
                - filename: Original filename
                - mime_type: MIME type
                - file_id: File ID for matching responses
                - text_content: Optional text content (for text files)

        Returns:
            Dict with:
                - summaries: List of dicts with file_id, summary, topics, metadata
                - failed: List of dicts with file_id, error
                - success_count: Number of successful summaries
                - failed_count: Number of failed summaries
        """
        logger.info(f"Generating batch summaries for {len(files)} files")

        # Build batch prompt with all files
        file_parts = []
        file_mapping = {}  # Map index to file_id for matching responses

        for idx, file_info in enumerate(files):
            file_id = file_info.get('file_id')
            filename = file_info.get('filename', f'file_{idx}')
            mime_type = file_info.get('mime_type', 'application/pdf')

            # Store mapping for response matching
            file_mapping[idx] = {
                'file_id': file_id,
                'filename': filename
            }

            # Add file to batch (either as URI or text content)
            if file_info.get('text_content'):
                # Text content - include inline
                text_content = file_info['text_content']
                # Truncate if too long
                if len(text_content) > 8000:
                    text_content = text_content[:8000] + "...[truncated]"
                file_parts.append(f"FILE {idx}: {filename}\nContent:\n{text_content}\n\n")
            else:
                # File URI - use Gemini Part
                uri = file_info.get('uri')
                if uri:
                    file_parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime_type))
                    file_parts.append(f"FILE {idx}: {filename}\n\n")

        # Craft batch prompt
        prompt = f"""Analyze these {len(files)} documents and provide summaries for each.

For EACH document, return:
1. **Summary**: A concise 40-50 word summary
2. **Topics**: 3 most important topics
3. **Content Type**: (assignment, lecture, syllabus, schedule, etc.)

Return ONLY valid JSON array with one object per file, in the same order.
Format: [{{"file_index": 0, "summary": "...", "topics": ["...", "..."], "doc_type": "..."}}, ...]"""

        # Add prompt to content
        file_parts.append(prompt)

        # Retry configuration
        max_attempts = 3  # Fewer retries for batch to fail fast
        base_delay = 2.0
        model_to_use = self.model_id

        for attempt in range(max_attempts):
            try:
                # Generate batch summaries using Gemini (run in thread pool to avoid blocking)
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_to_use,
                    contents=file_parts,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=len(files) * 500,  # ~500 tokens per summary
                        top_p=0.8,
                        top_k=20,
                        response_mime_type="application/json"
                    )
                )

                # Parse batch response
                response_text = response.text.strip()

                try:
                    # Remove markdown code blocks if present
                    if response_text.startswith("```json"):
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif response_text.startswith("```"):
                        response_text = response_text.split("```")[1].split("```")[0]

                    response_text = response_text.strip()
                    parsed = json.loads(response_text)

                    # Ensure parsed is a list
                    if not isinstance(parsed, list):
                        raise ValueError("Expected JSON array response")

                    # Match responses to files
                    summaries = []
                    failed = []

                    for item in parsed:
                        file_idx = item.get('file_index')
                        if file_idx is None or file_idx not in file_mapping:
                            continue

                        file_info = file_mapping[file_idx]
                        summary = item.get('summary', '')
                        topics = item.get('topics', [])
                        doc_type = item.get('doc_type', 'document')

                        if summary and topics:
                            summaries.append({
                                'file_id': file_info['file_id'],
                                'filename': file_info['filename'],
                                'summary': summary,
                                'topics': topics,
                                'metadata': {
                                    'doc_type': doc_type,
                                    'batch_summarized': True
                                }
                            })
                        else:
                            failed.append({
                                'file_id': file_info['file_id'],
                                'filename': file_info['filename'],
                                'error': 'Empty summary or topics'
                            })

                    logger.info(f"✅ Batch summary complete: {len(summaries)} succeeded, {len(failed)} failed")

                    return {
                        'summaries': summaries,
                        'failed': failed,
                        'success_count': len(summaries),
                        'failed_count': len(failed)
                    }

                except json.JSONDecodeError as e:
                    logger.error(f"Could not parse batch JSON response: {e}")
                    logger.error(f"Raw response: {response_text[:500]}")
                    raise ValueError(f"Invalid JSON response from Gemini API: {response_text[:200]}")

            except Exception as e:
                error_type = self._get_error_type(e)
                is_retryable = self._is_retryable_error(e)

                logger.warning(f"⚠️ Batch attempt {attempt + 1}/{max_attempts} failed: {error_type} - {str(e)[:100]}")

                if not is_retryable or attempt == max_attempts - 1:
                    # Return all as failed
                    logger.error(f"❌ Batch summarization failed after {attempt + 1} attempts")
                    failed = [
                        {'file_id': info['file_id'], 'filename': info['filename'], 'error': str(e)[:200]}
                        for info in file_mapping.values()
                    ]
                    return {
                        'summaries': [],
                        'failed': failed,
                        'success_count': 0,
                        'failed_count': len(failed)
                    }

                # Exponential backoff
                exp_delay = base_delay * (2 ** attempt)
                jitter = random.uniform(0.75, 1.25)
                wait_time = exp_delay * jitter

                if error_type == 'RATE_LIMIT' and attempt == 0:
                    model_to_use = self.fallback_model
                    logger.info(f"   → Switching to fallback model: {model_to_use}")

                logger.info(f"   → Retrying batch in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

        # Fallback: all failed
        failed = [
            {'file_id': info['file_id'], 'filename': info['filename'], 'error': 'Max retries exceeded'}
            for info in file_mapping.values()
        ]
        return {
            'summaries': [],
            'failed': failed,
            'success_count': 0,
            'failed_count': len(failed)
        }

    async def summarize_text_content(
        self,
        content: str,
        filename: str
    ) -> Tuple[str, List[str], Dict]:
        """
        Generate summary from raw text content with exponential backoff retry logic

        Args:
            content: Text content to summarize
            filename: Name/title of the content

        Returns:
            Tuple of (summary, topics_list, metadata_dict)

        Raises:
            Exception: After 5 retry attempts fail
        """
        logger.info(f"Generating summary for text content: {filename}")

        # Truncate content if too long (keep first 8000 chars for context)
        if len(content) > 8000:
            content = content[:8000] + "...[truncated]"

        prompt = f"""Analyze this content and provide a structured analysis:

Title: {filename}

Content:
{content}

Provide:
1. **Summary**: A concise 150-200 word summary capturing the main content and purpose
2. **Topics**: 5-10 key topics, concepts, or keywords
3. **Content Type**: (e.g., assignment description, course page, syllabus, schedule, etc.)
4. **Time/Date References**: Any specific dates, weeks, or time periods mentioned

Return ONLY valid JSON with no explanatory text."""

        # Retry configuration
        max_attempts = 5
        base_delay = 2.0
        model_to_use = self.model_id

        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=model_to_use,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=2500,
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": "A concise 150-200 word summary"
                                },
                                "topics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "5-10 key topics, concepts, or keywords"
                                },
                                "doc_type": {
                                    "type": "string",
                                    "description": "Content type (assignment, page, syllabus, etc.)"
                                },
                                "time_references": {
                                    "type": "string",
                                    "description": "Specific dates, weeks, or time periods"
                                }
                            },
                            "required": ["summary", "topics", "doc_type", "time_references"]
                        }
                    )
                )

                # Success! Parse and return response
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

                except json.JSONDecodeError as e:
                    logger.error(f"Could not parse JSON for {filename}: {e}")
                    logger.error(f"Raw response: {response_text[:200]}")
                    raise ValueError(f"Invalid JSON response from Gemini API: {response_text[:200]}")

            except Exception as e:
                error_type = self._get_error_type(e)
                is_retryable = self._is_retryable_error(e)

                logger.warning(f"⚠️ Attempt {attempt + 1}/{max_attempts} failed for {filename}: {error_type} - {str(e)[:100]}")

                if not is_retryable:
                    logger.error(f"❌ Non-retryable error for {filename}: {error_type}")
                    raise

                if attempt == max_attempts - 1:
                    logger.error(f"❌ All {max_attempts} attempts failed for {filename}")
                    raise

                # Exponential backoff with jitter
                exp_delay = base_delay * (2 ** attempt)
                jitter = random.uniform(0.75, 1.25)
                wait_time = exp_delay * jitter

                # Try fallback model on first rate limit
                if error_type == 'RATE_LIMIT' and attempt == 0:
                    model_to_use = self.fallback_model
                    logger.info(f"   → Switching to fallback model: {model_to_use}")

                logger.info(f"   → Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

        raise Exception(f"Failed to generate summary for {filename} after {max_attempts} attempts")
