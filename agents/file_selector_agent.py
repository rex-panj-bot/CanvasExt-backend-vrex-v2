"""
File Selector Agent - Hybrid Search with Vector Retrieval and LLM Reranking
Efficiently selects relevant files using a 2-step approach:
1. Vector search for fast retrieval of candidates
2. Single LLM call for intelligent reranking with reasoning
"""

from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class FileSelectorAgent:
    """
    Selects relevant files using Hybrid Search:
    - Step A: Vector similarity search for fast candidate retrieval
    - Step B: LLM reranking with reasoning for intelligent selection
    """

    # Configuration
    MAX_CANDIDATES = 30  # Vector search retrieval limit
    MAX_FILES_HARD_CAP = 18  # Maximum files to return
    RELATIVE_THRESHOLD_DELTA = 0.50  # Score cutoff relative to max score (wider = more files)

    # Embedding model for query embedding
    EMBEDDING_MODEL = "text-embedding-004"

    def __init__(self, google_api_key: str, chat_storage=None, file_summarizer=None):
        """
        Initialize File Selector Agent

        Args:
            google_api_key: Google API key for Gemini
            chat_storage: ChatStorage instance for vector search
            file_summarizer: FileSummarizer instance for query embedding
        """
        self.client = genai.Client(api_key=google_api_key)
        self.model_id = "gemini-2.5-flash-lite"  # Fast model for reranking
        self.fallback_model = "gemini-2.0-flash"
        self.chat_storage = chat_storage
        self.file_summarizer = file_summarizer

    async def select_relevant_files(
        self,
        user_query: str,
        file_summaries: List[Dict],
        course_id: str = None,
        syllabus_summary: Optional[str] = None,
        syllabus_doc_id: Optional[str] = None,
        selected_docs: Optional[List[str]] = None,
        max_files: int = 18,
        user_api_key: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Select relevant files using hybrid search (vector + LLM reranking)

        Args:
            user_query: The user's question
            file_summaries: List of dicts with doc_id, filename, summary, topics, metadata
            course_id: Course ID for vector search
            syllabus_summary: Course syllabus summary text (optional)
            syllabus_doc_id: Document ID of syllabus (optional)
            selected_docs: List of doc_ids user manually selected (None = use all)
            max_files: Maximum files to return (capped at MAX_FILES_HARD_CAP)

        Returns:
            Tuple of:
            - List of selected file dicts with _score and _reason
            - List of selection reasoning dicts for UI (file_id, filename, score, reason)
        """
        try:
            # Cap max_files
            max_files = min(max_files, self.MAX_FILES_HARD_CAP)

            print(f"\n   HYBRID FILE SELECTOR")
            print(f"      Query: {user_query[:100]}...")
            print(f"      Available summaries: {len(file_summaries)}")
            print(f"      Manual selection: {len(selected_docs) if selected_docs else 'None (global)'}")

            if not file_summaries:
                print(f"      No file summaries available")
                return [], []

            # Get syllabus summary if not provided but doc_id is given
            if not syllabus_summary and syllabus_doc_id:
                syllabus_summary = await self.get_syllabus_summary(syllabus_doc_id, file_summaries)

            # Determine working set of files
            if selected_docs and len(selected_docs) > 0:
                # Scoped mode: Filter to user's selection
                selected_set = set(selected_docs)
                working_summaries = [f for f in file_summaries if f.get('doc_id') in selected_set]
                print(f"      Scoped to {len(working_summaries)} user-selected files")
            else:
                working_summaries = file_summaries

            # STEP A: Fast Retrieval via Vector Search
            candidates = await self._vector_retrieval(
                user_query,
                course_id,
                working_summaries
            )

            if not candidates:
                print(f"      Vector search returned no candidates, using all files")
                candidates = working_summaries[:self.MAX_CANDIDATES]

            print(f"      Step A (Vector Retrieval): {len(candidates)} candidates")

            # STEP B: LLM Reranking with Reasoning
            # Use user's API key if provided to avoid server rate limits
            reranked_results = await self._llm_rerank_with_reasoning(
                user_query,
                candidates,
                syllabus_summary,
                user_api_key=user_api_key
            )

            if not reranked_results:
                print(f"      LLM reranking failed, using vector scores")
                # Fallback: use vector similarity scores
                for c in candidates:
                    c['_score'] = c.get('similarity_score', 0.5)
                    c['_reason'] = "Selected by semantic similarity"
                return candidates[:max_files], self._build_reasoning_output(candidates[:max_files])

            # Apply Dynamic Filtering (Relative Scoring)
            filtered_results = self._apply_dynamic_filtering(reranked_results, max_files)

            # Build output
            selected_files = []
            reasoning_output = []

            for result in filtered_results:
                doc_id = result.get('file_id')
                # Find full file info
                file_info = next(
                    (f for f in candidates if f.get('doc_id') == doc_id),
                    None
                )
                if file_info:
                    file_info['_score'] = result.get('score', 0.5)
                    file_info['_reason'] = result.get('reason', 'Selected as relevant')
                    selected_files.append(file_info)

                    reasoning_output.append({
                        'file_id': doc_id,
                        'filename': file_info.get('filename', 'Unknown'),
                        'score': result.get('score', 0.5),
                        'reason': result.get('reason', 'Selected as relevant')
                    })

            print(f"      Step B (LLM Reranking): {len(selected_files)} files selected")
            print(f"      Dynamic Filtering: {len(filtered_results)} files after threshold")

            for r in reasoning_output[:5]:
                print(f"         {r['filename'][:40]}... (score: {r['score']:.2f})")
                print(f"            Reason: {r['reason'][:60]}...")

            return selected_files, reasoning_output

        except Exception as e:
            logger.error(f"Error in file selection: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    async def _vector_retrieval(
        self,
        user_query: str,
        course_id: str,
        fallback_summaries: List[Dict]
    ) -> List[Dict]:
        """
        Step A: Fast retrieval using vector similarity search

        Args:
            user_query: User's question
            course_id: Course ID for database lookup
            fallback_summaries: Summaries to use if vector search unavailable

        Returns:
            List of candidate file dicts with similarity_score
        """
        try:
            # Check if vector search is available
            if not self.chat_storage or not self.file_summarizer or not course_id:
                print(f"      Vector search unavailable, using fallback")
                return fallback_summaries[:self.MAX_CANDIDATES]

            # Generate query embedding
            print(f"      Generating query embedding...")
            query_embedding = await self.file_summarizer.generate_query_embedding(user_query)

            if not query_embedding:
                print(f"      Failed to generate query embedding, using fallback")
                return fallback_summaries[:self.MAX_CANDIDATES]

            # Perform vector search
            print(f"      Searching vector database...")
            candidates = self.chat_storage.search_similar_summaries(
                course_id=course_id,
                query_embedding=query_embedding,
                limit=self.MAX_CANDIDATES
            )

            if not candidates:
                print(f"      Vector search returned no results, using fallback")
                return fallback_summaries[:self.MAX_CANDIDATES]

            print(f"      Vector search found {len(candidates)} candidates")

            # Top 3 candidates for debugging
            for c in candidates[:3]:
                print(f"         {c.get('filename', 'Unknown')[:40]}... (similarity: {c.get('similarity_score', 0):.3f})")

            return candidates

        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
            return fallback_summaries[:self.MAX_CANDIDATES]

    async def _llm_rerank_with_reasoning(
        self,
        user_query: str,
        candidates: List[Dict],
        syllabus_summary: Optional[str],
        user_api_key: Optional[str] = None
    ) -> List[Dict]:
        """
        Step B: LLM reranking with reasoning in a single call

        Args:
            user_query: User's question
            candidates: Candidate files from vector search
            syllabus_summary: Optional syllabus for context
            user_api_key: User's API key (use instead of server key to avoid rate limits)

        Returns:
            List of dicts with file_id, score (0-1), reason
        """
        try:
            if not candidates:
                return []

            # Build files context for prompt
            files_context = self._build_files_context(candidates)

            syllabus_context = ""
            if syllabus_summary:
                syllabus_context = f"\n**Course Context (from syllabus):**\n{syllabus_summary[:500]}...\n"

            # Single LLM call for reranking + reasoning
            prompt = f"""Analyze these {len(candidates)} course files and select the most relevant ones for answering the student's question.

**Student Question:**
{user_query}
{syllabus_context}

**Candidate Files:**
{files_context}

**Task:**
1. Analyze each file's relevance to the question
2. Assign a relevance score (0.0 to 1.0)
3. Provide a brief reason for selection (1 sentence, for display to student)

**PRIORITY SCORING - Use these guidelines carefully:**

HIGH VALUE (0.85-1.0) - Prioritize these:
- Study guides, review sheets, exam prep materials
- Lecture slides/notes covering the exact topic
- Practice problems or sample exams
- Chapter summaries or key concept documents

MEDIUM VALUE (0.60-0.84):
- Lectures covering related topics
- Readings that provide context
- Assignments that reinforce concepts

LOW VALUE (0.30-0.59):
- Tangentially related materials
- General course documents

AVOID (0.0-0.29) - Score these LOW:
- Administrative documents (extra credit, class evaluation)
- Technical/browser requirements
- Unrelated assignments or readings

**Output Format:**
Return ONLY a JSON array, ordered by relevance (highest first):
[
  {{
    "file_id": "doc_id_here",
    "score": 0.95,
    "reason": "Contains the study guide for Exam 1 covering mitosis"
  }},
  ...
]

Include ALL files that have score >= 0.2. Return ONLY the JSON array."""

            print(f"      Calling LLM for reranking {len(candidates)} candidates...")
            # Need enough tokens for JSON array with file_id, score, reason for each candidate
            # Roughly ~200 tokens per candidate, so 30 candidates needs ~6000+ tokens
            response = await self._call_ai_with_fallback(prompt, max_tokens=8000, user_api_key=user_api_key)

            if not response:
                print(f"      ⚠️ LLM reranking returned no response")
                logger.warning("LLM reranking returned no response")
                return []

            print(f"      ✅ LLM response received ({len(response)} chars)")
            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]

            results = json.loads(response_text.strip())

            if not isinstance(results, list):
                logger.error("LLM response is not a list")
                return []

            # Validate and clean results
            valid_results = []
            candidate_ids = {c.get('doc_id') for c in candidates}
            print(f"      Validating {len(results)} results against {len(candidate_ids)} candidates")
            # Show sample IDs for debugging
            sample_candidate_ids = list(candidate_ids)[:2]
            print(f"      Sample candidate IDs: {sample_candidate_ids}")

            matched = 0
            unmatched = 0
            for r in results:
                if not isinstance(r, dict):
                    continue
                file_id = r.get('file_id')
                if file_id not in candidate_ids:
                    unmatched += 1
                    if unmatched <= 3:
                        print(f"      ⚠️ Unmatched file_id: {file_id[:50] if file_id else 'None'}...")
                    continue

                matched += 1
                valid_results.append({
                    'file_id': file_id,
                    'score': min(1.0, max(0.0, float(r.get('score', 0.5)))),
                    'reason': str(r.get('reason', 'Selected as relevant'))[:200]
                })

            print(f"      Matched: {matched}, Unmatched: {unmatched}")

            # Sort by score descending
            valid_results.sort(key=lambda x: x['score'], reverse=True)

            return valid_results

        except Exception as e:
            print(f"      ❌ Error in LLM reranking: {e}")
            logger.error(f"Error in LLM reranking: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _apply_dynamic_filtering(
        self,
        reranked_results: List[Dict],
        max_files: int
    ) -> List[Dict]:
        """
        Apply dynamic filtering with relative scoring

        Rules:
        1. Hard cap at max_files (default 18)
        2. Dynamic threshold: cutoff = max_score - 0.25
           - If best match is 0.95, cutoff is 0.70 (drop mediocre)
           - If best match is 0.40, cutoff is 0.15 (keep weak matches)

        Args:
            reranked_results: Results from LLM reranking (sorted by score desc)
            max_files: Maximum files to return

        Returns:
            Filtered list of results
        """
        if not reranked_results:
            return []

        # Get max score
        max_score = reranked_results[0].get('score', 0.5)

        # Calculate dynamic cutoff
        cutoff = max(0.0, max_score - self.RELATIVE_THRESHOLD_DELTA)

        print(f"      Dynamic threshold: max_score={max_score:.2f}, cutoff={cutoff:.2f}")

        # Filter by cutoff
        filtered = [r for r in reranked_results if r.get('score', 0) >= cutoff]

        # Apply hard cap
        filtered = filtered[:max_files]

        print(f"      After filtering: {len(filtered)} files (from {len(reranked_results)})")

        return filtered

    def _build_files_context(self, file_summaries: List[Dict]) -> str:
        """Build formatted string of file summaries for LLM prompt"""
        context_lines = []

        for idx, file_info in enumerate(file_summaries, 1):
            doc_id = file_info.get("doc_id", "unknown")
            filename = file_info.get("filename", "unknown")
            summary = file_info.get("summary", "")
            topics = file_info.get("topics", [])

            # Parse topics if string
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics)
                except:
                    topics = []

            topics_str = ", ".join(topics[:5]) if topics else "N/A"
            truncated_summary = summary[:200] + '...' if len(summary) > 200 else summary

            context_lines.append(
                f"{idx}. **{filename}** (ID: {doc_id})\n"
                f"   Topics: {topics_str}\n"
                f"   Summary: {truncated_summary}\n"
            )

        return "\n".join(context_lines)

    def _build_reasoning_output(self, files: List[Dict]) -> List[Dict]:
        """Build reasoning output for fallback case"""
        return [
            {
                'file_id': f.get('doc_id'),
                'filename': f.get('filename', 'Unknown'),
                'score': f.get('_score', 0.5),
                'reason': f.get('_reason', 'Selected by semantic similarity')
            }
            for f in files
        ]

    async def _call_ai_with_fallback(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        user_api_key: Optional[str] = None
    ) -> Optional[str]:
        """Call AI with fallback to secondary model on rate limit"""
        import asyncio

        # Use user's API key if provided to avoid server rate limits
        if user_api_key:
            client = genai.Client(api_key=user_api_key)
            print(f"      📡 Calling {self.model_id} (using user API key)...")
        else:
            client = self.client
            print(f"      📡 Calling {self.model_id} (using server API key)...")

        try:
            # Run synchronous API call in thread pool
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            if not response or not response.text:
                print(f"      ⚠️ No response text from {self.model_id}")
                return None

            return response.text.strip()

        except Exception as api_error:
            error_str = str(api_error).lower()
            print(f"      ⚠️ API error: {api_error}")
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                print(f"      🔄 Rate limited, trying fallback model {self.fallback_model}")
                logger.warning(f"Rate limited, trying fallback model")
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,  # Use same client (user's key if provided)
                        model=self.fallback_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                    return response.text.strip() if response and response.text else None
                except Exception as fallback_error:
                    print(f"      ❌ Fallback also failed: {fallback_error}")
                    return None
            logger.error(f"AI call failed: {api_error}")
            return None

    async def get_syllabus_summary(
        self,
        syllabus_doc_id: Optional[str],
        file_summaries: List[Dict]
    ) -> Optional[str]:
        """
        Extract syllabus summary if available

        Args:
            syllabus_doc_id: Document ID of syllabus (can be None)
            file_summaries: All file summaries

        Returns:
            Syllabus summary text or None
        """
        # Try exact match by doc_id
        if syllabus_doc_id:
            for file_info in file_summaries:
                if file_info.get("doc_id") == syllabus_doc_id:
                    return file_info.get("summary", "")

        # Search by filename
        for file_info in file_summaries:
            filename = file_info.get("filename", "").lower()
            if "syllabus" in filename:
                return file_info.get("summary", "")

        return None
