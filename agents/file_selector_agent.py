"""
File Selector Agent - Anchor Cluster Implementation
Intelligently selects course materials using a two-stage "Anchor Cluster" approach
"""

from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple
import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class FileSelectorAgent:
    """Selects relevant files using Anchor Cluster pattern"""

    def __init__(self, google_api_key: str):
        """
        Initialize File Selector Agent

        Args:
            google_api_key: Google API key for Gemini
        """
        self.client = genai.Client(api_key=google_api_key)
        self.model_id = "gemini-2.0-flash-lite"  # Lightweight, separate quota
        self.fallback_model = "gemini-2.0-flash"  # Fallback when rate limited

    async def select_relevant_files(
        self,
        user_query: str,
        file_summaries: List[Dict],
        syllabus_summary: Optional[str] = None,
        syllabus_doc_id: Optional[str] = None,
        selected_docs: Optional[List[str]] = None,
        max_files: int = 15
    ) -> List[Dict]:
        """
        Route to appropriate selection strategy based on user state

        Args:
            user_query: The user's question
            file_summaries: List of dicts with doc_id, filename, summary, topics, metadata
            syllabus_summary: Course syllabus summary text
            syllabus_doc_id: Document ID of syllabus
            selected_docs: List of doc_ids user manually selected (None = Global Discovery)
            max_files: Maximum files to return

        Returns:
            List of selected file dicts
        """
        try:
            print(f"\n   🎯 ANCHOR CLUSTER FILE SELECTOR")
            print(f"      Query: {user_query[:150]}...")
            print(f"      Available summaries: {len(file_summaries)}")
            print(f"      Manual selection: {len(selected_docs) if selected_docs else 0} files")

            if not file_summaries:
                print(f"      ❌ No file summaries available")
                return []

            # Limit file summaries to prevent prompt overflow
            MAX_FILES = 100
            if len(file_summaries) > MAX_FILES:
                logger.warning(f"⚠️  Limiting from {len(file_summaries)} to {MAX_FILES}")
                file_summaries = file_summaries[:MAX_FILES]

            # Route based on user state
            if not selected_docs or len(selected_docs) == 0:
                # Scenario 1: Global Discovery
                print(f"   🌍 SCENARIO 1: Global Discovery (no manual selection)")
                return await self.global_discovery(
                    user_query, file_summaries, syllabus_summary, max_files
                )
            else:
                # Scenario 2: Scoped Refinement
                print(f"   🔍 SCENARIO 2: Scoped Refinement ({len(selected_docs)} files selected)")
                return await self.scoped_refinement(
                    user_query, file_summaries, syllabus_summary, syllabus_doc_id,
                    selected_docs, max_files
                )

        except Exception as e:
            logger.error(f"Error in file selection: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def global_discovery(
        self,
        user_query: str,
        file_summaries: List[Dict],
        syllabus_summary: Optional[str],
        max_files: int
    ) -> List[Dict]:
        """
        Scenario 1: Global Discovery (user has NOT manually selected files)

        AI-Driven Flow:
        1. AI analyzes query intent (topic, type, scope)
        2. AI identifies anchor files based on summaries (not filenames)
        3. Extract keywords from AI-selected anchors
        4. Score all files using anchor keywords
        5. Return top matches

        Args:
            user_query: User's question
            file_summaries: All available file summaries
            syllabus_summary: Syllabus text for context
            max_files: Max files to return

        Returns:
            List of selected file dicts
        """
        try:
            print(f"\n   📍 GLOBAL DISCOVERY FLOW (AI-Driven)")

            # Step 1: AI analyzes query intent
            query_intent = await self._analyze_query_intent(user_query, syllabus_summary)
            print(f"      Main Topic: {query_intent.get('main_topic', 'N/A')}")
            print(f"      Query Type: {query_intent.get('query_type', 'N/A')}")
            print(f"      Scope: {query_intent.get('scope', 'N/A')}")
            print(f"      Context Hints: {query_intent.get('context_hints', [])[:5]}")

            # Step 2: AI identifies anchor files based on content
            anchors = await self._identify_anchors_ai(
                query_intent,
                file_summaries,
                syllabus_summary,
                max_anchors=5
            )
            print(f"      AI selected {len(anchors)} anchor files")
            for anchor in anchors[:3]:
                reasoning = anchor.get('_anchor_reasoning', 'N/A')
                print(f"         🎯 {anchor['filename']}")
                print(f"            → {reasoning}")

            # Fallback: If AI selection fails, use legacy heuristics
            if not anchors:
                print(f"      ⚠️  AI anchor selection failed, using fallback heuristics")
                anchors = self._identify_anchor_files(file_summaries)
                print(f"      Found {len(anchors)} anchor files via fallback")

            # Step 3: Extract keywords from anchors
            anchor_keywords = await self._extract_anchor_keywords(anchors, user_query)
            print(f"      Extracted {len(anchor_keywords)} keywords from anchors")
            print(f"         Keywords: {anchor_keywords[:10]}")

            # Step 4: Score all files using anchor keywords + query intent
            scored_files = await self._score_files_with_anchors(
                file_summaries,
                user_query,
                anchor_keywords,
                query_intent.get('context_hints', [])
            )

            # Step 5: Return top matches
            selected = scored_files[:max_files]
            print(f"   ✅ Selected {len(selected)} files from AI-driven global discovery")
            for file in selected[:5]:
                print(f"      📄 {file.get('filename')} (score: {file.get('_score', 0):.2f})")

            return selected

        except Exception as e:
            logger.error(f"Error in global discovery: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def scoped_refinement(
        self,
        user_query: str,
        file_summaries: List[Dict],
        syllabus_summary: Optional[str],
        syllabus_doc_id: Optional[str],
        selected_docs: List[str],
        max_files: int
    ) -> List[Dict]:
        """
        Scenario 2: Scoped Refinement (user HAS manually selected files)

        AI-Driven Flow:
        1. AI analyzes query intent
        2. Filter to user's selection
        3. AI identifies anchors WITHIN selection based on content
        4. Extract keywords from AI-selected anchors
        5. Score and prune the selection
        6. Return pruned subset

        Args:
            user_query: User's question
            file_summaries: All file summaries
            syllabus_summary: Syllabus text (ground truth)
            syllabus_doc_id: Syllabus doc ID
            selected_docs: List of doc_ids user selected
            max_files: Max files to return

        Returns:
            Pruned list of selected file dicts (subset of user's selection)
        """
        try:
            print(f"\n   🔬 SCOPED REFINEMENT FLOW (AI-Driven)")
            print(f"      User selected {len(selected_docs)} files")

            # Step 1: AI analyzes query intent
            query_intent = await self._analyze_query_intent(user_query, syllabus_summary)
            print(f"      Main Topic: {query_intent.get('main_topic', 'N/A')}")
            print(f"      Query Type: {query_intent.get('query_type', 'N/A')}")
            print(f"      Context Hints: {query_intent.get('context_hints', [])[:5]}")

            # Step 2: Filter to only user's selection
            selected_set = set(selected_docs)
            scoped_summaries = [f for f in file_summaries if f.get('doc_id') in selected_set]
            print(f"      Scoped to {len(scoped_summaries)} files from user selection")

            # CRITICAL: Ensure syllabus is available even if not in selection
            if syllabus_doc_id and syllabus_doc_id not in selected_set:
                print(f"      ⚠️  Syllabus not in selection, retrieving for ground truth...")
                syllabus_file = next((f for f in file_summaries if f.get('doc_id') == syllabus_doc_id), None)
                if syllabus_file:
                    # Don't add to scoped_summaries (user didn't select it)
                    # But use its summary for context
                    print(f"      ✅ Retrieved syllabus: {syllabus_file.get('filename')}")

            # Step 3: AI identifies anchors WITHIN user's selection
            anchors = await self._identify_anchors_ai(
                query_intent,
                scoped_summaries,
                syllabus_summary,
                max_anchors=3
            )
            print(f"      AI selected {len(anchors)} anchor files in selection")
            for anchor in anchors[:3]:
                reasoning = anchor.get('_anchor_reasoning', 'N/A')
                print(f"         🎯 {anchor['filename']}")
                print(f"            → {reasoning}")

            # Fallback: If AI selection fails, use legacy heuristics
            if not anchors:
                print(f"      ⚠️  AI anchor selection failed, using fallback heuristics")
                anchors = self._identify_anchor_files(scoped_summaries)
                print(f"      Found {len(anchors)} anchor files via fallback")

            # Step 4: Extract keywords from anchors (or fallback to query intent)
            if anchors:
                anchor_keywords = await self._extract_anchor_keywords(anchors, user_query)
                print(f"      Extracted {len(anchor_keywords)} keywords from scoped anchors")
            else:
                print(f"      ⚠️  No anchors in selection, using query context hints as fallback")
                anchor_keywords = query_intent.get('context_hints', [])
                print(f"      Fallback keywords: {anchor_keywords[:5]}")

            # Step 5: Score ONLY the user's selection
            scored_files = await self._score_files_with_anchors(
                scoped_summaries,
                user_query,
                anchor_keywords,
                query_intent.get('context_hints', [])
            )

            # Step 6: Prune - return only relevant files (may be < user's selection)
            # Use a relevance threshold to filter out low-scoring files
            RELEVANCE_THRESHOLD = 0.3  # Minimum score to be considered relevant
            pruned = [f for f in scored_files if f.get('_score', 0) >= RELEVANCE_THRESHOLD]

            # Cap at max_files
            pruned = pruned[:max_files]

            print(f"   ✅ Pruned selection: {len(selected_docs)} → {len(pruned)} files")
            if len(pruned) < len(selected_docs):
                print(f"      Removed {len(selected_docs) - len(pruned)} irrelevant files")

            for file in pruned[:5]:
                print(f"      📄 {file.get('filename')} (score: {file.get('_score', 0):.2f})")

            return pruned

        except Exception as e:
            logger.error(f"Error in scoped refinement: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def _parse_event_from_query(
        self,
        user_query: str,
        syllabus_summary: Optional[str]
    ) -> Dict:
        """
        Parse query + syllabus to extract event, date range, and topics

        Args:
            user_query: User's question
            syllabus_summary: Syllabus text

        Returns:
            Dict with 'event', 'topics', 'date_range'
        """
        try:
            if not syllabus_summary:
                # No syllabus - extract what we can from query alone
                return {
                    'event': self._extract_event_name(user_query),
                    'topics': [],
                    'date_range': None
                }

            # Use AI to parse event from query + syllabus
            prompt = f"""Parse the student's query and the course syllabus to extract:
1. The specific event/assessment (e.g., "Exam 1", "Midterm", "Final")
2. The date range covered by this event (e.g., "Week 1-4", "Sept 1 - Oct 15")
3. The main topics/concepts covered in this period

**Student Query:**
{user_query}

**Course Syllabus:**
{syllabus_summary[:1000]}{"..." if len(syllabus_summary) > 1000 else ""}

**Task:**
Extract the event information in JSON format:
{{
  "event": "name of event (e.g., Exam 1, Midterm)",
  "date_range": "date range if mentioned (e.g., Week 1-4, Sept 1-Oct 15)",
  "topics": ["topic1", "topic2", "topic3", ...]
}}

If the query doesn't mention a specific event, extract general topics being asked about.
Return ONLY the JSON, no other text."""

            response = await self._call_ai_with_fallback(prompt, max_tokens=500)
            if not response:
                return {'event': None, 'topics': [], 'date_range': None}

            # Parse JSON
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]

            parsed = json.loads(response_text.strip())
            return {
                'event': parsed.get('event'),
                'topics': parsed.get('topics', []),
                'date_range': parsed.get('date_range')
            }

        except Exception as e:
            logger.error(f"Error parsing event: {e}")
            return {
                'event': None,
                'topics': [],
                'date_range': None
            }

    async def _analyze_query_intent(
        self,
        user_query: str,
        syllabus_summary: Optional[str]
    ) -> Dict:
        """
        AI analyzes user query to understand intent

        Args:
            user_query: The user's question
            syllabus_summary: Course syllabus summary for context

        Returns:
            Dict with:
                - main_topic: Primary subject of query
                - query_type: Type of query (exam_prep, conceptual, assignment_help, logistics)
                - scope: Level of detail needed (overview, detailed, specific)
                - context_hints: List of related keywords/concepts
        """
        try:
            syllabus_context = ""
            if syllabus_summary:
                syllabus_context = f"\n**Course Syllabus Context:**\n{syllabus_summary[:800]}...\n"

            prompt = f"""Analyze this student's query to understand their intent and needs.

**Student Query:**
{user_query}
{syllabus_context}

**Task:**
Analyze the query and extract:
1. **main_topic**: The primary subject/concept being asked about
2. **query_type**: The type of query - choose one:
   - "exam_prep": Preparing for an exam/test/quiz
   - "assignment_help": Working on homework/project/assignment
   - "conceptual": Understanding concepts/theories
   - "logistics": Course logistics (dates, policies, requirements)
   - "general": General question about course content
3. **scope**: Level of detail needed - choose one:
   - "overview": High-level understanding needed
   - "detailed": In-depth explanation needed
   - "specific": Specific answer to narrow question
4. **context_hints**: List of 5-10 related keywords, concepts, or topics that would be relevant

Return ONLY valid JSON in this format:
{{
  "main_topic": "string",
  "query_type": "exam_prep|assignment_help|conceptual|logistics|general",
  "scope": "overview|detailed|specific",
  "context_hints": ["keyword1", "keyword2", ...]
}}"""

            response = await self._call_ai_with_fallback(prompt, max_tokens=500)
            if not response:
                return {
                    'main_topic': user_query[:100],
                    'query_type': 'general',
                    'scope': 'detailed',
                    'context_hints': []
                }

            # Parse JSON
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]

            intent = json.loads(response_text.strip())
            return intent

        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                'main_topic': user_query[:100],
                'query_type': 'general',
                'scope': 'detailed',
                'context_hints': []
            }

    async def _identify_anchors_ai(
        self,
        query_intent: Dict,
        file_summaries: List[Dict],
        syllabus_summary: Optional[str],
        max_anchors: int = 5
    ) -> List[Dict]:
        """
        AI identifies most authoritative/useful anchor files based on query intent

        Args:
            query_intent: Output from _analyze_query_intent()
            file_summaries: All available file summaries
            syllabus_summary: Course syllabus summary
            max_anchors: Maximum number of anchors to return

        Returns:
            List of selected anchor file dicts with reasoning
        """
        try:
            if not file_summaries:
                return []

            # Build files context
            files_context = self._build_files_context(file_summaries)

            # Build syllabus context
            syllabus_context = ""
            if syllabus_summary:
                syllabus_context = f"\n**Course Syllabus:**\n{syllabus_summary[:600]}...\n"

            prompt = f"""Select 3-5 "anchor" files that would be most useful for answering the student's query.

**Query Analysis:**
- Main Topic: {query_intent.get('main_topic')}
- Query Type: {query_intent.get('query_type')}
- Scope: {query_intent.get('scope')}
- Context Hints: {', '.join(query_intent.get('context_hints', [])[:10])}
{syllabus_context}

**Available Files:**
{files_context}

**Task:**
Identify {max_anchors} "anchor" files that would provide the MOST useful context. Prioritize files that:
1. **Provide structure/overview** (syllabus, module overviews, outlines, study guides)
2. **Define scope** (rubrics, review sheets, exam guides, assignment descriptions)
3. **Are comprehensive** on the main topic (lectures, textbook chapters covering the topic)
4. **Match the specific assessment** if query is about exam/assignment (e.g., "Exam 1 Study Guide" for Exam 1 questions)

Return ONLY valid JSON array of doc_ids with reasoning:
[
  {{
    "doc_id": "string",
    "filename": "string",
    "reasoning": "why this is a good anchor (1 sentence)"
  }},
  ...
]

Select up to {max_anchors} files. Return ONLY the JSON array."""

            response = await self._call_ai_with_fallback(prompt, max_tokens=1000)
            if not response:
                return []

            # Parse JSON
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]

            selected_anchors = json.loads(response_text.strip())

            # Map back to full file objects
            anchor_files = []
            selected_doc_ids = [a.get('doc_id') for a in selected_anchors if isinstance(a, dict)]

            for file_info in file_summaries:
                if file_info.get('doc_id') in selected_doc_ids:
                    # Find reasoning
                    reasoning = next(
                        (a.get('reasoning', '') for a in selected_anchors if a.get('doc_id') == file_info.get('doc_id')),
                        ''
                    )
                    file_info['_anchor_reasoning'] = reasoning
                    file_info['_anchor_score'] = 1.0
                    anchor_files.append(file_info)

            return anchor_files[:max_anchors]

        except Exception as e:
            logger.error(f"Error identifying AI anchors: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _identify_anchor_files(
        self,
        file_summaries: List[Dict],
        date_range: Optional[str] = None,
        scope: str = 'global'
    ) -> List[Dict]:
        """
        DEPRECATED: Legacy method for backwards compatibility
        Now only used as fallback if AI anchor selection fails

        Identify high-authority "anchor" files based on simple heuristics

        Args:
            file_summaries: Files to search
            date_range: Optional date range from syllabus (unused)
            scope: 'global' or 'scoped' (unused)

        Returns:
            List of anchor file dicts
        """
        # Simple heuristic: look for files with authoritative keywords
        anchor_keywords = [
            'study guide', 'review', 'recap', 'rubric', 'study_guide',
            'exam review', 'test review', 'practice', 'summary', 'overview',
            'syllabus', 'outline'
        ]

        anchors = []
        for file_info in file_summaries:
            filename = file_info.get('filename', '').lower()
            if any(keyword in filename for keyword in anchor_keywords):
                file_info['_anchor_score'] = 1.0
                anchors.append(file_info)

        anchors.sort(key=lambda x: x.get('_anchor_score', 0), reverse=True)
        return anchors[:5]

    async def _extract_anchor_keywords(
        self,
        anchor_files: List[Dict],
        user_query: str
    ) -> List[str]:
        """
        Extract specific keywords/concepts from anchor file summaries

        Args:
            anchor_files: List of anchor file dicts
            user_query: User's query for context

        Returns:
            List of extracted keywords
        """
        try:
            if not anchor_files:
                return []

            # Build context from anchor summaries
            anchor_context = ""
            for idx, anchor in enumerate(anchor_files[:5], 1):  # Limit to top 5
                filename = anchor.get('filename', 'Unknown')
                summary = anchor.get('summary', '')
                topics = anchor.get('topics', [])

                # Parse topics if string
                if isinstance(topics, str):
                    try:
                        topics = json.loads(topics)
                    except:
                        topics = []

                topics_str = ", ".join(topics[:5]) if topics else "N/A"

                anchor_context += f"{idx}. **{filename}**\n"
                anchor_context += f"   Topics: {topics_str}\n"
                anchor_context += f"   Summary: {summary[:300]}...\n\n"

            # Use AI to extract keywords
            prompt = f"""Extract specific keywords and concepts from these high-authority course materials (study guides, review sheets).

**Student Query Context:**
{user_query}

**Anchor Materials:**
{anchor_context}

**Task:**
Extract 10-20 specific keywords, concepts, theories, formulas, or terms that appear in these materials and are relevant to the query.

**Response Format:**
Return ONLY a JSON array of keywords:
["keyword1", "keyword2", "keyword3", ...]

Examples of good keywords:
- Specific theories: "Sociobiology", "Natural Selection"
- Formulas: "F=ma", "PV=nRT"
- Concepts: "operant conditioning", "market equilibrium"
- Specific terms: "mitochondria", "GDP deflator"

Return ONLY the JSON array, no other text."""

            response = await self._call_ai_with_fallback(prompt, max_tokens=500)
            if not response:
                return []

            # Parse JSON array
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]

            keywords = json.loads(response_text.strip())
            return keywords if isinstance(keywords, list) else []

        except Exception as e:
            logger.error(f"Error extracting anchor keywords: {e}")
            # Fallback: extract topics from anchor files
            keywords = []
            for anchor in anchor_files[:3]:
                topics = anchor.get('topics', [])
                if isinstance(topics, str):
                    try:
                        topics = json.loads(topics)
                    except:
                        topics = []
                keywords.extend(topics)
            return keywords[:20]

    async def _score_files_with_anchors(
        self,
        file_summaries: List[Dict],
        user_query: str,
        anchor_keywords: List[str],
        syllabus_topics: List[str]
    ) -> List[Dict]:
        """
        Score files based on:
        1. Match with anchor keywords (HIGH PRIORITY)
        2. Match with syllabus topics (MEDIUM PRIORITY)
        3. Match with user query (STANDARD PRIORITY)

        Args:
            file_summaries: Files to score
            user_query: User's query
            anchor_keywords: Keywords extracted from anchors
            syllabus_topics: Topics from syllabus

        Returns:
            List of files sorted by relevance score (descending)
        """
        try:
            # Build scoring context
            anchor_keywords_str = ", ".join(anchor_keywords[:15]) if anchor_keywords else "N/A"
            syllabus_topics_str = ", ".join(syllabus_topics[:10]) if syllabus_topics else "N/A"

            # Build files context
            files_context = self._build_files_context(file_summaries)

            # Use AI to score files
            prompt = f"""Score the relevance of each file based on the user's query and extracted context.

**User Query:**
{user_query}

**High-Priority Keywords** (from study guides/review materials):
{anchor_keywords_str}

**Syllabus Topics:**
{syllabus_topics_str}

**Files to Score:**
{files_context}

**Scoring Criteria (in order of importance):**
1. HIGH (1.0 score): File directly covers anchor keywords (study guide concepts)
2. MEDIUM (0.6 score): File covers syllabus topics relevant to query
3. STANDARD (0.3 score): File relates to user query but not high-priority keywords
4. LOW (0.1 score): File has minimal relevance

**Task:**
Return a JSON object mapping doc_id to relevance score (0.0 to 1.0):
{{
  "doc_id_1": 0.9,
  "doc_id_2": 0.7,
  "doc_id_3": 0.4,
  ...
}}

Return ONLY the JSON, no other text."""

            response = await self._call_ai_with_fallback(prompt, max_tokens=2000)
            if not response:
                # Fallback: return files as-is
                return file_summaries

            # Parse JSON scores
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]

            scores = json.loads(response_text.strip())

            # Assign scores to files
            for file_info in file_summaries:
                doc_id = file_info.get('doc_id')
                file_info['_score'] = scores.get(doc_id, 0.0)

            # Sort by score (descending)
            file_summaries.sort(key=lambda x: x.get('_score', 0), reverse=True)

            return file_summaries

        except Exception as e:
            logger.error(f"Error scoring files: {e}")
            # Fallback: return files as-is
            return file_summaries

    async def _call_ai_with_fallback(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.2
    ) -> Optional[str]:
        """Call AI with fallback to secondary model on rate limit"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            if not response or not response.text:
                return None

            return response.text.strip()

        except Exception as api_error:
            error_str = str(api_error).lower()
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                logger.warning(f"⚠️ Rate limited, trying fallback model")
                try:
                    response = self.client.models.generate_content(
                        model=self.fallback_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                    return response.text.strip() if response and response.text else None
                except:
                    return None
            return None

    def _build_files_context(self, file_summaries: List[Dict]) -> str:
        """Build formatted string of file summaries"""
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
            truncated_summary = summary[:150] + '...' if len(summary) > 150 else summary

            context_lines.append(
                f"{idx}. **{filename}**\n"
                f"   - ID: {doc_id}\n"
                f"   - Topics: {topics_str}\n"
                f"   - Summary: {truncated_summary}\n"
            )

        return "\n".join(context_lines)

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
                    print(f"      ✅ Found syllabus by ID: {file_info.get('filename')}")
                    return file_info.get("summary", "")

        # Search by filename
        print(f"      🔍 Searching for syllabus in {len(file_summaries)} files...")
        for file_info in file_summaries:
            filename = file_info.get("filename", "").lower()
            if "syllabus" in filename:
                print(f"      ✅ Found syllabus by filename: {file_info.get('filename')}")
                return file_info.get("summary", "")

        print(f"      ❌ No syllabus found")
        return None
