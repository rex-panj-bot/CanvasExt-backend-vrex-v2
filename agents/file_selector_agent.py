"""
File Selector Agent V2 - Smart 3-Stage Selection
Intelligently selects relevant files using:
1. Query Understanding - Extract intent, topics, and context
2. Context-Aware Retrieval - Vector search with document type weighting
3. Intelligent Selection - LLM reranking with rich context
"""

import os
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple
import json
import logging
import re

logger = logging.getLogger(__name__)

# Production mode - suppress verbose logging
PRODUCTION_MODE = os.getenv('PRODUCTION', 'true').lower() == 'true'

def debug_print(*args, **kwargs):
    """Print only in development mode"""
    if not PRODUCTION_MODE:
        print(*args, **kwargs)


class FileSelectorAgent:
    """
    Smart File Selector V2 - 3-Stage Selection:
    - Stage 1: Query Understanding (intent, topics, context)
    - Stage 2: Context-Aware Retrieval (vector search + document weighting)
    - Stage 3: Intelligent Selection (LLM reranking with rich context)
    
    V2.1 Update: Works without summaries/embeddings using lightweight filename-based matching
    """

    # Configuration
    MAX_CANDIDATES = 35  # Increased to allow for weighting to filter
    MAX_FILES_HARD_CAP = 18  # Maximum files to return
    MAX_KEYWORD_ANCHORS = 5  # Max anchor docs when using keyword-only matching (no embeddings)
    RELATIVE_THRESHOLD_DELTA = 0.45  # Slightly tighter threshold for better precision

    # Embedding model for query embedding
    EMBEDDING_MODEL = "text-embedding-004"
    
    # Filename patterns for document type detection and topic extraction
    FILENAME_PATTERNS = {
        # Document types
        "lecture": ["lecture", "lec", "class", "lesson", "week"],
        "homework": ["homework", "hw", "assignment", "problem set", "pset", "ps"],
        "exam": ["exam", "midterm", "final", "test", "quiz"],
        "study_guide": ["study guide", "review", "study sheet", "cheat sheet"],
        "notes": ["notes", "note", "summary"],
        "slides": ["slides", "slide", "powerpoint", "ppt", "presentation"],
        "reading": ["reading", "chapter", "textbook", "article", "paper"],
        "syllabus": ["syllabus", "syllabi", "course outline", "schedule"],
        "solution": ["solution", "answer", "key", "solutions"],
        "lab": ["lab", "laboratory", "experiment"],
        "project": ["project", "proj"],
    }
    
    # Common academic topic keywords to extract from filenames
    TOPIC_KEYWORDS = [
        # Sciences
        "biology", "chemistry", "physics", "calculus", "statistics", "algebra",
        "genetics", "cell", "dna", "protein", "evolution", "ecology",
        "organic", "inorganic", "thermodynamics", "mechanics", "waves",
        # Math
        "derivative", "integral", "matrix", "vector", "probability", "regression",
        "function", "equation", "theorem", "proof", "limit",
        # CS
        "algorithm", "data structure", "programming", "database", "network",
        "machine learning", "ai", "neural", "recursion", "sorting",
        # Humanities
        "history", "philosophy", "psychology", "sociology", "economics",
        "literature", "writing", "essay", "analysis", "theory",
        # Business
        "finance", "accounting", "marketing", "management", "strategy",
    ]
    
    # Document type weights for different intents
    DOC_TYPE_WEIGHTS = {
        "study": {
            "study_guide": 1.5,
            "review": 1.5,
            "summary": 1.4,
            "lecture": 1.3,
            "slides": 1.3,
            "notes": 1.2,
            "reading": 1.0,
            "assignment": 0.9,
            "syllabus": 1.1,
            "default": 1.0
        },
        "test_prep": {
            "study_guide": 1.6,
            "review": 1.6,
            "exam": 1.5,
            "practice": 1.5,
            "quiz": 1.4,
            "summary": 1.3,
            "lecture": 1.1,
            "assignment": 0.8,
            "default": 1.0
        },
        "homework_help": {
            "assignment": 1.5,
            "homework": 1.5,
            "problem_set": 1.4,
            "lecture": 1.2,
            "notes": 1.2,
            "example": 1.3,
            "solution": 1.4,
            "default": 1.0
        },
        "concept_explanation": {
            "lecture": 1.4,
            "notes": 1.3,
            "slides": 1.3,
            "reading": 1.2,
            "textbook": 1.2,
            "summary": 1.1,
            "default": 1.0
        },
        "comparison": {
            "lecture": 1.3,
            "notes": 1.3,
            "summary": 1.2,
            "review": 1.2,
            "default": 1.0
        },
        "general": {
            "default": 1.0
        }
    }

    def __init__(self, google_api_key: str, chat_storage=None, file_summarizer=None):
        """
        Initialize File Selector Agent

        Args:
            google_api_key: Google API key for Gemini
            chat_storage: ChatStorage instance for vector search
            file_summarizer: FileSummarizer instance for query embedding
        """
        self.client = genai.Client(api_key=google_api_key)
        self.model_id = "gemini-2.5-flash-lite"  # Fast model for analysis
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
        Select relevant files using 3-stage smart selection

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

            debug_print(f"\n   ✨ SMART FILE SELECTOR V2 (3-Stage)")
            debug_print(f"      Query: {user_query[:100]}...")
            debug_print(f"      Available summaries: {len(file_summaries)}")
            debug_print(f"      Manual selection: {len(selected_docs) if selected_docs else 'None (global)'}")

            if not file_summaries:
                debug_print(f"      No file summaries available")
                return [], []

            # Get syllabus summary if not provided but doc_id is given
            if not syllabus_summary and syllabus_doc_id:
                syllabus_summary = await self.get_syllabus_summary(syllabus_doc_id, file_summaries)

            # Determine working set of files
            if selected_docs and len(selected_docs) > 0:
                # Scoped mode: Filter to user's selection
                selected_set = set(selected_docs)
                working_summaries = [f for f in file_summaries if f.get('doc_id') in selected_set]
                debug_print(f"      Scoped to {len(working_summaries)} user-selected files")
            else:
                working_summaries = file_summaries

            # ========== STAGE 1: Query Understanding ==========
            debug_print(f"\n   📊 STAGE 1: Query Understanding")
            query_analysis = await self._analyze_query(user_query, syllabus_summary, user_api_key)
            
            if query_analysis:
                debug_print(f"      Intent: {query_analysis.get('intent', 'unknown')}")
                debug_print(f"      Topics: {query_analysis.get('topics', [])[:5]}")
                debug_print(f"      Specificity: {query_analysis.get('specificity', 'unknown')}")
                debug_print(f"      Question Type: {query_analysis.get('question_type', 'unknown')}")
                if query_analysis.get('syllabus_modules'):
                    debug_print(f"      Syllabus Modules: {query_analysis.get('syllabus_modules', [])[:3]}")
            else:
                debug_print(f"      ⚠️ Query analysis failed, using defaults")
                query_analysis = {
                    "intent": "general",
                    "topics": [],
                    "specificity": "broad",
                    "temporal_context": "general",
                    "question_type": "what"
                }

            # ========== STAGE 2: Context-Aware Retrieval ==========
            debug_print(f"\n   🔍 STAGE 2: Context-Aware Retrieval")
            candidates = await self._context_aware_retrieval(
                user_query,
                course_id,
                working_summaries,
                query_analysis
            )

            if not candidates:
                debug_print(f"      Vector search returned no candidates, using all files")
                candidates = working_summaries[:self.MAX_CANDIDATES]

            debug_print(f"      Retrieved {len(candidates)} candidates")

            # ========== STAGE 3: Intelligent Selection ==========
            debug_print(f"\n   🎯 STAGE 3: Intelligent Selection")
            reranked_results = await self._intelligent_rerank(
                user_query,
                candidates,
                query_analysis,
                syllabus_summary,
                user_api_key
            )

            if not reranked_results:
                debug_print(f"      LLM reranking failed, using weighted scores")
                # Fallback: use weighted similarity scores
                for c in candidates:
                    c['_score'] = c.get('weighted_score', c.get('similarity_score', 0.5))
                    c['_reason'] = "Selected by semantic similarity"
                candidates.sort(key=lambda x: x['_score'], reverse=True)
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

            debug_print(f"\n   ✅ Final Selection: {len(selected_files)} files")
            for r in reasoning_output[:5]:
                debug_print(f"      • {r['filename'][:40]}... (score: {r['score']:.2f})")

            return selected_files, reasoning_output

        except Exception as e:
            logger.error(f"Error in file selection: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    async def _analyze_query(
        self,
        user_query: str,
        syllabus_summary: Optional[str],
        user_api_key: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Stage 1: Query Understanding
        
        Analyzes the user's query to extract:
        - intent: study, test_prep, homework_help, concept_explanation, comparison
        - topics: key concepts/terms mentioned
        - specificity: broad, focused, specific
        - temporal_context: exam_prep, weekly_review, assignment, general
        - question_type: what, how, why, compare, calculate
        - syllabus_modules: relevant course modules if syllabus available
        
        Args:
            user_query: User's question
            syllabus_summary: Course syllabus for module mapping
            user_api_key: Optional user API key
            
        Returns:
            Dict with query analysis or None on error
        """
        try:
            syllabus_context = ""
            if syllabus_summary:
                syllabus_context = f"""
**Course Syllabus Summary:**
{syllabus_summary[:1500]}

Based on the syllabus, identify which course modules/weeks/units are most relevant to the query.
"""

            prompt = f"""Analyze this student's question to understand what they're looking for.

**Student Question:**
{user_query}
{syllabus_context}

**Analyze and return JSON with:**

1. **intent** - What is the student trying to do?
   - "study" = General studying/learning a topic
   - "test_prep" = Preparing for exam/quiz/test
   - "homework_help" = Working on an assignment
   - "concept_explanation" = Understanding a specific concept
   - "comparison" = Comparing two or more things
   - "general" = Unclear/general question

2. **topics** - Key concepts, terms, or subjects mentioned (list of 2-8 strings)

3. **specificity** - How focused is the question?
   - "broad" = General topic, multiple possible answers
   - "focused" = Specific topic area
   - "specific" = Very specific concept or question

4. **temporal_context** - Time-related context
   - "exam_prep" = Near exam time, studying for test
   - "weekly_review" = Regular study session
   - "assignment" = Working on homework/project
   - "general" = No clear temporal context

5. **question_type** - Type of question being asked
   - "what" = Definition/description
   - "how" = Process/mechanism
   - "why" = Explanation/reasoning
   - "compare" = Comparison/contrast
   - "calculate" = Math/computation
   - "apply" = Application of concept
   - "list" = Enumerate items

6. **syllabus_modules** - If syllabus provided, list relevant module/week/unit names (max 3)

Return ONLY valid JSON:
{{"intent": "...", "topics": [...], "specificity": "...", "temporal_context": "...", "question_type": "...", "syllabus_modules": [...]}}"""

            response = await self._call_ai_with_fallback(
                prompt, 
                max_tokens=500, 
                temperature=0.1,
                user_api_key=user_api_key
            )

            if not response:
                return None

            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0]

            analysis = json.loads(response_text.strip())
            
            # Validate required fields
            required_fields = ["intent", "topics", "specificity", "temporal_context", "question_type"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = "general" if field != "topics" else []
                    
            return analysis

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return None

    async def _context_aware_retrieval(
        self,
        user_query: str,
        course_id: str,
        fallback_summaries: List[Dict],
        query_analysis: Dict
    ) -> List[Dict]:
        """
        Stage 2: Context-Aware Retrieval
        
        Performs vector search and applies document type weighting based on query intent.
        Falls back to lightweight keyword-based matching when no embeddings exist.
        
        Args:
            user_query: User's question
            course_id: Course ID for database lookup
            fallback_summaries: Summaries to use if vector search unavailable
            query_analysis: Results from Stage 1
            
        Returns:
            List of candidate file dicts with similarity_score and weighted_score
        """
        try:
            # First, try vector search if available
            vector_candidates = None
            
            if self.chat_storage and self.file_summarizer and course_id:
                # Generate query embedding
                topics = query_analysis.get('topics', [])
                enhanced_query = user_query
                if topics:
                    enhanced_query = f"{user_query}\n\nKey topics: {', '.join(topics)}"
                
                query_embedding = await self.file_summarizer.generate_query_embedding(enhanced_query)

                if query_embedding:
                    # Perform vector search
                    vector_candidates = self.chat_storage.search_similar_summaries(
                        course_id=course_id,
                        query_embedding=query_embedding,
                        limit=self.MAX_CANDIDATES
                    )
                    
                    if vector_candidates:
                        debug_print(f"      Vector search found {len(vector_candidates)} candidates")
            
            # If vector search succeeded, use those results
            if vector_candidates:
                weighted_candidates = self._apply_document_weighting(vector_candidates, query_analysis)
                weighted_candidates.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
                
                for c in weighted_candidates[:3]:
                    debug_print(f"         {c.get('filename', 'Unknown')[:40]}... "
                               f"(sim: {c.get('similarity_score', 0):.3f}, "
                               f"weighted: {c.get('weighted_score', 0):.3f})")
                return weighted_candidates
            
            # FALLBACK: Use lightweight keyword-based matching
            # This handles the case when summaries/embeddings don't exist
            debug_print(f"      Vector search unavailable or returned no results")
            debug_print(f"      Using lightweight keyword-based matching...")
            
            candidates = fallback_summaries[:self.MAX_CANDIDATES] if fallback_summaries else []
            
            if not candidates:
                debug_print(f"      No fallback summaries available")
                return []
            
            # Apply lightweight keyword matching to compute similarity scores
            candidates = self._lightweight_keyword_matching(candidates, user_query, query_analysis)
            
            # Apply document type weighting on top of keyword scores
            weighted_candidates = self._apply_document_weighting(candidates, query_analysis)
            
            # Sort by weighted score
            weighted_candidates.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
            
            # Cap at MAX_KEYWORD_ANCHORS for keyword-only mode (no embeddings)
            # This prevents slow responses when courses have many files
            weighted_candidates = weighted_candidates[:self.MAX_KEYWORD_ANCHORS]
            
            # Log top candidates
            for c in weighted_candidates[:3]:
                debug_print(f"         {c.get('filename', 'Unknown')[:40]}... "
                           f"(keyword: {c.get('similarity_score', 0):.3f}, "
                           f"weighted: {c.get('weighted_score', 0):.3f})")
            
            debug_print(f"      ⚡ Keyword mode: capped at {len(weighted_candidates)} anchor docs")

            return weighted_candidates

        except Exception as e:
            logger.error(f"Error in context-aware retrieval: {e}")
            # Ultimate fallback - keyword matching with anchor cap
            if fallback_summaries:
                candidates = self._lightweight_keyword_matching(
                    fallback_summaries[:self.MAX_CANDIDATES], 
                    user_query, 
                    query_analysis
                )
                weighted = self._apply_document_weighting(candidates, query_analysis)
                weighted.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
                return weighted[:self.MAX_KEYWORD_ANCHORS]  # Cap at anchor limit
            return []
    
    def _lightweight_keyword_matching(
        self,
        candidates: List[Dict],
        user_query: str,
        query_analysis: Dict
    ) -> List[Dict]:
        """
        Lightweight keyword-based matching when embeddings are unavailable.
        
        Uses:
        - Query topics from Stage 1
        - Filename parsing for document type and topics
        - Simple keyword overlap scoring
        
        Args:
            candidates: List of file candidates (from summaries or catalog)
            user_query: User's question
            query_analysis: Results from Stage 1
            
        Returns:
            Candidates with similarity_score added based on keyword matching
        """
        query_lower = user_query.lower()
        query_words = set(query_lower.split())
        query_topics = set(t.lower() for t in query_analysis.get('topics', []))
        
        # Combine query words and extracted topics
        all_query_terms = query_words | query_topics
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                      'as', 'until', 'while', 'about', 'against', 'what', 'which',
                      'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'i',
                      'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your',
                      'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them'}
        
        all_query_terms = all_query_terms - stop_words
        
        for candidate in candidates:
            # Get all text to search in
            filename = candidate.get('filename', '').lower()
            # Also check 'name' field (display name from catalog)
            display_name = candidate.get('name', '').lower()
            summary = candidate.get('summary', '').lower()
            topics = candidate.get('topics', [])
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics)
                except:
                    topics = []
            topics_str = ' '.join(t.lower() for t in topics) if topics else ''
            
            # Combine all searchable text
            search_text = f"{filename} {display_name} {summary} {topics_str}"
            search_words = set(search_text.split())
            
            # Extract topics from filename
            filename_topics = self._extract_topics_from_filename(filename or display_name)
            search_words.update(filename_topics)
            
            # Calculate keyword overlap score
            overlap = all_query_terms & search_words
            
            # Base score from keyword overlap
            if all_query_terms:
                base_overlap_score = len(overlap) / len(all_query_terms)
            else:
                base_overlap_score = 0.0
            
            # Boost for exact phrase matches
            phrase_boost = 0.0
            for topic in query_topics:
                if len(topic) > 3 and topic in search_text:
                    phrase_boost += 0.15  # Bonus for exact topic match
            
            # Boost for filename containing query terms
            filename_boost = 0.0
            combined_filename = f"{filename} {display_name}"
            for term in all_query_terms:
                if len(term) > 3 and term in combined_filename:
                    filename_boost += 0.1  # Bonus for query term in filename
            
            # Calculate final similarity score
            similarity_score = min(1.0, base_overlap_score + phrase_boost + filename_boost)
            
            # Ensure minimum score if any match exists
            if overlap and similarity_score < 0.1:
                similarity_score = 0.1
            
            # If no keyword matches but file exists, give small base score
            # (will be weighted by document type later)
            if similarity_score == 0:
                similarity_score = 0.05  # Small base score
            
            candidate['similarity_score'] = similarity_score
            candidate['keyword_matches'] = list(overlap)[:10]  # Store matches for debugging
        
        return candidates
    
    def _extract_topics_from_filename(self, filename: str) -> set:
        """
        Extract potential topics from a filename.
        
        Parses patterns like:
        - "Lecture 5 - Cell Division.pdf" → {"cell", "division", "lecture"}
        - "HW3_Genetics_Problems.pdf" → {"genetics", "problems", "homework"}
        - "Midterm_Review_Guide.pdf" → {"midterm", "review", "guide", "exam"}
        
        Args:
            filename: The filename to parse
            
        Returns:
            Set of extracted topic keywords
        """
        if not filename:
            return set()
        
        # Remove extension
        name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        
        # Normalize: replace common separators with spaces
        for sep in ['_', '-', '.', '(', ')', '[', ']', '{', '}']:
            name = name.replace(sep, ' ')
        
        # Split into words and lowercase
        words = name.lower().split()
        
        extracted = set()
        
        # Add all words that are long enough
        for word in words:
            # Skip pure numbers and very short words
            if len(word) > 2 and not word.isdigit():
                extracted.add(word)
        
        # Check for known topic keywords
        name_lower = name.lower()
        for keyword in self.TOPIC_KEYWORDS:
            if keyword in name_lower:
                extracted.add(keyword)
        
        # Check for document type patterns and add related terms
        for doc_type, patterns in self.FILENAME_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    extracted.add(doc_type)
                    # Add related search terms
                    if doc_type == "homework":
                        extracted.update(["assignment", "problem"])
                    elif doc_type == "exam":
                        extracted.update(["test", "midterm", "final", "quiz"])
                    elif doc_type == "lecture":
                        extracted.update(["class", "notes", "slides"])
                    elif doc_type == "study_guide":
                        extracted.update(["review", "summary"])
                    break
        
        return extracted

    def _apply_document_weighting(
        self,
        candidates: List[Dict],
        query_analysis: Dict
    ) -> List[Dict]:
        """
        Apply document type weighting based on query intent
        
        Args:
            candidates: List of file candidates
            query_analysis: Query understanding results
            
        Returns:
            Candidates with weighted_score added
        """
        intent = query_analysis.get('intent', 'general')
        weights = self.DOC_TYPE_WEIGHTS.get(intent, self.DOC_TYPE_WEIGHTS['general'])
        default_weight = weights.get('default', 1.0)
        
        # Keywords for document type detection
        doc_type_keywords = {
            "study_guide": ["study guide", "study sheet", "review guide", "study notes"],
            "review": ["review", "final review", "midterm review", "exam review"],
            "summary": ["summary", "overview", "recap"],
            "lecture": ["lecture", "lec", "class notes", "lesson"],
            "slides": ["slides", "powerpoint", "ppt", "presentation"],
            "notes": ["notes", "note"],
            "reading": ["reading", "chapter", "textbook", "article"],
            "assignment": ["assignment", "homework", "hw", "problem set", "pset"],
            "homework": ["homework", "hw"],
            "problem_set": ["problem set", "pset", "problems"],
            "exam": ["exam", "midterm", "final", "test"],
            "practice": ["practice", "sample", "old exam"],
            "quiz": ["quiz"],
            "example": ["example", "sample", "demo"],
            "solution": ["solution", "answer", "key"],
            "syllabus": ["syllabus"]
        }
        
        for candidate in candidates:
            filename = candidate.get('filename', '').lower()
            summary = candidate.get('summary', '').lower()
            metadata = candidate.get('metadata', {})
            
            # Get document type from metadata if available
            doc_type_from_metadata = metadata.get('doc_type', '').lower() if isinstance(metadata, dict) else ''
            
            # Detect document type from filename and summary
            detected_type = None
            search_text = f"{filename} {summary} {doc_type_from_metadata}"
            
            for doc_type, keywords in doc_type_keywords.items():
                for keyword in keywords:
                    if keyword in search_text:
                        detected_type = doc_type
                        break
                if detected_type:
                    break
            
            # Get weight for this document type
            weight = weights.get(detected_type, default_weight) if detected_type else default_weight
            
            # Apply weight to similarity score
            base_score = candidate.get('similarity_score', 0.5)
            weighted_score = base_score * weight
            
            candidate['detected_doc_type'] = detected_type or 'unknown'
            candidate['doc_weight'] = weight
            candidate['weighted_score'] = min(weighted_score, 1.0)  # Cap at 1.0
            
        return candidates

    async def _intelligent_rerank(
        self,
        user_query: str,
        candidates: List[Dict],
        query_analysis: Dict,
        syllabus_summary: Optional[str],
        user_api_key: Optional[str] = None
    ) -> List[Dict]:
        """
        Stage 3: Intelligent Selection with LLM reranking
        
        Single LLM call with rich context including:
        - Query analysis results
        - Syllabus module mapping
        - Candidate files with enhanced metadata
        
        Args:
            user_query: User's question
            candidates: Candidate files from Stage 2
            query_analysis: Results from Stage 1
            syllabus_summary: Optional syllabus for context
            user_api_key: User's API key
            
        Returns:
            List of dicts with file_id, score (0-1), reason
        """
        try:
            if not candidates:
                return []

            # Build files context for prompt
            files_context = self._build_enhanced_files_context(candidates)

            # Build query analysis context
            query_context = self._build_query_analysis_context(query_analysis)
            
            syllabus_context = ""
            if syllabus_summary:
                syllabus_context = f"\n**Course Context (from syllabus):**\n{syllabus_summary[:600]}...\n"

            # Dynamic scoring guidelines based on intent
            scoring_guidelines = self._get_scoring_guidelines(query_analysis)

            prompt = f"""You are an intelligent file selector for a student's study assistant. Select the most relevant files for the student's question.

**Student Question:**
{user_query}

{query_context}
{syllabus_context}

**Candidate Files ({len(candidates)} total):**
{files_context}

**SCORING GUIDELINES FOR THIS QUERY:**
{scoring_guidelines}

**Task:**
1. Consider the student's intent ({query_analysis.get('intent', 'general')}) and topics ({', '.join(query_analysis.get('topics', [])[:5])})
2. Analyze each file's relevance based on:
   - Topic match with student's question
   - Document type appropriateness for their intent
   - Specificity match (if specific question, prefer focused docs)
3. Assign relevance score (0.0 to 1.0) following the scoring guidelines
4. Provide a brief reason (1 sentence, displayed to student)

**Output Format:**
Return ONLY a JSON array, ordered by relevance (highest first).
CRITICAL: Use the EXACT file_id string (the long hash).

[
  {{
    "file_id": "1424277_abc123def456...",
    "score": 0.95,
    "reason": "Contains the study guide covering mitosis and cell division"
  }}
]

Include files with score >= 0.15. Return ONLY the JSON array."""

            debug_print(f"      Calling LLM for intelligent reranking...")
            response = await self._call_ai_with_fallback(
                prompt, 
                max_tokens=16000, 
                user_api_key=user_api_key
            )

            if not response:
                debug_print(f"      ⚠️ LLM reranking returned no response")
                return []

            debug_print(f"      ✅ LLM response received ({len(response)} chars)")
            
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

            for r in results:
                if not isinstance(r, dict):
                    continue
                file_id = r.get('file_id')
                if file_id not in candidate_ids:
                    continue

                valid_results.append({
                    'file_id': file_id,
                    'score': min(1.0, max(0.0, float(r.get('score', 0.5)))),
                    'reason': str(r.get('reason', 'Selected as relevant'))[:200]
                })

            # Sort by score descending
            valid_results.sort(key=lambda x: x['score'], reverse=True)
            
            debug_print(f"      Validated {len(valid_results)} results")

            return valid_results

        except Exception as e:
            debug_print(f"      ❌ Error in intelligent reranking: {e}")
            logger.error(f"Error in intelligent reranking: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _build_query_analysis_context(self, query_analysis: Dict) -> str:
        """Build formatted string of query analysis for LLM prompt"""
        lines = ["**Query Analysis:**"]
        
        intent_descriptions = {
            "study": "The student is studying/learning about this topic",
            "test_prep": "The student is preparing for an exam or test",
            "homework_help": "The student needs help with an assignment",
            "concept_explanation": "The student wants to understand a specific concept",
            "comparison": "The student wants to compare/contrast concepts",
            "general": "General question about the topic"
        }
        
        intent = query_analysis.get('intent', 'general')
        lines.append(f"- Intent: {intent} - {intent_descriptions.get(intent, '')}")
        lines.append(f"- Key Topics: {', '.join(query_analysis.get('topics', ['general']))}")
        lines.append(f"- Question Specificity: {query_analysis.get('specificity', 'broad')}")
        lines.append(f"- Question Type: {query_analysis.get('question_type', 'what')} question")
        
        if query_analysis.get('syllabus_modules'):
            lines.append(f"- Relevant Course Modules: {', '.join(query_analysis.get('syllabus_modules', []))}")
        
        temporal = query_analysis.get('temporal_context', 'general')
        if temporal != 'general':
            temporal_desc = {
                "exam_prep": "(likely near exam time)",
                "weekly_review": "(regular study session)",
                "assignment": "(working on homework)"
            }
            lines.append(f"- Context: {temporal_desc.get(temporal, '')}")
            
        return "\n".join(lines)

    def _build_enhanced_files_context(self, file_summaries: List[Dict]) -> str:
        """Build formatted string of file summaries with enhanced metadata for LLM prompt"""
        context_lines = []

        for idx, file_info in enumerate(file_summaries, 1):
            doc_id = file_info.get("doc_id", "unknown")
            filename = file_info.get("filename", "unknown")
            summary = file_info.get("summary", "")
            topics = file_info.get("topics", [])
            doc_type = file_info.get("detected_doc_type", "unknown")
            weighted_score = file_info.get("weighted_score", 0)

            # Parse topics if string
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics)
                except:
                    topics = []

            topics_str = ", ".join(topics[:5]) if topics else "N/A"
            truncated_summary = summary[:180] + '...' if len(summary) > 180 else summary

            # Format: Include detected doc type and relevance score
            context_lines.append(
                f"{idx}. file_id: \"{doc_id}\"\n"
                f"   Filename: {filename}\n"
                f"   Doc Type: {doc_type}\n"
                f"   Relevance Score: {weighted_score:.2f}\n"
                f"   Topics: {topics_str}\n"
                f"   Summary: {truncated_summary}\n"
            )

        return "\n".join(context_lines)

    def _get_scoring_guidelines(self, query_analysis: Dict) -> str:
        """Get scoring guidelines based on query intent"""
        intent = query_analysis.get('intent', 'general')
        specificity = query_analysis.get('specificity', 'broad')
        question_type = query_analysis.get('question_type', 'what')
        
        base_guidelines = """
GENERAL PRINCIPLES:
- Higher scores for documents that directly address the question's topics
- Prefer documents matching the student's learning intent
- Consider document type appropriateness for the task
"""
        
        intent_guidelines = {
            "study": """
FOR STUDYING/LEARNING:
HIGH VALUE (0.85-1.0):
- Study guides, review sheets covering the topic
- Lecture notes/slides on the specific topic
- Summaries and overviews

MEDIUM VALUE (0.60-0.84):
- Related lectures that provide context
- Readings on the topic

LOW VALUE (<0.60):
- Assignments without explanations
- Administrative documents
""",
            "test_prep": """
FOR EXAM PREPARATION:
HIGH VALUE (0.85-1.0):
- Study guides and review materials
- Practice exams, sample questions
- Exam review documents
- Summary/overview documents

MEDIUM VALUE (0.60-0.84):
- Lecture notes on key topics
- Quiz materials

LOW VALUE (<0.60):
- General readings
- Assignments (unless practice-focused)
""",
            "homework_help": """
FOR HOMEWORK/ASSIGNMENT HELP:
HIGH VALUE (0.85-1.0):
- Assignment descriptions and instructions
- Lecture notes explaining relevant concepts
- Examples and worked problems
- Solution guides (if available)

MEDIUM VALUE (0.60-0.84):
- Related lecture slides
- Textbook readings on the topic

LOW VALUE (<0.60):
- Unrelated assignments
- General course documents
""",
            "concept_explanation": """
FOR UNDERSTANDING A CONCEPT:
HIGH VALUE (0.85-1.0):
- Lecture notes/slides explaining the concept
- Readings/textbook sections on the topic
- Summary documents

MEDIUM VALUE (0.60-0.84):
- Related lectures providing context
- Examples demonstrating the concept

LOW VALUE (<0.60):
- Assignments without explanations
- Administrative documents
""",
            "comparison": """
FOR COMPARING/CONTRASTING:
HIGH VALUE (0.85-1.0):
- Documents covering multiple topics being compared
- Lecture notes discussing relationships
- Summary/review materials

MEDIUM VALUE (0.60-0.84):
- Documents covering individual topics

LOW VALUE (<0.60):
- Unrelated materials
"""
        }
        
        specificity_note = ""
        if specificity == "specific":
            specificity_note = "\n⚠️ SPECIFIC QUESTION: Prefer documents that narrowly focus on the exact topic. Penalize overly broad documents."
        elif specificity == "broad":
            specificity_note = "\n📚 BROAD QUESTION: Include a variety of materials for comprehensive coverage."
        
        return base_guidelines + intent_guidelines.get(intent, "") + specificity_note

    def _apply_dynamic_filtering(
        self,
        reranked_results: List[Dict],
        max_files: int
    ) -> List[Dict]:
        """
        Apply dynamic filtering with relative scoring

        Rules:
        1. Hard cap at max_files (default 18)
        2. Dynamic threshold: cutoff = max_score - RELATIVE_THRESHOLD_DELTA
           - If best match is 0.95, cutoff is 0.50 (drop poor matches)
           - If best match is 0.40, cutoff is ~0 (keep weak matches)

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

        debug_print(f"      Dynamic threshold: max_score={max_score:.2f}, cutoff={cutoff:.2f}")

        # Filter by cutoff
        filtered = [r for r in reranked_results if r.get('score', 0) >= cutoff]

        # Apply hard cap
        filtered = filtered[:max_files]

        debug_print(f"      After filtering: {len(filtered)} files (from {len(reranked_results)})")

        return filtered

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
        else:
            client = self.client

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
                return None

            return response.text.strip()

        except Exception as api_error:
            error_str = str(api_error).lower()
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                debug_print(f"      🔄 Rate limited, trying fallback model {self.fallback_model}")
                try:
                    response = await asyncio.to_thread(
                        client.models.generate_content,
                        model=self.fallback_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                    return response.text.strip() if response and response.text else None
                except Exception as fallback_error:
                    debug_print(f"      ❌ Fallback also failed: {fallback_error}")
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
