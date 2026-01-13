"""
Multi-Model Batch Processor for Parallel Summarization
Distributes batch summarization across multiple Gemini models with proactive rate limiting
"""

import os
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Production mode - suppress verbose logging
PRODUCTION_MODE = os.getenv('PRODUCTION', 'true').lower() == 'true'

def debug_debug_print(*args, **kwargs):
    """Print only in development mode"""
    if not PRODUCTION_MODE:
        debug_print(*args, **kwargs)


class ProactiveRateLimiter:
    """Tracks and prevents rate limit violations BEFORE they happen"""

    def __init__(self, models: List[Dict]):
        """
        Initialize rate limiter with model configurations

        Args:
            models: List of dicts with 'name', 'rpm', 'rpd' keys
        """
        self.models = {m['name']: m for m in models}
        self.minute_start = time.time()
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    def can_submit_request(self, model_name: str, estimated_tokens: int = 0) -> Tuple[bool, float]:
        """
        Check if we can submit a request WITHOUT hitting rate limits

        Args:
            model_name: Name of the model to check
            estimated_tokens: Estimated tokens for this request (input + output)

        Returns:
            (can_submit: bool, wait_time: float)
            - can_submit: True if request can be submitted now
            - wait_time: Seconds to wait if can't submit (0.0 if can submit)
        """
        model = self.models[model_name]
        current_time = time.time()

        # Reset minute window if needed (every 60 seconds)
        if current_time - self.minute_start >= 60:
            self.minute_start = current_time
            model['request_count'] = 0
            model['token_count'] = 0  # Reset token counter

        # Reset daily counter if new day
        now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if now > self.day_start:
            self.day_start = now
            model['daily_count'] = 0
            model['daily_token_count'] = 0  # Reset daily tokens

        # PROACTIVE CHECK: Leave buffer to prevent edge cases
        minute_capacity = model['request_count'] < (model['rpm'] - 1)
        daily_capacity = model['daily_count'] < (model['rpd'] - 5)  # 5-request daily buffer

        # Token limit checks (90% safety margin)
        token_minute_capacity = (model['token_count'] + estimated_tokens) < (model['tpm'] * 0.9)
        token_daily_capacity = (model['daily_token_count'] + estimated_tokens) < (model['tpd'] * 0.9)

        if minute_capacity and daily_capacity and token_minute_capacity and token_daily_capacity:
            return True, 0.0
        elif not minute_capacity or not token_minute_capacity:
            # Wait for next minute window
            wait_time = 60 - (current_time - self.minute_start) + 0.5  # +0.5s safety margin
            return False, wait_time
        else:
            # Daily limit approaching - this model unavailable
            return False, float('inf')

    def reserve_request(self, model_name: str, estimated_tokens: int = 0):
        """
        Reserve a request slot (call BEFORE making API call)

        Args:
            model_name: Name of the model
            estimated_tokens: Estimated tokens for this request
        """
        model = self.models[model_name]
        model['request_count'] += 1
        model['daily_count'] += 1
        model['token_count'] += estimated_tokens
        model['daily_token_count'] += estimated_tokens

    def get_stats(self) -> Dict:
        """Get current rate limit usage statistics"""
        stats = {}
        for name, model in self.models.items():
            stats[name] = {
                'rpm_used': model['request_count'],
                'rpm_limit': model['rpm'],
                'rpd_used': model['daily_count'],
                'rpd_limit': model['rpd'],
                'rpm_remaining': model['rpm'] - model['request_count'],
                'rpd_remaining': model['rpd'] - model['daily_count'],
                # Token stats
                'tpm_used': model.get('token_count', 0),
                'tpm_limit': model.get('tpm', 0),
                'tpd_used': model.get('daily_token_count', 0),
                'tpd_limit': model.get('tpd', 0),
                'tpm_remaining': model.get('tpm', 0) - model.get('token_count', 0),
                'tpd_remaining': model.get('tpd', 0) - model.get('daily_token_count', 0)
            }
        return stats


def estimate_batch_tokens(batch: List[Dict]) -> int:
    """
    Estimate total tokens for a batch (input + output)

    Uses multiple estimation methods:
    - Text: 4 characters = 1 token
    - Binary/Images: 750 bytes = 1 token (conservative estimate)
    - Adds 500 tokens per file for output summary

    Args:
        batch: List of file dicts with 'text_content' or 'file_size'

    Returns:
        Estimated total tokens
    """
    input_tokens = 0

    for f in batch:
        # Try to get text content length first
        text_content = f.get('text_content', '')
        if text_content:
            # Text-based estimation: 4 chars = 1 token
            input_tokens += len(text_content) // 4
        else:
            # Fallback to file size for binary files (images, PDFs, etc.)
            # Conservative estimate: 750 bytes = 1 token for binary data
            file_size = f.get('file_size', 0)
            if file_size > 0:
                input_tokens += file_size // 750
            else:
                # Default fallback: assume 2000 tokens per file if no info available
                input_tokens += 2000

    output_tokens = len(batch) * 500  # 500 tokens per summary (conservative)
    total = input_tokens + output_tokens

    # Add 10% safety margin
    return int(total * 1.1)


class MultiModelBatchProcessor:
    """
    Distributes batch summarization across multiple Gemini models in parallel

    Proactive Strategy:
    1. Submit batches at start of minute window for max throughput
    2. Distribute across 3 models based on their RPM capacity
    3. Track usage to prevent hitting rate limits BEFORE they occur
    4. Process all batches in parallel with asyncio.gather()
    """

    def __init__(self, google_api_key: str):
        """
        Initialize multi-model batch processor

        Args:
            google_api_key: Google API key for Gemini models
        """
        from .file_summarizer import FileSummarizer

        # Initialize 2 models with UPDATED Dec 2025 rate limits
        # Google reduced quotas significantly on Dec 7, 2025
        # Using only gemini-2.5-flash-lite (best free tier limits) and gemini-2.0-flash as backup
        # Note: gemini-2.0-flash-lite has been deprecated, use 2.5 variants instead
        self.models = [
            {
                'name': 'gemini-2.5-flash-lite',
                'summarizer': FileSummarizer(google_api_key, 'gemini-2.5-flash-lite'),
                'rpm': 15,             # Dec 2025: 15 RPM free tier
                'rpd': 1000,           # Dec 2025: 1000 RPD (best available)
                'tpm': 250_000,        # Tokens per minute limit
                'tpd': 6_000_000,      # Tokens per day
                'weight': 0.75,        # Primary model - 75% of batches
                'request_count': 0,
                'daily_count': 0,
                'token_count': 0,
                'daily_token_count': 0
            },
            {
                'name': 'gemini-2.0-flash',
                'summarizer': FileSummarizer(google_api_key, 'gemini-2.0-flash'),
                'rpm': 10,             # Dec 2025: Reduced to 10 RPM
                'rpd': 1500,           # Dec 2025: 1500 RPD free tier
                'tpm': 1_000_000,      # Tokens per minute limit
                'tpd': 4_000_000,      # Tokens per day
                'weight': 0.25,        # Backup model - 25% of batches
                'request_count': 0,
                'daily_count': 0,
                'token_count': 0,
                'daily_token_count': 0
            }
        ]

        # Total capacity: 60 RPM (2x improvement over current 30 RPM)
        self.total_rpm = sum(m['rpm'] for m in self.models)

        # Proactive rate limiter
        self.rate_limiter = ProactiveRateLimiter(self.models)

    def _get_available_model(self, estimated_tokens: int = 0) -> Optional[Dict]:
        """
        Get a model that has capacity (proactive selection)

        Args:
            estimated_tokens: Estimated tokens for the batch

        Returns:
            Model dict or None if all models at capacity
        """
        # Try models in order of priority (highest RPM first)
        for model in sorted(self.models, key=lambda m: m['rpm'], reverse=True):
            can_submit, wait_time = self.rate_limiter.can_submit_request(model['name'], estimated_tokens)
            if can_submit:
                return model

        return None

    async def process_batches_parallel(
        self,
        batches: List[List[Dict]],
        course_id: str,
        file_uploader,
        chat_storage,
        canvas_user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Process ALL batches in parallel across multiple models

        Args:
            batches: List of file batches to summarize
            course_id: Canvas course ID
            file_uploader: FileUploadManager instance
            chat_storage: ChatStorage instance
            canvas_user_id: Optional Canvas user ID

        Returns:
            List of result dicts with status for each file
        """
        debug_print(f"🚀 PARALLEL PROCESSING: {len(batches)} batches across {len(self.models)} models")
        debug_print(f"   Max throughput: {self.total_rpm} batches/minute (vs. 30 sequential)")

        # Phase 1: Assign batches to models and create tasks (start immediately, no waiting)
        tasks = []
        batch_assignments = []

        for batch_idx, batch in enumerate(batches):
            # Estimate tokens for this batch
            estimated_tokens = estimate_batch_tokens(batch)

            # Get model with available capacity (considering token limits)
            model = await self._get_or_wait_for_model(estimated_tokens)

            # Reserve request slot BEFORE submitting (with token count)
            self.rate_limiter.reserve_request(model['name'], estimated_tokens)

            # Track assignment for logging (include token count)
            batch_assignments.append((batch_idx, model['name'], estimated_tokens))

            # Create task (will execute in parallel)
            task = self._process_single_batch(
                batch,
                model['summarizer'],
                course_id,
                file_uploader,
                chat_storage,
                canvas_user_id,
                batch_idx,
                model['name']
            )
            tasks.append(task)

        # Phase 2: Execute ALL batches in parallel
        debug_print(f"   ⚡ Submitting {len(tasks)} batches in parallel...")
        self._log_model_distribution(batch_assignments)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Phase 3: Collect and flatten results
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                debug_print(f"   ❌ Batch failed with exception: {result}")

        # Log final stats
        self._log_final_stats()

        return all_results

    async def _get_or_wait_for_model(self, estimated_tokens: int = 0) -> Dict:
        """
        Get a model with available capacity, waiting if needed

        Args:
            estimated_tokens: Estimated tokens for the batch

        Returns:
            Model dict
        """
        while True:
            model = self._get_available_model(estimated_tokens)
            if model:
                return model

            # All models at capacity - wait for next minute window
            current_time = time.time()
            minute_elapsed = current_time - self.rate_limiter.minute_start
            wait_time = 60 - minute_elapsed + 0.5

            debug_print(f"   ⏳ All models at capacity (token limits), waiting {wait_time:.1f}s for next minute window...")
            await asyncio.sleep(wait_time)

    async def _process_single_batch(
        self,
        batch: List[Dict],
        summarizer,
        course_id: str,
        file_uploader,
        chat_storage,
        canvas_user_id: Optional[str],
        batch_idx: int,
        model_name: str
    ) -> List[Dict]:
        """
        Process a single batch using the assigned model

        Args:
            batch: List of file info dicts
            summarizer: FileSummarizer instance for this model
            course_id: Canvas course ID
            file_uploader: FileUploadManager instance
            chat_storage: ChatStorage instance
            canvas_user_id: Optional Canvas user ID
            batch_idx: Batch number for logging
            model_name: Model name for logging

        Returns:
            List of result dicts
        """
        try:
            # Import here to avoid circular dependency
            import sys
            import os

            # Add backend directory to path if not already present
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)

            # Import the batch summary function
            import server
            _generate_batch_summaries = server._generate_batch_summaries

            # Use existing batch summary logic with assigned model
            batch_results = await _generate_batch_summaries(
                batch,
                course_id,
                file_uploader,
                summarizer,  # Use assigned model's summarizer
                chat_storage,
                canvas_user_id
            )

            debug_print(f"   ✅ Batch {batch_idx+1} complete ({model_name}): {len(batch_results)} files")
            return batch_results

        except Exception as e:
            debug_print(f"   ❌ Batch {batch_idx+1} failed ({model_name}): {e}")
            import traceback
            traceback.print_exc()
            # Return error results for all files in batch
            return [
                {"status": "error", "filename": f.get('filename'), "file_id": f.get('file_id'), "error": str(e)}
                for f in batch
            ]

    def _log_model_distribution(self, batch_assignments: List[Tuple[int, str, int]]):
        """Log how batches are distributed across models"""
        distribution = {}
        token_distribution = {}
        for _, model_name, tokens in batch_assignments:
            distribution[model_name] = distribution.get(model_name, 0) + 1
            token_distribution[model_name] = token_distribution.get(model_name, 0) + tokens

        debug_print(f"   📊 Model distribution:")
        for model_name, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            total_tokens = token_distribution.get(model_name, 0)
            debug_print(f"      {model_name}: {count} batches (~{total_tokens:,} tokens)")

    def _log_final_stats(self):
        """Log final rate limit usage statistics"""
        stats = self.rate_limiter.get_stats()
        debug_print(f"   📈 Rate limit usage:")
        for model_name, stat in stats.items():
            rpm_pct = (stat['rpm_used'] / stat['rpm_limit']) * 100 if stat['rpm_limit'] > 0 else 0
            rpd_pct = (stat['rpd_used'] / stat['rpd_limit']) * 100 if stat['rpd_limit'] > 0 else 0
            tpm_pct = (stat.get('tpm_used', 0) / stat.get('tpm_limit', 1)) * 100
            tpd_pct = (stat.get('tpd_used', 0) / stat.get('tpd_limit', 1)) * 100

            debug_print(f"      {model_name}:")
            debug_print(f"         RPM: {stat['rpm_used']}/{stat['rpm_limit']} ({rpm_pct:.0f}%)")
            debug_print(f"         RPD: {stat['rpd_used']}/{stat['rpd_limit']} ({rpd_pct:.0f}%)")
            debug_print(f"         TPM: {stat.get('tpm_used', 0):,}/{stat.get('tpm_limit', 0):,} ({tpm_pct:.0f}%)")
            debug_print(f"         TPD: {stat.get('tpd_used', 0):,}/{stat.get('tpd_limit', 0):,} ({tpd_pct:.0f}%)")
