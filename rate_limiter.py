"""
Token bucket rate limiter implementation for API rate limiting.
Uses in-memory storage with automatic cleanup for expired entries.
"""
import asyncio
import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: float  # Maximum number of tokens
    refill_rate: float  # Tokens per second
    tokens: float  # Current token count
    last_refill: float  # Last refill timestamp


class RateLimiter:
    """
    Token bucket rate limiter with per-API-key tracking.
    Supports both requests per second and requests per minute.
    """
    
    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # Clean up expired buckets every 5 minutes
    
    async def start_cleanup_task(self):
        """Start background task to clean up expired buckets"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_buckets())
    
    async def _cleanup_expired_buckets(self):
        """Periodically clean up buckets that haven't been used in a while"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                current_time = time.time()
                async with self._lock:
                    # Remove buckets that haven't been used in 1 hour
                    expired_keys = [
                        key for key, bucket in self._buckets.items()
                        if current_time - bucket.last_refill > 3600
                    ]
                    for key in expired_keys:
                        del self._buckets[key]
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit buckets")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")
    
    def _refill_bucket(self, bucket: TokenBucket, current_time: float):
        """Refill bucket tokens based on elapsed time"""
        elapsed = current_time - bucket.last_refill
        if elapsed > 0:
            tokens_to_add = elapsed * bucket.refill_rate
            bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
            bucket.last_refill = current_time
    
    async def _get_or_create_bucket(
        self,
        key: str,
        requests_per_second: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        burst_size: int = 10
    ) -> TokenBucket:
        """Get or create a token bucket for the given key"""
        async with self._lock:
            if key not in self._buckets:
                # Calculate refill rate and capacity
                if requests_per_second:
                    refill_rate = float(requests_per_second)
                    capacity = float(max(requests_per_second, burst_size))
                elif requests_per_minute:
                    refill_rate = float(requests_per_minute) / 60.0
                    capacity = float(max(requests_per_minute, burst_size))
                else:
                    # Default: 60 requests per minute
                    refill_rate = 60.0 / 60.0
                    capacity = float(max(60, burst_size))
                
                self._buckets[key] = TokenBucket(
                    capacity=capacity,
                    refill_rate=refill_rate,
                    tokens=capacity,  # Start with full bucket
                    last_refill=time.time()
                )
            
            bucket = self._buckets[key]
            current_time = time.time()
            self._refill_bucket(bucket, current_time)
            return bucket
    
    async def check_rate_limit(
        self,
        key: str,
        requests_per_second: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        burst_size: int = 10
    ) -> tuple[bool, Optional[float]]:
        """
        Check if request should be allowed based on rate limit.
        
        Args:
            key: Unique identifier for rate limiting (e.g., API key)
            requests_per_second: Optional requests per second limit
            requests_per_minute: Optional requests per minute limit
            burst_size: Burst capacity
            
        Returns:
            Tuple of (allowed: bool, retry_after: Optional[float])
            - allowed: True if request is allowed, False if rate limited
            - retry_after: Seconds to wait before retry (None if allowed)
        """
        bucket = await self._get_or_create_bucket(
            key, requests_per_second, requests_per_minute, burst_size
        )
        
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, None
        else:
            # Calculate retry after time
            tokens_needed = 1.0 - bucket.tokens
            retry_after = tokens_needed / bucket.refill_rate if bucket.refill_rate > 0 else 1.0
            return False, retry_after


# Global rate limiter instance
rate_limiter = RateLimiter()

