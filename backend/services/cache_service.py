"""
Cache Service

Redis-based caching service for improved performance and reduced database load.

Author: AI-EEG Learning Platform Team
"""

import asyncio
import json
import pickle
from typing import Any, Dict, List, Optional

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from utils.config import settings
from utils.logging_config import get_request_logger

logger = get_request_logger("cache_service")


class CacheService:
    """
    Redis-based caching service with fallback to in-memory cache.
    """

    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.memory_cache_ttl = {}
        self.initialized = False

    async def initialize(self) -> None:
        """
        Initialize the cache service.
        """
        if self.initialized:
            return

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=False,  # Keep bytes for pickle
                    retry_on_timeout=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )

                # Test connection
                await self.redis_client.ping()
                logger.info("Redis cache initialized successfully")

            except Exception as e:
                logger.warning(
                    "Redis connection failed, falling back to memory cache",
                    error=str(e),
                )
                self.redis_client = None
        else:
            logger.warning("Redis not available, using memory cache")

        self.initialized = True

    async def close(self) -> None:
        """
        Close the cache service connections.
        """
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

        self.memory_cache.clear()
        self.memory_cache_ttl.clear()
        self.initialized = False

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.error("Redis get failed", key=key, error=str(e))

        # Fallback to memory cache
        if key in self.memory_cache:
            # Check TTL
            if key in self.memory_cache_ttl:
                ttl = self.memory_cache_ttl[key]
                if asyncio.get_event_loop().time() > ttl:
                    # Expired
                    del self.memory_cache[key]
                    del self.memory_cache_ttl[key]
                    return None

            return self.memory_cache[key]

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        await self.initialize()

        if ttl is None:
            ttl = settings.REDIS_CACHE_TTL

        try:
            pickled_value = pickle.dumps(value)
        except Exception as e:
            logger.error("Failed to pickle value", key=key, error=str(e))
            return False

        # Try Redis first
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, pickled_value)
                return True
            except Exception as e:
                logger.error("Redis set failed", key=key, error=str(e))

        # Fallback to memory cache
        try:
            self.memory_cache[key] = value
            if ttl:
                self.memory_cache_ttl[key] = asyncio.get_event_loop().time() + ttl
            return True
        except Exception as e:
            logger.error("Memory cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        await self.initialize()

        success = False

        # Try Redis first
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
                success = True
            except Exception as e:
                logger.error("Redis delete failed", key=key, error=str(e))

        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self.memory_cache_ttl:
                del self.memory_cache_ttl[key]
            success = True

        return success

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                return await self.redis_client.exists(key) > 0
            except Exception as e:
                logger.error("Redis exists failed", key=key, error=str(e))

        # Check memory cache
        return key in self.memory_cache

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                return await self.redis_client.expire(key, ttl) > 0
            except Exception as e:
                logger.error("Redis expire failed", key=key, error=str(e))

        # Update memory cache TTL
        if key in self.memory_cache:
            self.memory_cache_ttl[key] = asyncio.get_event_loop().time() + ttl
            return True

        return False

    async def get_ttl(self, key: str) -> int:
        """
        Get time to live for key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                ttl = await self.redis_client.ttl(key)
                return ttl
            except Exception as e:
                logger.error("Redis TTL failed", key=key, error=str(e))

        # Check memory cache
        if key in self.memory_cache_ttl:
            ttl = self.memory_cache_ttl[key] - asyncio.get_event_loop().time()
            return max(int(ttl), -1)

        if key in self.memory_cache:
            return -1  # No TTL

        return -2  # Key doesn't exist

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment integer value in cache.

        Args:
            key: Cache key
            amount: Amount to increment

        Returns:
            New value or None if failed
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                return await self.redis_client.incrby(key, amount)
            except Exception as e:
                logger.error("Redis incr failed", key=key, error=str(e))

        # Memory cache fallback
        current_value = self.memory_cache.get(key, 0)
        if not isinstance(current_value, int):
            return None

        new_value = current_value + amount
        self.memory_cache[key] = new_value
        return new_value

    async def hget(self, key: str, field: str) -> Optional[Any]:
        """
        Get field from hash in cache.

        Args:
            key: Hash key
            field: Field name

        Returns:
            Field value or None
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                value = await self.redis_client.hget(key, field)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.error("Redis hget failed", key=key, field=field, error=str(e))

        # Memory cache fallback - use nested dict
        hash_data = self.memory_cache.get(key, {})
        if isinstance(hash_data, dict):
            return hash_data.get(field)

        return None

    async def hset(
        self, key: str, field: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """
        Set field in hash cache.

        Args:
            key: Hash key
            field: Field name
            value: Field value
            ttl: Time to live for the hash

        Returns:
            True if successful
        """
        await self.initialize()

        try:
            pickled_value = pickle.dumps(value)
        except Exception as e:
            logger.error("Failed to pickle value", key=key, field=field, error=str(e))
            return False

        # Try Redis first
        if self.redis_client:
            try:
                await self.redis_client.hset(key, field, pickled_value)
                if ttl:
                    await self.redis_client.expire(key, ttl)
                return True
            except Exception as e:
                logger.error("Redis hset failed", key=key, field=field, error=str(e))

        # Memory cache fallback
        if key not in self.memory_cache:
            self.memory_cache[key] = {}

        hash_data = self.memory_cache[key]
        if isinstance(hash_data, dict):
            hash_data[field] = value
            if ttl:
                self.memory_cache_ttl[key] = asyncio.get_event_loop().time() + ttl
            return True

        return False

    async def hgetall(self, key: str) -> Dict[str, Any]:
        """
        Get all fields from hash in cache.

        Args:
            key: Hash key

        Returns:
            Dictionary of all fields
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                hash_data = await self.redis_client.hgetall(key)
                result = {}
                for field, value in hash_data.items():
                    result[field] = pickle.loads(value)
                return result
            except Exception as e:
                logger.error("Redis hgetall failed", key=key, error=str(e))

        # Memory cache fallback
        hash_data = self.memory_cache.get(key, {})
        if isinstance(hash_data, dict):
            return hash_data.copy()

        return {}

    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.

        Args:
            pattern: Key pattern

        Returns:
            List of matching keys
        """
        await self.initialize()

        # Try Redis first
        if self.redis_client:
            try:
                return [key.decode() for key in await self.redis_client.keys(pattern)]
            except Exception as e:
                logger.error("Redis keys failed", pattern=pattern, error=str(e))

        # Memory cache fallback
        import fnmatch

        return [
            key for key in self.memory_cache.keys() if fnmatch.fnmatch(key, pattern)
        ]

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Key pattern

        Returns:
            Number of keys deleted
        """
        await self.initialize()

        keys_to_delete = await self.keys(pattern)
        deleted_count = 0

        for key in keys_to_delete:
            if await self.delete(key):
                deleted_count += 1

        return deleted_count

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics
        """
        await self.initialize()

        stats = {
            "cache_type": "redis" if self.redis_client else "memory",
            "initialized": self.initialized,
        }

        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update(
                    {
                        "redis_connected_clients": info.get("connected_clients", 0),
                        "redis_used_memory": info.get("used_memory", 0),
                        "redis_total_keys": await self.redis_client.dbsize(),
                    }
                )
            except Exception as e:
                stats["redis_error"] = str(e)
        else:
            stats.update(
                {
                    "memory_cache_size": len(self.memory_cache),
                    "memory_cache_with_ttl": len(self.memory_cache_ttl),
                }
            )

        return stats

    # EEG-specific cache methods
    async def cache_eeg_features(
        self,
        user_id: str,
        session_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache EEG features for a user session.

        Args:
            user_id: User identifier
            session_id: Session identifier
            features: EEG features to cache
            ttl: Time to live in seconds
        """
        cache_key = f"eeg_features:{user_id}:{session_id}"
        await self.set(cache_key, features, ttl)

    async def get_cached_eeg_features(
        self, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached EEG features for a user session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Cached EEG features or None
        """
        cache_key = f"eeg_features:{user_id}:{session_id}"
        return await self.get(cache_key)

    async def cache_user_profile(
        self, user_id: str, profile_data: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """
        Cache user profile data.

        Args:
            user_id: User identifier
            profile_data: Profile data to cache
            ttl: Time to live in seconds
        """
        cache_key = f"user_profile:{user_id}"
        await self.set(cache_key, profile_data, ttl)

    async def get_cached_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached user profile data.

        Args:
            user_id: User identifier

        Returns:
            Cached profile data or None
        """
        cache_key = f"user_profile:{user_id}"
        return await self.get(cache_key)
