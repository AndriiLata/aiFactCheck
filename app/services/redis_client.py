from __future__ import annotations

import os, redis

def get_redis() -> redis.Redis:
    """
    Return a singleton Redis connection.
    REDIS_URL defaults to localhost.
    """
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # decode_responses=True → str in/out instead of bytes
    return redis.Redis.from_url(url, decode_responses=True)

# global, shared across imports
rdb = get_redis()
