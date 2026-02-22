"""
ARIA Server Configuration - Redis Bridge Settings
"""
import os
from typing import Dict, List

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "192.168.1.190")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Queue Configuration
QUEUE_PREFIX = "gpu:queue:"
RESULT_PREFIX = "gpu:result:"
PROCESSING_PREFIX = "gpu:processing:"

# Supported model queues
MODEL_QUEUES = {
    "tts": {
        "orpheus-3b": f"{QUEUE_PREFIX}tts:orpheus-3b",
        "orpheus-3b-q4": f"{QUEUE_PREFIX}tts:orpheus-3b-q4",
    },
    "music": {
        "musicgen-small": f"{QUEUE_PREFIX}music:musicgen-small",
        "musicgen-medium": f"{QUEUE_PREFIX}music:musicgen-medium",
    },
    "llm": {
        "llama-3b": f"{QUEUE_PREFIX}llm:llama-3b",
        "mistral-7b": f"{QUEUE_PREFIX}llm:mistral-7b",
    }
}

# Timeout Configuration
QUEUE_TIMEOUT = 2  # seconds for BRPOP timeout
PROCESSING_TIMEOUT = 300  # seconds TTL for processing keys
RESULT_TTL = 3600  # seconds TTL for result keys

# Retry Configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds initial backoff

# Required task fields
REQUIRED_TASK_FIELDS = [
    "job_id", "client_id", "model_type", "model_id", 
    "queued_at", "timeout_seconds", "callback_key", "payload"
]