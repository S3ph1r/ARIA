from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class AriaFileRef(BaseModel):
    ref_id: str
    local_path: str
    size_bytes: Optional[int] = None

class AriaFileRefs(BaseModel):
    input: Optional[List[AriaFileRef]] = None
    output: Optional[List[AriaFileRef]] = None

class AriaTaskPayload(BaseModel):
    """
    Standard DIAS-compatible task payload expected from the Redis queue.
    """
    job_id: str
    client_id: str
    model_type: str = Field(description="e.g., tts, music, llm, image, stt, cloud")
    provider: str = Field(default="local", description="e.g., google, openai, local")
    model_id: str = Field(description="e.g., gemini-1.5-flash, orpheus-3b, fish-s1-mini")
    queued_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    priority: int = Field(default=1, description="Higher is more urgent")
    timeout_seconds: int = Field(default=1800)
    callback_key: str = Field(description="Redis key to LPUSH the result to")
    file_refs: Optional[AriaFileRefs] = None
    payload: Dict[str, Any] = Field(description="Backend-specific parameters")
    schema_version: str = Field(default="1.0")

    @property
    def queue_key(self) -> str:
        """Helper to get the standard Redis queue key for this task."""
        # New Standard: aria:q:{env}:{prov}:{model}:{client}
        return f"aria:q:{self.model_type}:{self.provider}:{self.model_id}:{self.client_id}"


class AriaTaskResult(BaseModel):
    """
    Standard result payload sent back to Redis when a task is completed/failed.
    """
    job_id: str
    client_id: str
    model_type: str
    provider: str = Field(default="local")
    model_id: str
    status: str = Field(description="'done', 'error', 'timeout', 'cancelled'")
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_seconds: float
    output: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[str] = Field(default=None)
    error_code: Optional[str] = Field(default=None, description="e.g., OOM, 429, INFERENCE_FAILED")
    retry_count: int = Field(default=0)
    schema_version: str = Field(default="1.0")
