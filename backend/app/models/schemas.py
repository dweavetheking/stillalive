from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Resolution(int, Enum):
    LOW = 512
    HIGH = 768


# --- Requests ---

class GenerationSettings(BaseModel):
    motion_scale: int = Field(default=5, ge=1, le=10)
    seed: int = Field(default=-1, description="-1 for random")
    resolution: Resolution = Resolution.HIGH
    pose_style: str = Field(default="01")
    preset: Optional[str] = Field(default=None, description="subtle, standard, or expressive")


class GenerateRequest(BaseModel):
    image_file_id: str
    audio_file_id: str
    settings: GenerationSettings = GenerationSettings()


# --- Responses ---

class UploadImageResponse(BaseModel):
    file_id: str
    filename: str
    width: int
    height: int
    storage_key: Optional[str] = None


class UploadAudioResponse(BaseModel):
    file_id: str
    filename: str
    duration_seconds: float
    sample_rate: int
    storage_key: Optional[str] = None


class GenerationResult(BaseModel):
    video_filename: str
    duration_seconds: float
    resolution: str
    seed_used: int
    frames_generated: int
    storage_key: Optional[str] = None
    thumbnail_storage_key: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: str = "queued"
    stage_label: str = "Waiting to start..."
    progress: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    settings: GenerationSettings = GenerationSettings()
    queue_position: Optional[int] = None
    result: Optional[GenerationResult] = None
    error: Optional[str] = None


class GenerateAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: Optional[int] = None
    message: str


class BusyResponse(BaseModel):
    error: str = "busy"
    message: str = "A generation is already in progress"
    current_job_id: str


class HistoryEntry(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    settings: GenerationSettings
    result: Optional[GenerationResult] = None
    thumbnail_filename: Optional[str] = None
    elapsed_seconds: float = 0.0


class HistoryResponse(BaseModel):
    runs: list[HistoryEntry]


class HealthResponse(BaseModel):
    status: str = "ok"
    gpu: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    pipeline_loaded: bool = False
    current_job: Optional[str] = None
    queued_jobs: int = 0


class ErrorResponse(BaseModel):
    error: str
    message: str
