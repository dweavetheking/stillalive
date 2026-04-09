from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models.schemas import (
    GenerationResult,
    GenerationSettings,
    HistoryEntry,
    JobStatus,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)

STAGE_LABELS = {
    "queued": "Waiting to start...",
    "preprocessing_image": "Preparing your portrait...",
    "preprocessing_audio": "Processing your audio...",
    "encoding_audio": "Encoding audio features...",
    "loading_pipeline": "Warming up the AI engine...",
    "generating_frames": "Generating frames...",
    "decoding_frames": "Decoding VAE output...",
    "assembling_video": "Assembling your video...",
    "done": "Done!",
    "failed": "Generation failed",
    "cancelled": "Cancelled",
}


class JobManager:
    def __init__(self):
        self._current_job: JobStatusResponse | None = None
        self._pending_jobs: list[JobStatusResponse] = []
        self._history: list[HistoryEntry] = []
        self._lock = threading.Lock()
        self._load_history()

    @property
    def is_busy(self) -> bool:
        return (
            self._current_job is not None
            and self._current_job.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
        )

    @property
    def current_job_id(self) -> str | None:
        if self._current_job and self.is_busy:
            return self._current_job.job_id
        return None

    def create_job(self, gen_settings: GenerationSettings) -> JobStatusResponse:
        with self._lock:
            now = datetime.now(timezone.utc)
            job = JobStatusResponse(
                job_id=f"job_{uuid.uuid4().hex[:10]}",
                status=JobStatus.QUEUED,
                stage="queued",
                stage_label=STAGE_LABELS["queued"],
                progress=0.0,
                elapsed_seconds=0.0,
                created_at=now,
                settings=gen_settings,
            )
            self._pending_jobs.append(job)
            self._recompute_queue_positions_locked()
            return job.model_copy()

    def start_job(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            if self._current_job and self._current_job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                return None
            for idx, pending in enumerate(self._pending_jobs):
                if pending.job_id != job_id:
                    continue
                job = self._pending_jobs.pop(idx)
                if job.status == JobStatus.CANCELLED:
                    return None
                job.status = JobStatus.PROCESSING
                job.stage = "loading_pipeline"
                job.stage_label = STAGE_LABELS["loading_pipeline"]
                job.queue_position = None
                self._current_job = job
                self._recompute_queue_positions_locked()
                return job.model_copy()
            return None

    def update_stage(self, job_id: str, stage: str, progress: float = 0.0, label: str | None = None):
        with self._lock:
            if not self._current_job or self._current_job.job_id != job_id:
                return
            if self._current_job.status == JobStatus.CANCELLED:
                return

            self._current_job.status = JobStatus.PROCESSING
            self._current_job.stage = stage
            self._current_job.stage_label = label or STAGE_LABELS.get(stage, stage)
            self._current_job.progress = min(max(progress, 0.0), 1.0)
            elapsed = (datetime.now(timezone.utc) - self._current_job.created_at).total_seconds()
            self._current_job.elapsed_seconds = elapsed

            if self._current_job.progress > 0.01 and self._current_job.progress < 1.0:
                total_est = elapsed / self._current_job.progress
                remaining = max(0.0, total_est - elapsed)
                self._current_job.estimated_remaining_seconds = remaining
            else:
                self._current_job.estimated_remaining_seconds = None

    def complete_job(self, job_id: str, result: GenerationResult):
        with self._lock:
            if not self._current_job or self._current_job.job_id != job_id:
                return

            if self._current_job.status == JobStatus.CANCELLED:
                self._archive_current_job_locked()
                return

            now = datetime.now(timezone.utc)
            self._current_job.status = JobStatus.COMPLETED
            self._current_job.stage = "done"
            self._current_job.stage_label = STAGE_LABELS["done"]
            self._current_job.progress = 1.0
            self._current_job.completed_at = now
            self._current_job.result = result
            elapsed = (now - self._current_job.created_at).total_seconds()
            self._current_job.elapsed_seconds = elapsed
            self._current_job.estimated_remaining_seconds = 0.0
            self._archive_current_job_locked()

    def fail_job(self, job_id: str, error: str):
        with self._lock:
            if not self._current_job or self._current_job.job_id != job_id:
                return

            if self._current_job.status == JobStatus.CANCELLED:
                self._archive_current_job_locked()
                return

            now = datetime.now(timezone.utc)
            self._current_job.status = JobStatus.FAILED
            self._current_job.stage = "failed"
            self._current_job.stage_label = STAGE_LABELS["failed"]
            self._current_job.completed_at = now
            self._current_job.error = error
            elapsed = (now - self._current_job.created_at).total_seconds()
            self._current_job.elapsed_seconds = elapsed
            self._archive_current_job_locked()

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            now = datetime.now(timezone.utc)

            if self._current_job and self._current_job.job_id == job_id:
                if self._current_job.status not in (JobStatus.QUEUED, JobStatus.PROCESSING):
                    return False
                self._current_job.status = JobStatus.CANCELLED
                self._current_job.stage = "cancelled"
                self._current_job.stage_label = STAGE_LABELS["cancelled"]
                self._current_job.completed_at = now
                return True

            for idx, job in enumerate(self._pending_jobs):
                if job.job_id != job_id:
                    continue

                job.status = JobStatus.CANCELLED
                job.stage = "cancelled"
                job.stage_label = STAGE_LABELS["cancelled"]
                job.completed_at = now
                job.elapsed_seconds = (now - job.created_at).total_seconds()

                entry = HistoryEntry(
                    job_id=job.job_id,
                    status=job.status,
                    created_at=job.created_at,
                    completed_at=job.completed_at,
                    settings=job.settings,
                    result=job.result,
                    elapsed_seconds=job.elapsed_seconds,
                )
                self._history.append(entry)
                del self._pending_jobs[idx]
                if len(self._history) > settings.max_history_entries:
                    self._history = self._history[-settings.max_history_entries :]
                self._recompute_queue_positions_locked()
                self._save_history()
                return True
            return False

    def finalize_cancelled_job(self, job_id: str):
        with self._lock:
            if not self._current_job or self._current_job.job_id != job_id:
                return
            if self._current_job.status != JobStatus.CANCELLED:
                return

            if not self._current_job.completed_at:
                self._current_job.completed_at = datetime.now(timezone.utc)
            self._current_job.elapsed_seconds = (
                self._current_job.completed_at - self._current_job.created_at
            ).total_seconds()
            self._archive_current_job_locked()

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                return self._current_job.status == JobStatus.CANCELLED
            for job in self._pending_jobs:
                if job.job_id == job_id:
                    return job.status == JobStatus.CANCELLED
        return False

    def get_status(self, job_id: str) -> JobStatusResponse | None:
        with self._lock:
            if self._current_job and self._current_job.job_id == job_id:
                return self._current_job.model_copy()

            for job in self._pending_jobs:
                if job.job_id == job_id:
                    copy = job.model_copy()
                    copy.elapsed_seconds = (
                        datetime.now(timezone.utc) - copy.created_at
                    ).total_seconds()
                    return copy

            for entry in self._history:
                if entry.job_id == job_id:
                    stage = entry.last_stage or "done"
                    if entry.status == JobStatus.FAILED and not entry.last_stage:
                        stage = "failed"
                    elif entry.status == JobStatus.CANCELLED and not entry.last_stage:
                        stage = "cancelled"

                    return JobStatusResponse(
                        job_id=entry.job_id,
                        status=entry.status,
                        stage=stage,
                        stage_label=STAGE_LABELS.get(stage, stage),
                        progress=1.0 if entry.status == JobStatus.COMPLETED else entry.last_progress,
                        elapsed_seconds=entry.elapsed_seconds,
                        created_at=entry.created_at,
                        completed_at=entry.completed_at,
                        settings=entry.settings,
                        result=entry.result,
                        error=entry.error,
                    )
        return None

    def get_history(self, limit: int = 20) -> list[HistoryEntry]:
        with self._lock:
            return list(reversed(self._history[-limit:]))

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._pending_jobs)

    def _recompute_queue_positions_locked(self):
        offset = 1 if self._current_job else 0
        for idx, job in enumerate(self._pending_jobs):
            position = idx + offset
            job.queue_position = position if position > 0 else None

    def _archive_current_job_locked(self):
        if not self._current_job:
            return

        job = self._current_job
        entry = HistoryEntry(
            job_id=job.job_id,
            status=job.status,
            created_at=job.created_at,
            completed_at=job.completed_at,
            settings=job.settings,
            result=job.result,
            elapsed_seconds=job.elapsed_seconds,
            error=job.error,
            last_stage=job.stage,
            last_progress=job.progress,
        )
        self._history.append(entry)
        if len(self._history) > settings.max_history_entries:
            self._history = self._history[-settings.max_history_entries :]
        self._current_job = None
        self._save_history()

    def _save_history(self):
        try:
            path = Path(settings.history_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [entry.model_dump(mode="json") for entry in self._history]
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:
            logger.exception("Failed to save history")

    def _load_history(self):
        try:
            path = Path(settings.history_file)
            if path.exists():
                data = json.loads(path.read_text())
                self._history = [HistoryEntry.model_validate(entry) for entry in data]
                logger.info("Loaded %d history entries", len(self._history))
        except Exception:
            logger.exception("Failed to load history, starting fresh")
            self._history = []


# Singleton
job_manager = JobManager()
