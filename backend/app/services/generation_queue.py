from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from app.models.schemas import GenerateRequest
from app.services.job_manager import JobStatus, job_manager
from app.services.pipeline import pipeline

logger = logging.getLogger(__name__)


@dataclass
class GenerationTask:
    job_id: str
    request: GenerateRequest
    image_path: str
    audio_path: str


class GenerationQueue:
    def __init__(self):
        self._q: queue.Queue[str | None] = queue.Queue()
        self._tasks: dict[str, GenerationTask] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="generation-worker")
        self._thread.start()
        logger.info("Generation queue worker started")

    def stop(self):
        if not self._thread:
            return
        self._running.clear()
        self._q.put(None)
        self._thread.join(timeout=5)
        logger.info("Generation queue worker stopped")

    def enqueue(self, task: GenerationTask):
        with self._lock:
            self._tasks[task.job_id] = task
        self._q.put(task.job_id)

    def _pop_task(self, job_id: str) -> GenerationTask | None:
        with self._lock:
            return self._tasks.pop(job_id, None)

    def _run(self):
        while self._running.is_set():
            job_id = self._q.get()
            if job_id is None:
                break

            task = self._pop_task(job_id)
            if not task:
                continue

            status = job_manager.get_status(job_id)
            if not status or status.status == JobStatus.CANCELLED:
                continue

            active_job = job_manager.start_job(job_id)
            if not active_job:
                continue

            try:
                pipeline.generate(
                    job_id=task.job_id,
                    request=task.request,
                    image_path=task.image_path,
                    audio_path=task.audio_path,
                )
            except Exception:
                logger.exception("Unhandled error in generation worker for job %s", job_id)
                job_manager.fail_job(job_id, "Unhandled worker error")
            finally:
                job_manager.finalize_cancelled_job(job_id)


# Singleton
generation_queue = GenerationQueue()
