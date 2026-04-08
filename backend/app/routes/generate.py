import logging
from glob import glob
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from app.config import settings
from app.models.schemas import (
    GenerateAcceptedResponse,
    GenerateRequest,
    HistoryResponse,
    JobStatusResponse,
)
from app.services.generation_queue import GenerationTask, generation_queue
from app.services.job_manager import job_manager
from app.services.pipeline import pipeline
from app.services.storage import storage_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _find_upload(file_id: str) -> str:
    """Find an uploaded file by its ID prefix."""
    pattern = str(Path(settings.upload_dir) / f"{file_id}.*")
    matches = glob(pattern)
    if not matches:
        raise HTTPException(404, f"File not found: {file_id}")
    return matches[0]


@router.post("/generate", response_model=GenerateAcceptedResponse, status_code=202)
async def generate(request: GenerateRequest):
    if not pipeline.loaded:
        raise HTTPException(503, "Pipeline not loaded yet. Please wait for startup to complete.")

    # Validate uploaded files exist
    image_path = _find_upload(request.image_file_id)
    audio_path = _find_upload(request.audio_file_id)

    job = job_manager.create_job(request.settings)
    generation_queue.enqueue(
        GenerationTask(
            job_id=job.job_id,
            request=request,
            image_path=image_path,
            audio_path=audio_path,
        )
    )

    message = "Generation started" if not job.queue_position else "Generation queued"

    return GenerateAcceptedResponse(
        job_id=job.job_id,
        status=job.status,
        queue_position=job.queue_position,
        message=message,
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(404, f"Job not found: {job_id}")
    return status


@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(404, "Job not found or not cancellable")
    return {"job_id": job_id, "status": "cancelled", "message": "Job cancelled"}


@router.get("/download/{job_id}")
async def download_video(job_id: str):
    video_path = Path(settings.output_dir) / f"{job_id}.mp4"
    if video_path.exists():
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    status = job_manager.get_status(job_id)
    storage_key = status.result.storage_key if status and status.result else None
    if storage_key:
        url = storage_service.create_presigned_url(storage_key)
        if url:
            return RedirectResponse(url=url, status_code=307)

    raise HTTPException(404, "Video not found")


@router.get("/thumbnail/{job_id}")
async def get_thumbnail(job_id: str):
    thumb_path = Path(settings.output_dir) / f"{job_id}_thumb.jpg"
    if thumb_path.exists():
        return FileResponse(
            path=str(thumb_path),
            media_type="image/jpeg",
        )

    status = job_manager.get_status(job_id)
    storage_key = status.result.thumbnail_storage_key if status and status.result else None
    if storage_key:
        url = storage_service.create_presigned_url(storage_key)
        if url:
            return RedirectResponse(url=url, status_code=307)

    raise HTTPException(404, "Thumbnail not found")


@router.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = 20):
    entries = job_manager.get_history(limit=limit)
    return HistoryResponse(runs=entries)
