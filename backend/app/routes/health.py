import torch
from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.job_manager import job_manager
from app.services.pipeline import pipeline

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    gpu = None
    vram_total = None
    vram_used = None

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        vram_used = round(torch.cuda.memory_allocated(0) / 1e9, 1)

    return HealthResponse(
        status="ok",
        gpu=gpu,
        vram_total_gb=vram_total,
        vram_used_gb=vram_used,
        pipeline_loaded=pipeline.loaded,
        current_job=job_manager.current_job_id,
        queued_jobs=job_manager.queue_depth(),
    )
