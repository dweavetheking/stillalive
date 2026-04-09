import logging
from collections import deque

import torch
from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.job_manager import job_manager
from app.services.pipeline import pipeline

router = APIRouter()

# In-memory ring buffer for recent log entries
_recent_logs: deque[str] = deque(maxlen=200)


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            _recent_logs.append(self.format(record))
        except Exception:
            pass


def install_log_buffer():
    handler = _BufferHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


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


@router.get("/logs")
async def get_logs(n: int = 50):
    lines = list(_recent_logs)
    return {"lines": lines[-n:]}


@router.get("/debug/check")
async def debug_check():
    from pathlib import Path
    from app.config import settings

    checks = {}

    # Check directories
    for name, path in [
        ("upload_dir", settings.upload_dir),
        ("working_dir", settings.working_dir),
        ("output_dir", settings.output_dir),
        ("v3_dir", settings.echomimic_v3_dir),
        ("model_dir", settings.echomimic_model_dir),
    ]:
        p = Path(path)
        checks[name] = {
            "path": path,
            "exists": p.exists(),
            "is_dir": p.is_dir() if p.exists() else None,
            "contents": sorted(p.name for p in list(p.iterdir())[:10]) if p.is_dir() else None,
        }

    # Check src imports
    import_checks = {}
    for mod_name in [
        "src.pipeline_wan_fun_inpaint_audio_2512",
        "src.wan_transformer3d_audio_2512",
        "src.wav2vec2",
    ]:
        try:
            __import__(mod_name)
            import_checks[mod_name] = "ok"
        except Exception as e:
            import_checks[mod_name] = str(e)
    checks["imports"] = import_checks

    # Check pipeline state
    checks["pipeline_loaded"] = pipeline.loaded
    checks["pipeline_has_pipe"] = pipeline.pipe is not None

    return checks
