import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import UploadAudioResponse, UploadImageResponse
from app.services.preprocessing import get_audio_duration, get_image_dimensions
from app.services.storage import storage_service

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/wave"}


@router.post("/upload/image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Invalid image type: {file.content_type}. Accepted: png, jpg, jpeg")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise HTTPException(400, f"Image too large: {size_mb:.1f}MB (max {settings.max_image_size_mb}MB)")

    file_id = f"img_{uuid.uuid4().hex[:8]}"
    ext = Path(file.filename).suffix or ".png"
    save_path = Path(settings.upload_dir) / f"{file_id}{ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)
    storage_key = storage_service.mirror_upload(str(save_path))

    width, height = get_image_dimensions(str(save_path))
    if width < 512 or height < 512:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "Image too small. Minimum supported size is 512x512.")
    if width > 1024 or height > 1024:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "Image too large. Maximum supported size is 1024x1024.")

    return UploadImageResponse(
        file_id=file_id,
        filename=file.filename,
        width=width,
        height=height,
        storage_key=storage_key,
    )


@router.post("/upload/audio", response_model=UploadAudioResponse)
async def upload_audio(file: UploadFile):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(400, f"Invalid audio type: {file.content_type}. Accepted: mp3, wav")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_audio_size_mb:
        raise HTTPException(400, f"Audio too large: {size_mb:.1f}MB (max {settings.max_audio_size_mb}MB)")

    file_id = f"aud_{uuid.uuid4().hex[:8]}"
    ext = Path(file.filename).suffix or ".wav"
    save_path = Path(settings.upload_dir) / f"{file_id}{ext}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    duration = get_audio_duration(str(save_path))
    if duration > settings.max_audio_duration_seconds:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            400,
            f"Audio too long: {duration:.1f}s (max {settings.max_audio_duration_seconds}s)",
        )
    storage_key = storage_service.mirror_upload(str(save_path))

    return UploadAudioResponse(
        file_id=file_id,
        filename=file.filename,
        duration_seconds=round(duration, 1),
        sample_rate=16000,
        storage_key=storage_key,
    )
