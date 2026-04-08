from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self):
        self._client = None
        self._enabled = settings.storage_backend.lower() == "s3"

        if not self._enabled:
            return

        try:
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=settings.storage_region,
                endpoint_url=settings.storage_endpoint_url,
                aws_access_key_id=settings.storage_access_key_id,
                aws_secret_access_key=settings.storage_secret_access_key,
            )
            logger.info("Object storage enabled (backend=s3)")
        except Exception:
            self._enabled = False
            logger.exception("Failed to initialize S3 client, falling back to local storage")

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None and bool(settings.storage_bucket)

    def _key_for_upload(self, file_path: str) -> str:
        p = Path(file_path)
        return f"{settings.storage_prefix}/uploads/{p.name}"

    def _key_for_video(self, job_id: str) -> str:
        return f"{settings.storage_prefix}/outputs/{job_id}.mp4"

    def _key_for_thumb(self, job_id: str) -> str:
        return f"{settings.storage_prefix}/outputs/{job_id}_thumb.jpg"

    def mirror_upload(self, file_path: str) -> str | None:
        if not self.enabled:
            return None

        key = self._key_for_upload(file_path)
        try:
            self._client.upload_file(file_path, settings.storage_bucket, key)
            return key
        except Exception:
            logger.exception("Failed to mirror upload to object storage: %s", file_path)
            return None

    def mirror_outputs(self, job_id: str, video_path: str, thumb_path: str) -> tuple[str | None, str | None]:
        if not self.enabled:
            return (None, None)

        video_key = self._key_for_video(job_id)
        thumb_key = self._key_for_thumb(job_id)

        try:
            self._client.upload_file(video_path, settings.storage_bucket, video_key)
        except Exception:
            logger.exception("Failed to mirror video output to object storage: %s", video_path)
            video_key = None

        try:
            self._client.upload_file(thumb_path, settings.storage_bucket, thumb_key)
        except Exception:
            logger.exception("Failed to mirror thumbnail output to object storage: %s", thumb_path)
            thumb_key = None

        return (video_key, thumb_key)

    def create_presigned_url(self, key: str) -> str | None:
        if not self.enabled:
            return None
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.storage_bucket, "Key": key},
                ExpiresIn=settings.storage_presign_expiry_seconds,
            )
        except Exception:
            logger.exception("Failed to create presigned URL for key: %s", key)
            return None


storage_service = StorageService()
