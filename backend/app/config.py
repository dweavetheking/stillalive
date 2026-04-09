from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model paths
    echomimic_model_dir: str = "/workspace/models"
    pose_dir: str = "/workspace/models/echomimic_v2/assets/halfbody_demo/pose"

    # Temp storage
    temp_dir: str = "/tmp/stillalive"
    upload_dir: str = "/tmp/stillalive/uploads"
    working_dir: str = "/tmp/stillalive/working"
    output_dir: str = "/tmp/stillalive/outputs"
    history_file: str = "/tmp/stillalive/history.json"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Limits
    max_image_size_mb: int = 10
    max_audio_size_mb: int = 20
    max_audio_duration_seconds: int = 180
    max_history_entries: int = 50

    # Cleanup
    cleanup_after_hours: int = 24

    # Storage backend
    storage_backend: str = "local"  # local or s3 (S3-compatible, includes R2)
    storage_bucket: str = ""
    storage_region: str = "auto"
    storage_endpoint_url: str | None = None
    storage_access_key_id: str | None = None
    storage_secret_access_key: str | None = None
    storage_prefix: str = "stillalive"
    storage_presign_expiry_seconds: int = 3600

    model_config = {"env_prefix": "", "env_file": ".env"}

    def ensure_dirs(self):
        for d in [self.temp_dir, self.upload_dir, self.working_dir, self.output_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
