from __future__ import annotations

import gc
import logging
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from app.config import settings
from app.models.schemas import GenerateRequest, GenerationResult, GenerationSettings
from app.services.job_manager import job_manager
from app.services.pose import load_pose_template
from app.services.preprocessing import preprocess_audio, preprocess_image
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

# Preset mappings
PRESETS = {
    "subtle": {"guidance_scale": 1.5, "num_inference_steps": 25},
    "standard": {"guidance_scale": 2.5, "num_inference_steps": 30},
    "expressive": {"guidance_scale": 3.5, "num_inference_steps": 35},
}

# Motion scale (1-10) maps to guidance_scale (1.0-3.5)
MOTION_SCALE_MIN = 1.0
MOTION_SCALE_MAX = 3.5


class EchoMimicPipeline:
    """Wrapper around EchoMimic V2 pipeline for FastAPI integration."""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.loaded = False

    def load(self):
        """Load all models and initialize the pipeline. Called once at startup."""
        logger.info("Loading EchoMimic V2 pipeline...")
        start = time.time()

        model_dir = settings.echomimic_model_dir

        # Import EchoMimic V2 components
        from diffusers import AutoencoderKL, DDIMScheduler
        from src.models.unet_2d_condition import UNet2DConditionModel
        from src.models.unet_3d_emo import EMOUNet3DConditionModel
        from src.models.whisper.audio2feature import load_audio_model
        from src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
        from src.models.pose_encoder import PoseEncoder

        # Load inference config
        infer_config = OmegaConf.load(
            os.path.join(model_dir, "echomimic_v2", "configs", "inference", "inference_v2.yaml")
        )

        # VAE
        vae = AutoencoderKL.from_pretrained(
            os.path.join(model_dir, "sd-vae-ft-mse")
        ).to(self.device, dtype=self.weight_dtype)

        # Reference UNet
        reference_unet = UNet2DConditionModel.from_pretrained(
            os.path.join(model_dir, "sd-image-variations-diffusers"),
            subfolder="unet",
        ).to(self.device, dtype=self.weight_dtype)
        reference_unet.load_state_dict(
            torch.load(
                os.path.join(model_dir, "echomimic_v2", "pretrained_weights", "reference_unet.pth"),
                map_location=self.device,
            )
        )

        # Denoising UNet (3D EMO)
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            os.path.join(model_dir, "sd-image-variations-diffusers"),
            os.path.join(model_dir, "echomimic_v2", "pretrained_weights", "motion_module.pth"),
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        ).to(self.device, dtype=self.weight_dtype)
        denoising_unet.load_state_dict(
            torch.load(
                os.path.join(model_dir, "echomimic_v2", "pretrained_weights", "denoising_unet.pth"),
                map_location=self.device,
            ),
            strict=False,
        )

        # Pose encoder
        pose_encoder = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
            self.device, dtype=self.weight_dtype
        )
        pose_encoder.load_state_dict(
            torch.load(
                os.path.join(model_dir, "echomimic_v2", "pretrained_weights", "pose_encoder.pth"),
                map_location=self.device,
            )
        )

        # Audio guider (whisper auto-downloads the "tiny" model)
        audio_processor = load_audio_model(
            model_path="tiny",
            device=self.device,
        )

        # Scheduler
        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        # Assemble pipeline
        self.pipe = EchoMimicV2Pipeline(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            audio_guider=audio_processor,
            pose_encoder=pose_encoder,
            scheduler=scheduler,
        )
        self.pipe = self.pipe.to(self.device, dtype=self.weight_dtype)

        self.loaded = True
        elapsed = time.time() - start
        logger.info("Pipeline loaded in %.1fs", elapsed)

    def unload(self):
        """Release GPU memory."""
        if self.pipe:
            del self.pipe
            self.pipe = None
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, job_id: str, request: GenerateRequest, image_path: str, audio_path: str):
        """Run the full generation pipeline. Called in a background thread."""
        work_dir = Path(settings.working_dir) / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            gen_settings = request.settings

            # --- Stage: preprocessing_image ---
            job_manager.update_stage(job_id, "preprocessing_image", 0.05)
            resolution = gen_settings.resolution.value
            processed_image_path = str(work_dir / "ref_image.png")
            preprocess_image(image_path, processed_image_path, resolution)
            ref_image = Image.open(processed_image_path).convert("RGB")

            if job_manager.is_cancelled(job_id):
                return

            # --- Stage: preprocessing_audio ---
            job_manager.update_stage(job_id, "preprocessing_audio", 0.18)
            processed_audio_path = str(work_dir / "audio.wav")
            duration, sample_rate = preprocess_audio(audio_path, processed_audio_path)

            if job_manager.is_cancelled(job_id):
                return

            # Calculate frame count from audio duration
            fps = 24
            num_frames = int(duration * fps)
            if num_frames < 1:
                raise ValueError("Audio too short to generate any frames")

            # --- Stage: aligning_pose ---
            job_manager.update_stage(job_id, "aligning_pose", 0.3)
            poses_tensor = load_pose_template(
                pose_dir=settings.pose_dir,
                pose_style=gen_settings.pose_style,
                num_frames=num_frames,
                width=resolution,
                height=resolution,
            )

            # Cast poses to correct dtype/device for pose encoder
            poses_tensor = poses_tensor.to(device=self.device, dtype=self.weight_dtype)

            if job_manager.is_cancelled(job_id):
                return

            # --- Resolve generation parameters ---
            if gen_settings.preset and gen_settings.preset in PRESETS:
                preset = PRESETS[gen_settings.preset]
                guidance_scale = preset["guidance_scale"]
                num_inference_steps = preset["num_inference_steps"]
            else:
                # Map motion_scale 1-10 -> guidance_scale 1.0-3.5
                t = (gen_settings.motion_scale - 1) / 9.0
                guidance_scale = MOTION_SCALE_MIN + t * (MOTION_SCALE_MAX - MOTION_SCALE_MIN)
                num_inference_steps = 30

            # Seed
            seed = gen_settings.seed
            if seed < 0:
                seed = random.randint(0, 2**31 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

            # Context window settings
            context_frames = 12
            context_overlap = 3

            # --- Stage: generating_frames ---
            job_manager.update_stage(job_id, "generating_frames", 0.4)
            logger.info(
                "Starting inference: %d frames, %dx%d, cfg=%.1f, steps=%d, seed=%d",
                num_frames, resolution, resolution, guidance_scale, num_inference_steps, seed,
            )

            with torch.no_grad():
                output = self.pipe(
                    ref_image=ref_image,
                    audio_path=processed_audio_path,
                    poses_tensor=poses_tensor,
                    width=resolution,
                    height=resolution,
                    video_length=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    audio_sample_rate=sample_rate,
                    fps=fps,
                    context_frames=context_frames,
                    context_overlap=context_overlap,
                    start_idx=0,
                )
            job_manager.update_stage(job_id, "generating_frames", 0.85)

            if job_manager.is_cancelled(job_id):
                return

            video_tensor = output.videos

            # --- Stage: assembling_video ---
            job_manager.update_stage(job_id, "assembling_video", 0.9)

            # Save silent video
            from src.utils.util import save_videos_grid

            silent_path = str(work_dir / "output_woa.mp4")
            save_videos_grid(
                videos=video_tensor,
                path=silent_path,
                n_rows=1,
                fps=fps,
                rescale=True,
            )

            # Merge audio
            output_dir = Path(settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            final_path = str(output_dir / f"{job_id}.mp4")

            from moviepy import VideoFileClip, AudioFileClip

            video_clip = VideoFileClip(silent_path)
            audio_clip = AudioFileClip(processed_audio_path).with_duration(video_clip.duration)
            final_clip = video_clip.with_audio(audio_clip)
            final_clip.write_videofile(
                final_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            video_clip.close()
            audio_clip.close()
            final_clip.close()

            job_manager.update_stage(job_id, "assembling_video", 1.0)

            # Generate thumbnail (first frame)
            thumbnail_path = str(output_dir / f"{job_id}_thumb.jpg")
            ref_image.save(thumbnail_path, "JPEG", quality=80)

            video_storage_key, thumb_storage_key = storage_service.mirror_outputs(
                job_id=job_id,
                video_path=final_path,
                thumb_path=thumbnail_path,
            )

            # --- Complete ---
            result = GenerationResult(
                video_filename=f"{job_id}.mp4",
                duration_seconds=duration,
                resolution=f"{resolution}x{resolution}",
                seed_used=seed,
                frames_generated=num_frames,
                storage_key=video_storage_key,
                thumbnail_storage_key=thumb_storage_key,
            )
            job_manager.complete_job(job_id, result)
            logger.info("Generation complete: %s", job_id)

        except Exception as e:
            logger.exception("Generation failed: %s", job_id)
            job_manager.fail_job(job_id, str(e))

        finally:
            # Cleanup working directory
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Singleton
pipeline = EchoMimicPipeline()
