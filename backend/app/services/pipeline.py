from __future__ import annotations

import gc
import logging
import os
import random
import shutil
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from app.config import settings
from app.models.schemas import GenerateRequest, GenerationResult, GenerationSettings
from app.services.job_manager import job_manager
from app.services.preprocessing import preprocess_audio, preprocess_image
from app.services.storage import storage_service

logger = logging.getLogger(__name__)

# Motion scale (1-10) maps to audio_guidance_scale (1.5-4.0)
AUDIO_GUIDANCE_MIN = 1.5
AUDIO_GUIDANCE_MAX = 4.0

FPS = 25  # V3 uses 25fps


def loudness_norm(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Normalize audio loudness to target LUFS."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if abs(loudness) > 100:
            return audio
        return pyln.normalize.loudness(audio, loudness, target_lufs)
    except Exception:
        return audio


class EchoMimicPipeline:
    """Wrapper around EchoMimic V3 Flash pipeline for FastAPI integration."""

    def __init__(self):
        self.pipe = None
        self.audio_encoder = None
        self.audio_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.loaded = False

    def load(self):
        """Load all models and initialize the V3 Flash pipeline."""
        logger.info("Loading EchoMimic V3 Flash pipeline...")
        start = time.time()

        model_dir = settings.echomimic_model_dir
        wan_base = os.path.join(model_dir, "Wan2.1-Fun-V1.1-1.3B-InP")
        v3_weights = os.path.join(model_dir, "EchoMimicV3")
        wav2vec_dir = os.path.join(model_dir, "chinese-wav2vec2-base")

        # --- Import V3 components (from echomimic_v3 repo on PYTHONPATH) ---
        from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from transformers import AutoFeatureExtractor, AutoTokenizer
        from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
        from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
        from src.wan_vae import AutoencoderKLWan
        from src.wan_text_encoder import WanT5EncoderModel
        from src.wan_image_encoder import CLIPModel
        from src.wav2vec2 import Wav2Vec2Model

        # --- Load config ---
        import json
        config_path = os.path.join(wan_base, "config.json")
        with open(config_path) as f:
            wan_config = json.load(f)
        vae_kwargs = wan_config.get("vae_kwargs", {})

        # --- VAE ---
        logger.info("Loading VAE...")
        vae_path = os.path.join(wan_base, "Wan2.1_VAE.pth")
        vae = AutoencoderKLWan.from_pretrained(
            vae_path, additional_kwargs=vae_kwargs
        ).to(self.device, dtype=self.weight_dtype)
        self.vae_temporal_ratio = getattr(vae.config, "temporal_compression_ratio", 4)

        # --- Text encoder ---
        logger.info("Loading text encoder...")
        t5_path = os.path.join(wan_base, "models_t5_umt5-xxl-enc-bf16.pth")
        text_encoder = WanT5EncoderModel.from_pretrained(
            t5_path, torch_dtype=self.weight_dtype
        ).to(self.device).eval()

        # --- Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(wan_base, "google", "umt5-xxl")
        )

        # --- CLIP image encoder ---
        logger.info("Loading CLIP image encoder...")
        clip_path = os.path.join(wan_base, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        clip_image_encoder = CLIPModel.from_pretrained(
            clip_path
        ).to(self.device, dtype=self.weight_dtype).eval()

        # --- Transformer (V3 Flash weights) ---
        logger.info("Loading V3 Flash transformer...")
        transformer = WanTransformerAudioMask3DModel.from_pretrained(
            v3_weights,
            subfolder="echomimicv3-flash-pro",
            torch_dtype=self.weight_dtype,
        ).to(self.device)

        # --- Scheduler ---
        scheduler = FlowUniPCMultistepScheduler(
            shift=5.0,
            use_dynamic_shifting=False,
        )

        # --- Assemble pipeline ---
        self.pipe = WanFunInpaintAudioPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )

        # --- Audio encoder (Wav2Vec2 Flash variant) ---
        logger.info("Loading Wav2Vec2 audio encoder...")
        self.audio_processor = AutoFeatureExtractor.from_pretrained(wav2vec_dir)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir).to(self.device).eval()

        self.loaded = True
        elapsed = time.time() - start
        logger.info("Pipeline loaded in %.1fs", elapsed)

    def unload(self):
        """Release GPU memory."""
        for attr in ("pipe", "audio_encoder", "audio_processor"):
            if getattr(self, attr, None):
                delattr(self, attr)
                setattr(self, attr, None)
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode_audio(self, audio_path: str, video_length: int) -> torch.Tensor:
        """Extract audio embeddings using Wav2Vec2 with sliding window."""
        mel_input, sr = librosa.load(audio_path, sr=16000)
        mel_input = loudness_norm(mel_input, sr)

        audio_feature = np.squeeze(
            self.audio_processor(mel_input, sampling_rate=16000).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, seq_len=video_length, output_hidden_states=True
            )

        # Stack hidden states from all layers (skip first/input layer)
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        audio_emb = audio_emb.cpu().detach()

        # Sliding window: 5-frame context (±2) for each temporal position
        indices = (torch.arange(5) - 2) * 1
        center_indices = torch.arange(0, video_length, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_emb.shape[0] - 1)
        audio_embeds = audio_emb[center_indices]  # [F, 5, 12, 768]
        audio_embeds = audio_embeds.unsqueeze(0).to(device=self.device)  # [1, F, 5, 12, 768]

        return audio_embeds

    def generate(self, job_id: str, request: GenerateRequest, image_path: str, audio_path: str):
        """Run the V3 Flash generation pipeline."""
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
            job_manager.update_stage(job_id, "preprocessing_audio", 0.10)
            processed_audio_path = str(work_dir / "audio.wav")
            duration, sample_rate = preprocess_audio(audio_path, processed_audio_path)

            if job_manager.is_cancelled(job_id):
                return

            # Calculate frame count (aligned to VAE temporal compression)
            raw_frames = int(duration * FPS)
            if raw_frames < 1:
                raise ValueError("Audio too short to generate any frames")
            video_length = int((raw_frames - 1) // self.vae_temporal_ratio * self.vae_temporal_ratio) + 1

            # --- Stage: encoding_audio ---
            job_manager.update_stage(job_id, "encoding_audio", 0.15,
                                     label="Encoding audio features...")
            audio_embeds = self._encode_audio(processed_audio_path, video_length)

            if job_manager.is_cancelled(job_id):
                return

            # --- Prepare video input (first-frame inpainting) ---
            job_manager.update_stage(job_id, "generating_frames", 0.20,
                                     label="Preparing generation...")
            from src.utils import get_image_to_video_latent2
            input_video, input_video_mask, clip_image = get_image_to_video_latent2(
                ref_image, None,
                video_length=video_length,
                sample_size=[resolution, resolution],
            )

            # --- Resolve parameters ---
            # Map motion_scale 1-10 -> audio_guidance_scale
            t = (gen_settings.motion_scale - 1) / 9.0
            audio_guidance_scale = AUDIO_GUIDANCE_MIN + t * (AUDIO_GUIDANCE_MAX - AUDIO_GUIDANCE_MIN)

            num_inference_steps = 8
            guidance_scale = 6.0

            seed = gen_settings.seed
            if seed < 0:
                seed = random.randint(0, 2**31 - 1)
            generator = torch.Generator(device=self.device).manual_seed(seed)

            prompt = gen_settings.prompt
            negative_prompt = "Distorted, blurry, low quality, unnatural, glitch"

            # --- Stage: generating_frames ---
            job_manager.update_stage(job_id, "generating_frames", 0.25,
                                     label=f"Generating {video_length} frames (step 0/{num_inference_steps})...")
            logger.info(
                "Starting V3 Flash inference: %d frames, %dx%d, cfg=%.1f, acfg=%.1f, steps=%d, seed=%d",
                video_length, resolution, resolution, guidance_scale, audio_guidance_scale,
                num_inference_steps, seed,
            )

            # Progress callback via scheduler wrapping
            step_count = [0]
            original_step = self.pipe.scheduler.step

            def _progress_step(*args, **kwargs):
                result = original_step(*args, **kwargs)
                step_count[0] += 1
                frac = step_count[0] / num_inference_steps
                progress = 0.25 + frac * 0.55  # 0.25 → 0.80
                job_manager.update_stage(
                    job_id, "generating_frames", progress,
                    label=f"Generating frames (step {step_count[0]}/{num_inference_steps})...",
                )
                return result

            self.pipe.scheduler.step = _progress_step
            try:
                with torch.no_grad():
                    sample = self.pipe(
                        prompt,
                        num_frames=video_length,
                        negative_prompt=negative_prompt,
                        audio_embeds=audio_embeds,
                        audio_scale=1.0,
                        ip_mask=None,  # Flash: no face mask needed
                        use_un_ip_mask=False,
                        height=resolution,
                        width=resolution,
                        generator=generator,
                        neg_scale=1.0,
                        neg_steps=0,
                        use_dynamic_cfg=False,
                        use_dynamic_acfg=False,
                        guidance_scale=guidance_scale,
                        audio_guidance_scale=audio_guidance_scale,
                        num_inference_steps=num_inference_steps,
                        video=input_video,
                        mask_video=input_video_mask,
                        clip_image=clip_image,
                        cfg_skip_ratio=0.0,
                        shift=5.0,
                    ).videos
            finally:
                self.pipe.scheduler.step = original_step

            if job_manager.is_cancelled(job_id):
                return

            # --- Stage: assembling_video ---
            job_manager.update_stage(job_id, "assembling_video", 0.85,
                                     label="Assembling video...")

            # Save silent video
            from src.utils import save_videos_grid
            silent_path = str(work_dir / "output_silent.mp4")
            save_videos_grid(
                videos=sample,
                path=silent_path,
                n_rows=1,
                fps=FPS,
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
                final_path, codec="libx264", audio_codec="aac", logger=None,
            )
            video_clip.close()
            audio_clip.close()
            final_clip.close()

            job_manager.update_stage(job_id, "assembling_video", 1.0)

            # Thumbnail
            thumbnail_path = str(output_dir / f"{job_id}_thumb.jpg")
            ref_image.save(thumbnail_path, "JPEG", quality=80)

            video_storage_key, thumb_storage_key = storage_service.mirror_outputs(
                job_id=job_id, video_path=final_path, thumb_path=thumbnail_path,
            )

            # --- Complete ---
            result = GenerationResult(
                video_filename=f"{job_id}.mp4",
                duration_seconds=duration,
                resolution=f"{resolution}x{resolution}",
                seed_used=seed,
                frames_generated=video_length,
                storage_key=video_storage_key,
                thumbnail_storage_key=thumb_storage_key,
            )
            job_manager.complete_job(job_id, result)
            logger.info("Generation complete: %s", job_id)

        except Exception as e:
            logger.exception("Generation failed: %s", job_id)
            job_manager.fail_job(job_id, str(e))

        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Singleton
pipeline = EchoMimicPipeline()
