"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";

import {
  API_BASE,
  cancelJob,
  getHistory,
  getJobStatus,
  getThumbnailUrl,
  getVideoUrl,
  startGeneration,
  uploadAudio,
  uploadImage
} from "../lib/api";
import type { JobStatusResponse, Resolution } from "../lib/api";

type SeedMode = "random" | "fixed";

type CropSettings = {
  zoom: number;
  offsetX: number;
  offsetY: number;
};

type HistoryRun = {
  job_id: string;
  status: string;
  created_at: string;
  result: JobStatusResponse["result"] | null;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function formatSeconds(seconds: number | null | undefined): string {
  if (seconds == null || Number.isNaN(seconds)) {
    return "--";
  }
  const safe = Math.max(0, Math.round(seconds));
  const mins = Math.floor(safe / 60);
  const secs = safe % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = url;
  });
}

async function buildCroppedImage(file: File, crop: CropSettings): Promise<File> {
  const sourceUrl = URL.createObjectURL(file);
  try {
    const image = await loadImage(sourceUrl);
    const canvas = document.createElement("canvas");
    const targetSize = 1024;
    canvas.width = targetSize;
    canvas.height = targetSize;

    const context = canvas.getContext("2d");
    if (!context) {
      throw new Error("Could not create crop context");
    }

    const minDim = Math.min(image.naturalWidth, image.naturalHeight);
    const zoom = clamp(crop.zoom, 1, 3);
    const srcSize = minDim / zoom;

    const maxPanX = Math.max(0, (image.naturalWidth - srcSize) / 2);
    const maxPanY = Math.max(0, (image.naturalHeight - srcSize) / 2);

    const centerX = image.naturalWidth / 2 + (clamp(crop.offsetX, -100, 100) / 100) * maxPanX;
    const centerY = image.naturalHeight / 2 + (clamp(crop.offsetY, -100, 100) / 100) * maxPanY;

    const sx = clamp(centerX - srcSize / 2, 0, image.naturalWidth - srcSize);
    const sy = clamp(centerY - srcSize / 2, 0, image.naturalHeight - srcSize);

    context.fillStyle = "#000";
    context.fillRect(0, 0, targetSize, targetSize);
    context.drawImage(image, sx, sy, srcSize, srcSize, 0, 0, targetSize, targetSize);

    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (b) => {
          if (!b) {
            reject(new Error("Could not build cropped image"));
            return;
          }
          resolve(b);
        },
        "image/png",
        0.95
      );
    });

    return new File([blob], "portrait-cropped.png", { type: "image/png" });
  } finally {
    URL.revokeObjectURL(sourceUrl);
  }
}

export default function Home() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);

  const [crop, setCrop] = useState<CropSettings>({
    zoom: 1.25,
    offsetX: 0,
    offsetY: -12
  });

  const [motionScale, setMotionScale] = useState<number>(5);
  const [resolution, setResolution] = useState<Resolution>(512);
  const [seedMode, setSeedMode] = useState<SeedMode>("random");
  const [seedInput, setSeedInput] = useState<string>("42");
  const [prompt, setPrompt] = useState<string>(
    "A person speaks naturally with gentle head movements and facial expressions"
  );

  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [history, setHistory] = useState<HistoryRun[]>([]);

  const progressPct = Math.round((status?.progress ?? 0) * 100);

  const canGenerate = useMemo(() => {
    if (!imageFile || !audioFile) {
      return false;
    }
    if (seedMode === "fixed") {
      return /^-?\d+$/.test(seedInput.trim());
    }
    return true;
  }, [imageFile, audioFile, seedInput, seedMode]);

  useEffect(() => {
    if (!imageFile) {
      setImagePreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(imageFile);
    setImagePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  useEffect(() => {
    let cancelled = false;

    getHistory(8)
      .then((result) => {
        if (cancelled) {
          return;
        }
        setHistory(
          result.runs.map((run) => ({
            job_id: run.job_id,
            status: run.status,
            created_at: run.created_at,
            result: run.result || null
          }))
        );
      })
      .catch(() => {
        // Non-blocking.
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const next = await getJobStatus(jobId);
        if (cancelled) {
          return;
        }

        setStatus(next);

        if (next.status === "completed" || next.status === "failed" || next.status === "cancelled") {
          const recent = await getHistory(8);
          if (!cancelled) {
            setHistory(
              recent.runs.map((run) => ({
                job_id: run.job_id,
                status: run.status,
                created_at: run.created_at,
                result: run.result || null
              }))
            );
          }
          return;
        }

        window.setTimeout(poll, 2000);
      } catch (e) {
        if (!cancelled) {
          const message = e instanceof Error ? e.message : "Failed to fetch status";
          setError(message);
        }
      }
    };

    poll();

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const onImageChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setImageFile(file);
  };

  const onAudioChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setAudioFile(file);
  };

  const onGenerate = async () => {
    if (!imageFile || !audioFile || !canGenerate) {
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      const croppedImage = await buildCroppedImage(imageFile, crop);

      const uploadedImage = await uploadImage(croppedImage);
      const uploadedAudio = await uploadAudio(audioFile);

      const seed = seedMode === "random" ? -1 : Number.parseInt(seedInput, 10);

      const accepted = await startGeneration({
        image_file_id: uploadedImage.file_id,
        audio_file_id: uploadedAudio.file_id,
        settings: {
          motion_scale: motionScale,
          resolution,
          seed,
          prompt,
        }
      });

      setJobId(accepted.job_id);
      const initialStatus = await getJobStatus(accepted.job_id);
      setStatus(initialStatus);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Generation failed to start";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const onCancel = async () => {
    if (!jobId || !status || (status.status !== "queued" && status.status !== "processing")) {
      return;
    }
    try {
      await cancelJob(jobId);
      const next = await getJobStatus(jobId);
      setStatus(next);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to cancel job";
      setError(message);
    }
  };

  return (
    <main className="shell">
      <div className="backdrop" />

      <header className="hero">
        <p className="eyebrow">StillAlive Studio</p>
        <h1>EchoMimic V3 Flash</h1>
        <p>
          Upload one portrait and one voice track, describe the motion, and render a talking-head video
          powered by EchoMimic V3 Flash on RunPod.
        </p>
        <p className="api-note">API target: {API_BASE}</p>
      </header>

      <section className="workspace-grid">
        <article className="card">
          <h2>1. Upload & Crop</h2>

          <label className="field">
            <span>Portrait (PNG/JPG)</span>
            <input accept="image/png,image/jpeg,image/jpg" type="file" onChange={onImageChange} />
          </label>

          {imagePreviewUrl ? (
            <>
              <div className="crop-stage" aria-label="Crop preview">
                <img
                  alt="Portrait preview"
                  className="crop-image"
                  src={imagePreviewUrl}
                  style={{
                    transform: `translate(${crop.offsetX * 0.7}px, ${crop.offsetY * 0.7}px) scale(${crop.zoom})`
                  }}
                />
                <div className="crop-mask" />
              </div>

              <div className="slider-grid">
                <label>
                  <span>Zoom</span>
                  <input
                    max={3}
                    min={1}
                    onChange={(e) => setCrop((prev) => ({ ...prev, zoom: Number(e.target.value) }))}
                    step={0.01}
                    type="range"
                    value={crop.zoom}
                  />
                </label>
                <label>
                  <span>Horizontal Framing</span>
                  <input
                    max={100}
                    min={-100}
                    onChange={(e) => setCrop((prev) => ({ ...prev, offsetX: Number(e.target.value) }))}
                    step={1}
                    type="range"
                    value={crop.offsetX}
                  />
                </label>
                <label>
                  <span>Vertical Framing</span>
                  <input
                    max={100}
                    min={-100}
                    onChange={(e) => setCrop((prev) => ({ ...prev, offsetY: Number(e.target.value) }))}
                    step={1}
                    type="range"
                    value={crop.offsetY}
                  />
                </label>
              </div>
            </>
          ) : (
            <p className="hint">Add a portrait, then center the head and shoulders before generating.</p>
          )}

          <label className="field">
            <span>Voiceover (MP3/WAV, max 2 min)</span>
            <input accept="audio/mpeg,audio/wav,audio/x-wav,audio/wave" type="file" onChange={onAudioChange} />
          </label>
        </article>

        <article className="card">
          <h2>2. Configure & Generate</h2>

          <label className="field">
            <span>Scene Description</span>
            <textarea
              rows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe how the person should move and speak..."
              style={{ width: "100%", resize: "vertical", fontFamily: "inherit", fontSize: "inherit" }}
            />
          </label>

          <div className="slider-grid">
            <label>
              <span>Motion Scale: {motionScale}</span>
              <input
                max={10}
                min={1}
                onChange={(e) => setMotionScale(Number(e.target.value))}
                type="range"
                value={motionScale}
              />
            </label>
          </div>

          <div className="radio-group">
            <span>Resolution</span>
            <label>
              <input
                checked={resolution === 512}
                name="resolution"
                onChange={() => setResolution(512)}
                type="radio"
                value="512"
              />
              512 × 512 (faster)
            </label>
            <label>
              <input
                checked={resolution === 768}
                name="resolution"
                onChange={() => setResolution(768)}
                type="radio"
                value="768"
              />
              768 × 768 (higher quality)
            </label>
          </div>

          <div className="radio-group">
            <span>Seed</span>
            <label>
              <input
                checked={seedMode === "random"}
                name="seedMode"
                onChange={() => setSeedMode("random")}
                type="radio"
              />
              Random each run
            </label>
            <label>
              <input
                checked={seedMode === "fixed"}
                name="seedMode"
                onChange={() => setSeedMode("fixed")}
                type="radio"
              />
              Fixed seed
            </label>
            <input
              disabled={seedMode !== "fixed"}
              inputMode="numeric"
              onChange={(e) => setSeedInput(e.target.value)}
              placeholder="42"
              type="text"
              value={seedInput}
            />
          </div>

          <div className="actions-row">
            <button className="primary" disabled={!canGenerate || isSubmitting} onClick={onGenerate}>
              {isSubmitting ? "Uploading..." : "Generate Video"}
            </button>
            <button
              className="ghost"
              disabled={!status || (status.status !== "queued" && status.status !== "processing")}
              onClick={onCancel}
              type="button"
            >
              Cancel
            </button>
          </div>

          {error ? <p className="error">{error}</p> : null}

          {status ? (
            <div className="status-box">
              <div className="status-row">
                <strong>Status</strong>
                <span className={`pill ${status.status}`}>{status.status}</span>
              </div>

              {status.queue_position ? <p>Queue position: {status.queue_position}</p> : null}
              <p>{status.stage_label}</p>

              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPct}%` }} />
              </div>
              <div className="status-meta">
                <span>{progressPct}%</span>
                <span>Elapsed: {formatSeconds(status.elapsed_seconds)}</span>
                <span>ETA: {formatSeconds(status.estimated_remaining_seconds)}</span>
              </div>

              {status.status === "completed" && jobId ? (
                <div className="result-stack">
                  <img alt="Result thumbnail" className="thumb" src={getThumbnailUrl(jobId)} />
                  <video className="video" controls src={getVideoUrl(jobId)} />
                  <a className="download-link" href={getVideoUrl(jobId)}>
                    Download MP4
                  </a>
                </div>
              ) : null}

              {status.status === "failed" && status.error ? <p className="error">{status.error}</p> : null}
            </div>
          ) : (
            <p className="hint">No active job yet.</p>
          )}
        </article>
      </section>

      <section className="history card">
        <h2>Recent Runs</h2>
        {history.length === 0 ? (
          <p className="hint">No runs yet.</p>
        ) : (
          <ul>
            {history.map((run) => (
              <li key={run.job_id}>
                <div>
                  <strong>{run.job_id}</strong>
                  <p>{new Date(run.created_at).toLocaleString()}</p>
                </div>
                <span className={`pill ${run.status}`}>{run.status}</span>
                <a href={getVideoUrl(run.job_id)}>Open</a>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
