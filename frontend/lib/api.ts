export type Resolution = 512 | 768;

export interface UploadResponse {
  file_id: string;
  filename: string;
  storage_key?: string | null;
}

export interface GenerateAcceptedResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  queue_position?: number | null;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  stage: string;
  stage_label: string;
  progress: number;
  elapsed_seconds: number;
  estimated_remaining_seconds?: number | null;
  created_at: string;
  completed_at?: string | null;
  queue_position?: number | null;
  result?: {
    video_filename: string;
    duration_seconds: number;
    resolution: string;
    seed_used: number;
    frames_generated: number;
    storage_key?: string | null;
    thumbnail_storage_key?: string | null;
  } | null;
  error?: string | null;
}

export interface HistoryResponse {
  runs: Array<{
    job_id: string;
    status: "queued" | "processing" | "completed" | "failed" | "cancelled";
    created_at: string;
    completed_at?: string | null;
    elapsed_seconds: number;
    thumbnail_filename?: string | null;
    result?: {
      duration_seconds: number;
      resolution: string;
      seed_used: number;
      frames_generated: number;
    } | null;
  }>;
}

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

function parseErrorDetail(payload: unknown): string {
  if (!payload || typeof payload !== "object") {
    return "Request failed";
  }

  const record = payload as Record<string, unknown>;
  const detail = record.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object") {
    const detailRecord = detail as Record<string, unknown>;
    if (typeof detailRecord.message === "string") {
      return detailRecord.message;
    }
    return JSON.stringify(detailRecord);
  }

  if (typeof record.message === "string") {
    return record.message;
  }

  return "Request failed";
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), init);
  if (!response.ok) {
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      // Ignore malformed response body.
    }
    throw new Error(parseErrorDetail(payload));
  }
  return (await response.json()) as T;
}

async function upload(path: string, file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return requestJson<UploadResponse>(path, {
    method: "POST",
    body: form
  });
}

export function uploadImage(file: File): Promise<UploadResponse> {
  return upload("/upload/image", file);
}

export function uploadAudio(file: File): Promise<UploadResponse> {
  return upload("/upload/audio", file);
}

export function startGeneration(args: {
  image_file_id: string;
  audio_file_id: string;
  settings: {
    motion_scale: number;
    seed: number;
    resolution: Resolution;
    preset?: "subtle" | "standard" | "expressive";
  };
}): Promise<GenerateAcceptedResponse> {
  return requestJson<GenerateAcceptedResponse>("/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(args)
  });
}

export function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(`/status/${jobId}`);
}

export function getHistory(limit = 10): Promise<HistoryResponse> {
  return requestJson<HistoryResponse>(`/history?limit=${limit}`);
}

export function cancelJob(jobId: string): Promise<{ status: string; message: string }> {
  return requestJson<{ status: string; message: string }>(`/cancel/${jobId}`, { method: "POST" });
}

export function getVideoUrl(jobId: string): string {
  return apiUrl(`/download/${jobId}`);
}

export function getThumbnailUrl(jobId: string): string {
  return apiUrl(`/thumbnail/${jobId}`);
}

export { API_BASE };
