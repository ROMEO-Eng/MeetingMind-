import type { ApiError, HealthStatus, MeetingAnalysis, QuestionResponse } from "@/lib/types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const LONG_REQUEST_MS = 10 * 60 * 1000;

export class ApiRequestError extends Error {
  constructor(
    message: string,
    readonly category: "timeout" | "network" | "http",
    readonly statusCode?: number,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

function isHealthStatus(data: unknown): data is HealthStatus {
  if (typeof data !== "object" || data === null) return false;
  return (
    "api_status" in data &&
    data.api_status === "connected" &&
    "model_status" in data &&
    (data.model_status === "ready" ||
      data.model_status === "loading" ||
      data.model_status === "unavailable") &&
    "model_status_detail" in data &&
    typeof data.model_status_detail === "string" &&
    "gpu_available" in data &&
    typeof data.gpu_available === "boolean" &&
    "status" in data &&
    (data.status === "ok" || data.status === "degraded") &&
    "model_ready" in data &&
    typeof data.model_ready === "boolean" &&
    data.model_ready === (data.model_status === "ready") &&
    "model_name" in data &&
    typeof data.model_name === "string"
  );
}

function safeBackendMessage(detail: unknown): string | null {
  if (
    typeof detail !== "string" ||
    detail.length > 300 ||
    /[\r\n]/.test(detail) ||
    /traceback|stack trace|file ".*", line \d+|exception:/i.test(detail)
  ) {
    return null;
  }
  return detail;
}

async function responseError(response: Response): Promise<ApiRequestError> {
  let message =
    response.status === 503
      ? "The AI model is not available in the current environment."
      : `The backend request failed (${response.status}).`;
  try {
    const data = (await response.json()) as ApiError;
    message = safeBackendMessage(data.detail) ?? message;
  } catch {
    // Keep the status-based message when the server did not return JSON.
  }
  return new ApiRequestError(message, "http", response.status);
}

async function fetchWithTimeout(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: AbortSignal.timeout(LONG_REQUEST_MS),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new ApiRequestError("The request exceeded its time limit.", "timeout");
    }
    throw new ApiRequestError("The MeetingMind backend could not be reached.", "network");
  }
  if (!response.ok) throw await responseError(response);
  return response;
}

export async function fetchHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/api/health`, {
    signal: AbortSignal.timeout(5000),
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Health request failed (${response.status}).`);

  const data: unknown = await response.json();
  if (!isHealthStatus(data)) throw new Error("The backend returned an invalid health status.");
  return data;
}

export async function fetchSample(): Promise<MeetingAnalysis> {
  const response = await fetchWithTimeout("/api/sample", { method: "POST" });
  return (await response.json()) as MeetingAnalysis;
}

export async function analyseMeeting(form: FormData): Promise<MeetingAnalysis> {
  const response = await fetchWithTimeout("/api/analyse", { method: "POST", body: form });
  return (await response.json()) as MeetingAnalysis;
}

export async function askMeeting(meetingId: string, question: string): Promise<QuestionResponse> {
  const response = await fetchWithTimeout("/api/qa", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ meeting_id: meetingId, question }),
  });
  return (await response.json()) as QuestionResponse;
}

export async function downloadNotes(meetingId: string, format: "csv" | "markdown"): Promise<void> {
  const response = await fetchWithTimeout(
    `/api/meetings/${encodeURIComponent(meetingId)}/downloads?format=${format}`,
  );
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = format === "csv" ? "meetingmind-action-items.csv" : "meetingmind-notes.md";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
