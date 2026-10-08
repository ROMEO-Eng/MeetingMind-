export type Priority = "High" | "Medium" | "Low";

export interface DecisionItem {
  decision: string;
  context: string;
  source_ids: string[];
  source_excerpt: string | null;
}

export interface TaskItem {
  task: string;
  owner: string | null;
  deadline: string | null;
  priority: Priority | null;
  source_ids: string[];
  source_excerpt: string | null;
}

export interface MeetingAnalysis {
  meeting_id: string;
  source_name: string;
  title: string;
  executive_summary: string;
  decisions: DecisionItem[];
  tasks: TaskItem[];
  key_points: string[];
  open_questions: string[];
  word_count: number;
  estimated_minutes: number;
  chunk_count: number;
  chunks_truncated: boolean;
}

export interface MeetingSource {
  chunk_id: string;
  excerpt: string;
  score: number;
}

export interface QuestionResponse {
  answer: string;
  found: boolean;
  sources: MeetingSource[];
}

export interface ApiError {
  detail?: string;
}

export interface HealthStatus {
  status: "ok" | "degraded";
  api_status: "connected";
  model_status: "ready" | "loading" | "unavailable";
  model_status_detail: string;
  model_ready: boolean;
  gpu_available: boolean;
  model_name: string;
}
